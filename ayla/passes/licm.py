"""Invariant code motion.

Hoists side-effect-free ops out of a ``for`` loop body when all their operands
are defined outside the loop. Uninitialized shared-memory allocations are also
hoisted through guards inside loops, letting an enclosing loop reuse one arena
across guarded iterations.
"""

from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    DeallocOp,
    BarrierOp,
    ForOp,
    IfOp,
    ModuleOp,
    SmemTensorType,
    TmemTensorType,
    KernelOp,
    effective_operands,
    is_side_effect_free,
)
from ..base.ir import (
    Op,
    Value,
    Block,
    find_ancestor_of_type,
    find_ancestor,
    find_ops_of_type,
)
from ..base.lowering_pass import LoweringPass, lowers


class LICM(LoweringPass):
    def __init__(self, *, reuse_tmem_allocation: bool = False,
                 hoist_pure_through_guards: bool = False):
        super().__init__(reuse_tmem_allocation=reuse_tmem_allocation,
                         hoist_pure_through_guards=hoist_pure_through_guards)

    def _reusable_lifetime(self, body: Block) -> tuple[list[Op], list[Op]]:
        """Move a sole, nonescaping allocation and its matching release together.

        Initialization stays in the body. Requiring one TMEM allocation in the
        whole kernel avoids increasing simultaneous TMEM capacity requirements.
        Only explicit releases in this same scope may move; conditional frees
        and live-out loop values cannot establish a reusable lifetime.
        """
        owner = body.parent_op
        if not self.reuse_tmem_allocation or owner is None or owner.results:
            return [], []
        kernel = find_ancestor_of_type(owner, KernelOp)
        if kernel is None:
            return [], []
        allocations = [
            op
            for op in find_ops_of_type(kernel, AllocOp)
            if isinstance(op.result().type, TmemTensorType)
        ]
        if len(allocations) != 1 or allocations[0] not in body.ops:
            return [], []
        alloc = allocations[0]
        users = alloc.result().users()
        releases = [op for op in users if isinstance(op, DeallocOp)]
        if (
            len(releases) != 1
            or releases[0] not in body.ops
            or body.ops.index(releases[0]) < body.ops.index(alloc)
            or any(
                user.parent_block is not body
                and find_ancestor(user, lambda candidate: candidate is owner) is None
                for user in users
            )
        ):
            return [], []
        # Releasing TMEM ends the CTA-wide use of the accumulator. Preserve
        # that synchronization at the original lifetime boundary: the next
        # iteration may reinitialize mbarriers and overwrite the same TMEM.
        # Warp-local load completion alone cannot order another warp's reads.
        if releases[0].attrs.get("reuse_boundary_preserved"):
            body.ops = [
                op for op in body.ops if op is not alloc and op is not releases[0]
            ]
        else:
            body.ops = [
                BarrierOp.build(attrs={"resource_lifetime_reuse": True})
                if op is releases[0]
                else op
                for op in body.ops
                if op is not alloc
            ]
            releases[0].attrs["reuse_boundary_preserved"] = True
        self.value_map[alloc.result()] = alloc.result()
        return [alloc], releases

    @lowers(IfOp)
    def lower_if(
        self,
        op: IfOp,
    ) -> tuple[list[Op], list[object]] | None:
        # Conditional allocation hoisting only serves loop storage reuse. Do
        # not lengthen lifetimes across unrelated, top-level guards.
        if find_ancestor_of_type(op, ForOp) is None:
            return None
        allocs, releases = self._reusable_lifetime(op.body)
        self.lower_block(op.body)
        hoisted: list[Op] = []
        remaining: list[Op] = []
        body_defined: set[Value] = set(op.body.arguments)
        for body_op in op.body.ops:
            storage = isinstance(body_op, AllocOp) and isinstance(
                body_op.result().type, SmemTensorType
            )
            pure = (
                self.hoist_pure_through_guards
                and not isinstance(body_op, AllocOp)
                and _can_hoist(body_op, body_defined)
            )
            if storage or pure:
                hoisted.append(body_op)
                for result in body_op.results:
                    self.value_map[result] = result
            else:
                body_defined.update(body_op.results)
                remaining.append(body_op)
        op.body.ops = remaining
        return allocs + hoisted + [op] + releases, []

    @lowers(ForOp)
    def lower_block_op(
        self,
        op: ForOp,
    ) -> tuple[list[Op], list[object]]:
        body = op.body
        allocs, releases = self._reusable_lifetime(body)
        # Defined inside the loop: block arguments (e.g. the iv) plus the
        # results of the body ops we keep.
        body_defined: set[Value] = set(body.arguments)
        hoisted: list[Op] = []
        remaining: list[Op] = []
        for body_op in body.ops:
            if _can_hoist(body_op, body_defined):
                hoisted.append(body_op)
            else:
                body_defined.update(body_op.results)
                remaining.append(body_op)
        body.ops = remaining
        for h in hoisted:
            for r in h.results:
                self.value_map[r] = r
        self.lower_block(op.body)
        return allocs + hoisted + [op] + releases, list(op.results)


def _can_hoist(op: Op, body_defined: set[Value]) -> bool:
    if any(v in body_defined for v in effective_operands(op)):
        return False
    return is_side_effect_free(op)


def licm(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LICM().run(module)
