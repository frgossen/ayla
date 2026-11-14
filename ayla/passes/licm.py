"""Invariant code motion.

Hoists side-effect-free ops out of a ``for`` loop body when all their operands
are defined outside the loop. Uninitialized shared-memory allocations are also
hoisted through guards inside loops, letting an enclosing loop reuse one arena
across guarded iterations.
"""

from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    DeallocOp,
    BarrierOp,
    ForOp,
    IfOp,
    ModuleOp,
    SmemTensorType,
    TmemTensorType,
    KernelOp,
    LockType,
    execution_issuer_warp,
    lock_has_tma_multicast,
    effective_operands,
    is_side_effect_free,
)
from ..analysis.index_proofs import uniform_guard_scope
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
                 reuse_completion_locks: bool = False,
                 hoist_pure_through_guards: bool = False):
        super().__init__(reuse_tmem_allocation=reuse_tmem_allocation,
                         reuse_completion_locks=reuse_completion_locks,
                         hoist_pure_through_guards=hoist_pure_through_guards)

    def _reusable_completion_locks(self, body: Block) -> list[AllocOp]:
        """Carry a private event's phase across balanced, uniform iterations.

        Each execution of the allocation scope must issue and observe every
        phase before returning. Initialization then belongs to the enclosing
        lifetime, while awaits continue to advance the event's per-thread phase.
        """
        owner = body.parent_op
        kernel = find_ancestor_of_type(owner, KernelOp) if owner else None
        module = find_ancestor_of_type(owner, ModuleOp) if owner else None
        if (not self.reuse_completion_locks or owner is None or owner.results
                or kernel is None or module is None or module.target != 'sm_100a'
                or kernel.is_clustered):
            return []
        if isinstance(owner, IfOp) and not _block_uniform(owner):
            return []
        hoisted = []
        for alloc in body.ops:
            if not isinstance(alloc, AllocOp):
                continue
            lock = alloc.result()
            if (not isinstance(lock.type, LockType)
                    or not lock.type.is_event
                    or alloc.attrs.get('reuse_group') is not None
                    or lock_has_tma_multicast(lock)):
                continue
            users = lock.users()
            if (not users or any(not isinstance(user, (AsyncOp, AwaitOp)) for user in users)
                    or any(find_ancestor(user, lambda parent: parent is owner) is None for user in users)):
                continue
            def site(user):
                while user.parent_block is not body:
                    assert user.parent_block is not None
                    user = user.parent_block.parent_op
                    assert user is not None
                return user
            if any(body.ops.index(site(user)) < body.ops.index(alloc) for user in users):
                continue
            issuers = {execution_issuer_warp(user) for user in users if isinstance(user, AsyncOp)}
            if len(issuers) != 1:
                continue
            if any(user.attrs.get('pending', 0) for user in users if isinstance(user, AwaitOp)):
                continue
            trace = _phase_trace(body, lock)
            if trace is None or trace[0] != 0 or trace[1] < 0 or trace[2] != 1:
                continue
            hoisted.append(alloc)
            self.value_map[lock] = lock
        body.ops = [op for op in body.ops if op not in hoisted]
        if hoisted:
            # Initialized resources require every observer to finish the
            # enclosing lifetime before its next execution. Preserve that
            # handoff just as TMEM lifetime motion preserves its release.
            if not isinstance(body.ops[-1], BarrierOp) or body.ops[-1].scope != 'block':
                body.ops.append(BarrierOp.build(attrs={'completion_lifetime_reuse': True}))
        return hoisted

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
        events = self._reusable_completion_locks(op.body)
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
        return allocs + events + hoisted + [op] + releases, []

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


def _block_uniform(op: IfOp) -> bool:
    return uniform_guard_scope(op.condition, find_ancestor_of_type(op, KernelOp)) is not None


def _phase_trace(block: Block, lock: Value) -> tuple[int, int, int] | None:
    """Net issued phases and the minimum/maximum outstanding on every path."""
    total = low = high = 0
    for op in block.ops:
        if isinstance(op, AsyncOp) and lock in op.completion_locks:
            part = (1, 0, 1)
        elif isinstance(op, AwaitOp) and op.lock is lock:
            part = (-1, -1, 0)
        elif isinstance(op, (ForOp, IfOp)):
            part = _phase_trace(op.body, lock)
            if part is None:
                return None
            delta, minimum, maximum = part
            if isinstance(op, IfOp):
                if delta or (part != (0, 0, 0) and not _block_uniform(op)):
                    return None
            elif (count := op.trip_count) == 0:
                part = (0, 0, 0)
            else:
                part = (delta * count, minimum + min(0, (count-1)*delta),
                        maximum + max(0, (count-1)*delta))
        else:
            if op.blocks and any(find_ancestor(user, lambda parent: parent is op) is not None
                                 for user in lock.users()):
                return None
            continue
        delta, minimum, maximum = part
        low = min(low, total + minimum)
        high = max(high, total + maximum)
        total += delta
    return total, low, high


def licm(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LICM().run(module)
