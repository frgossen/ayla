"""Invariant code motion.

Hoists side-effect-free ops out of a ``for`` loop body when all their operands
are defined outside the loop. Uninitialized shared-memory allocations are also
hoisted through guards inside loops, letting an enclosing loop reuse one arena
across guarded iterations.
"""

from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    ForOp,
    IfOp,
    ModuleOp,
    SmemTensorType,
    effective_operands,
    is_side_effect_free,
)
from ..base.ir import Op, Value, find_ancestor_of_type
from ..base.lowering_pass import LoweringPass, lowers


class LICM(LoweringPass):
    @lowers(IfOp)
    def lower_if(
        self,
        op: IfOp,
    ) -> tuple[list[Op], list[object]] | None:
        # Conditional allocation hoisting only serves loop storage reuse. Do
        # not lengthen lifetimes across unrelated, top-level guards.
        if find_ancestor_of_type(op, ForOp) is None:
            return None
        self.lower_block(op.body)
        hoisted: list[Op] = []
        remaining: list[Op] = []
        for body_op in op.body.ops:
            if isinstance(body_op, AllocOp) and isinstance(
                body_op.result().type, SmemTensorType
            ):
                hoisted.append(body_op)
            else:
                remaining.append(body_op)
        op.body.ops = remaining
        return hoisted + [op], []

    @lowers(ForOp)
    def lower_block_op(
        self,
        op: ForOp,
    ) -> tuple[list[Op], list[object]]:
        body = op.body
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
        return hoisted + [op], list(op.results)


def _can_hoist(op: Op, body_defined: set[Value]) -> bool:
    if any(v in body_defined for v in effective_operands(op)):
        return False
    return is_side_effect_free(op)


def licm(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LICM().run(module)
