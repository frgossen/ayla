"""Loop-invariant code motion.

Hoists side-effect-free ops out of a ``for`` loop body when all their operands
are defined outside the region. The same rule applies to ``if`` bodies, allowing
invariant storage to leave guards inside a loop.
Initialized register allocations are split into invariant storage and a scalar
constant, with a broadcasting move at the original position in the loop.
"""
from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    CollectiveAxis,
    CollectiveLevel,
    ConstantOp,
    ForOp,
    IfOp,
    ModuleOp,
    MoveOp,
    RegMemTensorType,
    RmemTensorType,
    effective_operands,
    is_side_effect_free,
)
from ..base.ir import Op, Value, find_ancestor_of_type
from ..base.lowering_pass import LoweringPass, lowers


class LICM(LoweringPass):

    @lowers(AllocOp)
    def lower_alloc(self, op: AllocOp) -> tuple[list[Op], list[object]]:
        typ = op.result().type
        if (
            op.init is not None
            and isinstance(typ, RegMemTensorType)
            and find_ancestor_of_type(op, ForOp) is not None
        ):
            constant = ConstantOp.build(op.init, RmemTensorType.build([], typ.dtype))
            reset = MoveOp.build(
                op.result(), constant.result(),
                collective={axis: CollectiveLevel.THREAD for axis in CollectiveAxis},
            )
            del op.attrs["init"]
            return [op, constant, reset], list(op.results)
        return [op], list(op.results)

    @lowers(ForOp, IfOp)
    def lower_block_op(
        self, op: ForOp | IfOp,
    ) -> tuple[list[Op], list[object]]:
        body = op.body
        # Nested allocations can leave their enclosing loop in the same pass.
        self.lower_block(body)
        # Defined inside the region: block arguments (e.g. the iv) plus the
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
        return hoisted + [op], list(op.results)


def _can_hoist(op: Op, body_defined: set[Value]) -> bool:
    if any(v in body_defined for v in effective_operands(op)):
        return False
    return is_side_effect_free(op)


def licm(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LICM().run(module)
