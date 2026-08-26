"""Peel the first iteration from a loop."""
from __future__ import annotations

from ..base.ir import Op, Value, clone_op
from ..base.lowering_pass import LoweringPass
from ..gpu_dialect import ForOp, IndexConstantOp, ModuleOp


class LoopPeeling(LoweringPass):
    """Replace a selected loop's first iteration with straight-line code."""

    def __init__(
        self,
        loops: set[ForOp] | None = None,
    ) -> None:
        super().__init__()
        self.register(ForOp, type(self).lower_for)
        self.loops = loops
        self.peeled_ops: list[Op] = []

    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        if self.loops is not None and op not in self.loops:
            return None
        if op.trip_count <= 0:
            return None

        self.lower_block(op.body)

        iv = IndexConstantOp.build(value=op.lb)
        vmap: dict[Value, Value] = {op.iv: iv.result()}
        peeled = [clone_op(body_op, vmap) for body_op in op.body.ops]
        self.peeled_ops.extend(peeled)

        result: list[Op] = [iv, *peeled]
        if op.trip_count > 1:
            op.attrs["lb"] = op.lb + op.step
            result.append(op)
        return result, []


def loop_peeling(
    module: ModuleOp,
    loops: set[ForOp] | None = None,
) -> ModuleOp:
    """Convenience entry point."""
    return LoopPeeling(loops).run(module)
