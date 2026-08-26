"""Fold a zeroed WGMMA accumulator into its first static matmul."""
from __future__ import annotations

from ..base.ir import Op, iter_ops
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    ForOp,
    MatmulOp,
    ModuleOp,
    WrmemTensorType,
    root_buffer,
)
from .loop_peeling import LoopPeeling


def _zeroed_wgmma_accumulator(op: Op):
    if not (
        isinstance(op, MatmulOp)
        and op.init is None
        and isinstance(op.dst.type, WrmemTensorType)
        and op.dst.type.layout == "wgmma_layout_0"
    ):
        return None
    accumulator = root_buffer(op.dst)
    alloc = accumulator.definition
    return alloc if isinstance(alloc, AllocOp) and alloc.init == 0 else None


def _enclosing_loop(op: Op) -> ForOp | None:
    while op.parent_block is not None:
        parent = op.parent_block.parent_op
        if parent is None:
            return None
        if isinstance(parent, ForOp):
            return parent
        op = parent
    return None


class FoldWgmmaAccumulatorInit(Pass):
    """Use WGMMA scale-D to initialize a peeled accumulator.

    Run this after peeling the loop containing the accumulator's first
    matmul. The walk encounters that static matmul before the residual loop.
    """

    def run(self, module: ModuleOp) -> ModuleOp:
        loops = set()
        seen_allocs = set()
        for op in iter_ops(module):
            alloc = _zeroed_wgmma_accumulator(op)
            if alloc is None or alloc in seen_allocs:
                continue
            seen_allocs.add(alloc)
            # Use parent links rather than labels: earlier transforms may
            # rename the loop containing an accumulator's first WGMMA.
            loop = _enclosing_loop(op)
            if loop is not None:
                loops.add(loop)

        peeling = LoopPeeling(loops)
        module = peeling.run(module)
        for op in peeling.peeled_ops:
            self.run_op(op)
        return module

    def run_op(self, op: Op) -> None:
        if isinstance(op, MatmulOp):
            alloc = _zeroed_wgmma_accumulator(op)
            if alloc is not None:
                op.attrs["init"] = alloc.attrs.pop("init")
        super().run_op(op)


def fold_wgmma_accumulator_init(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return FoldWgmmaAccumulatorInit().run(module)
