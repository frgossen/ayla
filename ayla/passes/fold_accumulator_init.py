"""Fold a zero accumulator fill into the first loop-carried matmul.

WGMMA can either overwrite or accumulate its destination.  When an accumulator
is zero-filled immediately before a loop, use the loop IV as the first matmul's
uniform ``scale_d`` condition instead: iteration zero overwrites, later
iterations accumulate.  This removes a register-wide runtime fill without
peeling or duplicating the loop body.
"""
from __future__ import annotations

from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass
from ..gpu_dialect import AllocOp, ForOp, MatmulOp, ModuleOp, WrmemTensorType


def _uses(op: Op, value: Value) -> bool:
    if value in op.operands:
        return True
    return any(
        _uses(nested, value)
        for region in op.regions
        for block in region.blocks
        for nested in block.ops
    )


def _first_matmul(op: Op, dst: Value) -> MatmulOp | None:
    if isinstance(op, MatmulOp) and op.dst is dst:
        return op
    # A nested loop would execute the candidate more than once while the outer
    # IV is still zero, repeatedly overwriting instead of accumulating.
    if isinstance(op, ForOp):
        return None
    for region in op.regions:
        for block in region.blocks:
            for nested in block.ops:
                found = _first_matmul(nested, dst)
                if found is not None:
                    return found
    return None


class FoldAccumulatorInit(Pass):

    def run_block(self, block: Block) -> None:
        for index, alloc in enumerate(block.ops):
            if not (
                isinstance(alloc, AllocOp)
                and isinstance(alloc.result().type, WrmemTensorType)
                and alloc.init == 0
            ):
                continue
            accumulator = alloc.result()
            first_user = next(
                (op for op in block.ops[index + 1:] if _uses(op, accumulator)),
                None,
            )
            if not (
                isinstance(first_user, ForOp)
                and first_user.lb == 0
                and first_user.trip_count > 0
            ):
                continue
            first_matmul = next(
                (
                    found
                    for op in first_user.body.entry_block.ops
                    if (found := _first_matmul(op, accumulator)) is not None
                ),
                None,
            )
            if first_matmul is None or first_matmul.scale_d is not None:
                continue
            first_matmul.operands.append(first_user.iv)
            alloc.attrs.pop("init")
        super().run_block(block)


def fold_accumulator_init(module: ModuleOp) -> ModuleOp:
    return FoldAccumulatorInit().run(module)
