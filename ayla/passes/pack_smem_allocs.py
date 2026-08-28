"""Reorder static shared-memory allocations without increasing their size."""

from __future__ import annotations

from ..base.ir import Block
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    ModuleOp,
    SmemTensorType,
    physical_layout,
)


def _alignment(op: AllocOp) -> int:
    typ = op.result().type
    assert isinstance(typ, SmemTensorType)
    return 1024 if typ.swizzle is not None and physical_layout(typ) else 128


def _size(op: AllocOp) -> int:
    typ = op.result().type
    assert isinstance(typ, SmemTensorType)
    return typ.num_elements * typ.dtype.byte_size


def _footprint(allocs: list[AllocOp]) -> int:
    offset = 0
    for alloc in allocs:
        alignment = _alignment(alloc)
        offset = (offset + alignment - 1) // alignment * alignment
        offset += _size(alloc)
    return offset


class PackSmemAllocs(Pass):
    """Reorder lower-aligned tensor allocations before the first widest one."""

    def run_block(self, block: Block) -> None:
        allocs = [
            op for op in block.ops
            if isinstance(op, AllocOp)
            and isinstance(op.result().type, SmemTensorType)
        ]
        if len(allocs) < 2:
            super().run_block(block)
            return

        alignments = [_alignment(op) for op in allocs]
        widest = max(alignments)
        first_wide_index = alignments.index(widest)
        first_wide = allocs[first_wide_index]

        # TODO: A general kernel-wide packing pass should use find_ops to
        # collect nested SMEM allocations, safely hoist them to a common block,
        # and optimize their complete layout. This heuristic only reorders
        # allocations that are already direct children of the same block.
        to_move = []
        reordered = []
        for index, (alloc, alignment) in enumerate(
            zip(allocs, alignments, strict=True),
        ):
            if index > first_wide_index and alignment < widest:
                to_move.append(alloc)
            else:
                reordered.append(alloc)

        reordered = to_move + reordered
        if to_move and _footprint(reordered) <= _footprint(allocs):
            for alloc in to_move:
                block.ops.remove(alloc)
            first_wide_index = block.ops.index(first_wide)
            block.ops[first_wide_index:first_wide_index] = to_move
        super().run_block(block)


def pack_smem_allocs(module: ModuleOp) -> ModuleOp:
    return PackSmemAllocs().run(module)
