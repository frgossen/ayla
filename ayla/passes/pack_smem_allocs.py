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
    """Hoist lower-aligned tensor allocations before the first widest one."""

    def run_block(self, block: Block) -> None:
        allocs = [
            op for op in block.ops
            if isinstance(op, AllocOp)
            and isinstance(op.result().type, SmemTensorType)
        ]
        if len(allocs) >= 2:
            widest = max(map(_alignment, allocs))
            first_wide = next(op for op in allocs if _alignment(op) == widest)
            first_wide_index = block.ops.index(first_wide)
            to_hoist = [
                op for op in allocs
                if block.ops.index(op) > first_wide_index
                and _alignment(op) < widest
            ]
            reordered = to_hoist + [op for op in allocs if op not in to_hoist]
            if to_hoist and _footprint(reordered) <= _footprint(allocs):
                for alloc in to_hoist:
                    block.ops.remove(alloc)
                first_wide_index = block.ops.index(first_wide)
                block.ops[first_wide_index:first_wide_index] = to_hoist
        super().run_block(block)


def pack_smem_allocs(module: ModuleOp) -> ModuleOp:
    return PackSmemAllocs().run(module)
