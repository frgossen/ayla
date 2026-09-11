"""Reorder static shared-memory allocations without increasing their size."""

from __future__ import annotations

from ..base.ir import Block, find_ops
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    KernelOp,
    ModuleOp,
    ReinterpretOp,
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


def _layout(allocs: list[AllocOp], start: int = 0) -> tuple[list[int], int]:
    offsets = []
    offset = start
    for alloc in allocs:
        alignment = _alignment(alloc)
        offset = (offset + alignment - 1) // alignment * alignment
        offsets.append(offset)
        offset += _size(alloc)
    return offsets, offset


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
        if to_move:
            _, reordered_bytes = _layout(reordered)
            _, original_bytes = _layout(allocs)
            if reordered_bytes <= original_bytes:
                for alloc in to_move:
                    block.ops.remove(alloc)
                first_wide_index = block.ops.index(first_wide)
                block.ops[first_wide_index:first_wide_index] = to_move
        super().run_block(block)


def pack_smem_allocs(module: ModuleOp) -> ModuleOp:
    return PackSmemAllocs().run(module)


class ReuseSmemAllocs(Pass):
    """Overlay SMEM allocations carrying distinct ``reuse_group`` values.

    Each ``reuse_group`` is an unsafe caller-provided lifetime assertion:
    lifetimes from distinct named groups do not overlap. Lifetimes within one
    group may overlap, so that group's allocations receive disjoint offsets.
    Untagged allocations remain independent. The motivating use is a phased
    GEMM-to-GEMM fusion whose schedules establish those lifetime boundaries.

    TODO: Infer allocation lifetimes and reuse compatible storage without a
    caller assertion.
    """

    def run(self, module: ModuleOp) -> ModuleOp:
        [kernel] = module.body.ops
        if not isinstance(kernel, KernelOp):
            raise ValueError("SMEM reuse requires a single-kernel module")
        allocations = find_ops(
            kernel,
            lambda op: isinstance(op, AllocOp)
            and isinstance(op.result().type, SmemTensorType),
        )
        groups: dict[str, list[AllocOp]] = {}
        base: AllocOp | None = None
        arena_alignment = 1
        for alloc in allocations:
            group = alloc.attrs.get("reuse_group")
            if group is None:
                continue
            if not isinstance(group, str):
                raise ValueError("reuse_group must be a string")
            if (
                "allocation_bytes" in alloc.attrs
                or "allocation_alignment" in alloc.attrs
            ):
                raise ValueError(
                    "cannot reuse an allocation that already has a layout"
                )
            groups.setdefault(group, []).append(alloc)
            if base is None:
                base = alloc
            arena_alignment = max(arena_alignment, _alignment(alloc))

        if not groups:
            return module
        if len(groups) < 2:
            raise ValueError("at least two reuse groups are required")

        # Build the complete placement plan before mutating the IR.
        placements: list[tuple[AllocOp, int]] = []
        arena_bytes = 0
        for group in groups.values():
            offsets, group_bytes = _layout(group)
            placements.extend(zip(group, offsets, strict=True))
            arena_bytes = max(arena_bytes, group_bytes)

        assert base is not None
        if base.parent_block is None:
            raise ValueError("reuse-group allocation is detached from the kernel")

        # The owner is a physical allocation; every other selected allocation
        # becomes a typed view at its planned offset.
        base.attrs["allocation_bytes"] = arena_bytes
        base.attrs["allocation_alignment"] = arena_alignment
        if base.parent_block is not kernel.body:
            base.parent_block.ops.remove(base)
            kernel.body.ops.insert(0, base)
        for alloc, offset in placements:
            if alloc is base:
                continue
            assert alloc.parent_block is not None
            block = alloc.parent_block
            reinterpret = ReinterpretOp.build(
                base.result(), alloc.result().type, byte_offset=offset,
            )
            for user in alloc.result().users():
                for i, operand in enumerate(user.operands):
                    if operand is alloc.result():
                        user.operands[i] = reinterpret.result()
            block.ops[block.ops.index(alloc)] = reinterpret
        return module


def reuse_smem_allocs(module: ModuleOp) -> ModuleOp:
    return ReuseSmemAllocs().run(module)
