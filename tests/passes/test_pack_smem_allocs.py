"""Tests for shared-memory allocation packing."""

import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    BF16Type,
    KernelOp,
    ModuleOp,
    MoveOp,
    ReinterpretOp,
    SmemTensorType,
    gpu_dialect,
)
from ayla.passes.pack_smem_allocs import pack_smem_allocs, reuse_smem_allocs


def _parse_module(text: str) -> ModuleOp:
    module = gpu_dialect.parse_module(text)
    return module


def test_reorders_lower_aligned_allocs_when_footprint_does_not_grow():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %output: smem<16x32xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={label: "output"}
            %lhs: smem<16x32xbf16> = alloc() attrs={label: "lhs"}
            %rhs: smem<16x32xbf16> = alloc() attrs={label: "rhs"}
          }
        }"""))
    pack_smem_allocs(module)
    [kernel] = [op for op in module.body.ops if isinstance(op, KernelOp)]
    labels = [
        op.attrs["label"] for op in kernel.body.ops if isinstance(op, AllocOp)
    ]
    assert labels == ["lhs", "rhs", "output"]


def test_keeps_order_when_reordering_would_add_padding():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %output: smem<16x32xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={label: "output"}
            %operand: smem<1xbf16> = alloc() attrs={label: "operand"}
          }
        }"""))
    pack_smem_allocs(module)
    [kernel] = [op for op in module.body.ops if isinstance(op, KernelOp)]
    labels = [
        op.attrs["label"] for op in kernel.body.ops if isinstance(op, AllocOp)
    ]
    assert labels == ["output", "operand"]


def test_reuses_storage_between_named_allocation_groups():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %independent: smem<16x16xbf16> = alloc() attrs={label: "shared"}
            %a0: smem<32x64xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={reuse_group: "first"}
            %a1: smem<32x64xbf16> = alloc() attrs={reuse_group: "first"}
            %b: smem<32x64xbf16> = alloc() attrs={reuse_group: "second"}
            move(%b: smem<32x64xbf16>, %b: smem<32x64xbf16>)
          }
        }"""))

    reuse_smem_allocs(module)
    [kernel] = module.body.ops
    allocations = find_ops_of_type(kernel, AllocOp)
    [independent] = [
        op for op in allocations if op.attrs.get("label") == "shared"
    ]
    [arena] = [
        op for op in allocations
        if op is not independent
    ]
    assert arena.result().type == SmemTensorType.build(
        (4096,),
        BF16Type(),
        attrs={
            "swizzle": "128B",
            "physical_layout": {
                "block_width": 64,
                "order": "block_major",
            },
        },
    )
    assert allocations == [arena, independent]
    reinterprets = find_ops_of_type(kernel, ReinterpretOp)
    assert len(reinterprets) == 3
    assert all(op.arg is arena.result() for op in reinterprets)
    assert [op.byte_offset for op in reinterprets] == [0, 4096, 0]
    second = next(
        op for op in reinterprets
        if op.result().users()
    )
    assert second.byte_offset == 0
    [move] = find_ops_of_type(kernel, MoveOp)
    assert move.dst is second.result()
    assert move.src is second.result()
    assert all("reuse_group" not in op.attrs for op in allocations)
