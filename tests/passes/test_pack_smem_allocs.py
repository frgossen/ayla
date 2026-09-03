"""Tests for shared-memory allocation packing."""

import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import (
    AllocOp,
    BF16Type,
    KernelOp,
    MoveOp,
    ReinterpretOp,
    SmemTensorType,
    gpu_dialect,
)
from ayla.passes.pack_smem_allocs import pack_smem_allocs, reuse_smem_allocs


def test_reorders_lower_aligned_allocs_when_footprint_does_not_grow():
    module = gpu_dialect.parse(textwrap.dedent("""\
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
    module = gpu_dialect.parse(textwrap.dedent("""\
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
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %persistent: smem<16x16xbf16> = alloc() attrs={label: "shared"}
            %a: smem<64x64xbf16> = alloc() attrs={reuse_group: "first"}
            %b: smem<32x64xbf16> = alloc() attrs={reuse_group: "second"}
            move(%b: smem<32x64xbf16>, %b: smem<32x64xbf16>)
          }
        }"""))

    reuse_smem_allocs(module)
    [kernel] = module.body.ops
    allocations = find_ops(kernel, lambda op: isinstance(op, AllocOp))
    [persistent] = [
        op for op in allocations if op.attrs.get("label") == "shared"
    ]
    [owner] = [
        op for op in allocations
        if op.attrs.get("reuse_group") == "first"
    ]
    assert owner.attrs["allocation_bytes"] == 8192
    assert allocations == [persistent, owner]
    [second] = find_ops(kernel, lambda op: isinstance(op, ReinterpretOp))
    assert second.storage is owner.result()
    assert second.byte_offset == 0
    assert second.result().type == SmemTensorType.build((32, 64), BF16Type())
    [move] = find_ops(kernel, lambda op: isinstance(op, MoveOp))
    assert move.dst is second.result()
    assert move.src is second.result()
    assert {op.attrs.get("reuse_group") for op in allocations} == {
        None, "first",
    }
