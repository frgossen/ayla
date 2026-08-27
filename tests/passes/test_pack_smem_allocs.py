"""Tests for shared-memory allocation packing."""

import textwrap

from ayla.gpu_dialect import AllocOp, KernelOp, gpu_dialect
from ayla.passes.pack_smem_allocs import pack_smem_allocs


def test_hoists_smaller_alignment_allocs_when_footprint_does_not_grow():
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


def test_keeps_order_when_hoisting_would_add_padding():
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
