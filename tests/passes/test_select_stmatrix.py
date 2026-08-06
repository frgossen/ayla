"""Tests for layout-driven STMatrix instruction selection."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import (
    StMatrixStoreOp,
    gpu_dialect,
)
from ayla.passes.select_stmatrix import select_stmatrix


def test_selects_stmatrix_from_compatible_physical_layout():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem_tensor<64x256xbf16 strides=[272, 1] attrs={physical_layout: {block: 128, kind: "blocked_padding", padding: 8}}> = alloc()
            %r: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            move(%s: smem_tensor<64x256xbf16 strides=[272, 1] attrs={physical_layout: {block: 128, kind: "blocked_padding", padding: 8}}>, %r: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = select_stmatrix(module)
    stores = find_ops(module, lambda op: isinstance(op, StMatrixStoreOp))
    assert len(stores) == 1
    assert len(stores[0].offsets) == 16
