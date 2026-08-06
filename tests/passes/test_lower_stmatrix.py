"""Tests for lowering compatible moves with STMatrix stores."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import (
    PTXTemplateOp,
    gpu_dialect,
)
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template


def test_lowers_dense_swizzled_move_with_stmatrix():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem_tensor<64x256xbf16 swizzle="128B"> = alloc()
            %r: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            move(%s: smem_tensor<64x256xbf16 swizzle="128B">, %r: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 16
    assert "and.b32 ${b32:st_swizzle_mask}, ${b32:st_addr_off}, 896;" in template


def test_unswizzled_move_uses_scalar_store_fallback():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem_tensor<64x256xbf16> = alloc()
            %r: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            move(%s: smem_tensor<64x256xbf16>, %r: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = "".join(op.template for op in templates)
    assert "stmatrix.sync" not in template
    assert "st.shared.b32" in template
