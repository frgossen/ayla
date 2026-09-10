"""Tests for lowering compatible moves with STMatrix stores."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import (
    PTXTemplateOp,
    gpu_dialect,
)
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template


def test_lowers_128b_block_major_move_with_stmatrix():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 16
    assert "and.b32 ${b32:st_swizzle_mask}, ${b32:st_addr_off}, 896;" in template
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in template


def test_lowers_128b_block_major_column_slice():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 1)
            %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %r_slice: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0"> = tile(%r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 1)
            move(%slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r_slice: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 8
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in template
    assert "${b32:operand.src.r32}" in template
    assert "${b32:operand.src.r63}" in template
    assert "${b32:operand.src.r31}" not in template


def test_lowers_legacy_attribute_column_slice():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 1)
            %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={wgmma_col_start: 128}
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 8


def test_unswizzled_move_uses_scalar_store_fallback():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16> = alloc() attrs={id: 0}
            %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x256xbf16>, %r: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = "".join(op.template for op in templates)
    assert "stmatrix.sync" not in template
    assert "st.shared.b32" in template


def test_scalar_store_uses_root_registers_for_nested_wrmem_view():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x64xbf16> = alloc() attrs={id: 0}
            %r: wrmem<128x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %rows: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = tile(%r: wrmem<128x256xbf16 warps=4 layout="wgmma_layout_0">, 1, 0)
            %tail: wrmem<64x64xbf16 warps=4 layout="wgmma_layout_0"> = tile(%rows: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 3)
            move(%s: smem<64x64xbf16>, %tail: wrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    template = "".join(op.template for op in templates)
    assert template.count("st.shared.b32") == 16
    assert "${b32:operand.src.r112}" in template
    assert "${b32:operand.src.r127}" in template
    assert "${b32:operand.src.r111}" not in template
