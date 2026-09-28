"""Tests for lowering compatible moves with STMatrix stores."""
import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    PTXTemplateOp,
    gpu_dialect,
)
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template


def test_lowers_128b_block_major_move_with_stmatrix():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 16
    assert "and.b32 ${b32:st_swizzle_mask}, ${b32:st_addr_off}, 896;" in template
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in template


def test_lowers_128b_block_major_column_slice():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 1)
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %r_slice: wgrmem<64x128xbf16 warps=4 layout="wgmma_layout_0"> = tile(%r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 1)
            move(%slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r_slice: wgrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 8
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in template
    assert "${b32:operand.src.r32}" in template
    assert "${b32:operand.src.r63}" in template
    assert "${b32:operand.src.r31}" not in template


def test_lowers_legacy_attribute_column_slice():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 1)
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={wgmma_col_start: 128}
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 8


def test_unswizzled_move_uses_scalar_store_fallback():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x256xbf16>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = "".join(op.template for op in templates)
    assert "stmatrix.sync" not in template
    assert "st.shared.b32" in template


def test_n8_fragment_uses_scalar_store_fallback():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x40xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x40xbf16>, %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))  # noqa: E501

    templates = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )
    template = "".join(op.template for op in templates)
    assert "stmatrix.sync" not in template
    assert template.count("st.shared.b32") == 10
    assert "${b32:operand.src.r9}" in template


def test_n8_fragment_scalar_store_supports_global_memory():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k(%g: gmem<64x40xbf16>) block_dim=[128, 1, 1] {
            %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            move(%g: gmem<64x40xbf16>, %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))  # noqa: E501

    templates = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )
    template = "".join(op.template for op in templates)
    assert template.count("st.global.b32") == 10
    assert "${b32:operand.src.r9}" in template


def test_scalar_store_uses_root_registers_for_nested_wgrmem_view():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x64xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<128x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %rows: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = tile(%r: wgrmem<128x256xbf16 warps=4 layout="wgmma_layout_0">, 1, 0)
            %tail: wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0"> = tile(%rows: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 3)
            move(%s: smem<64x64xbf16>, %tail: wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = "".join(op.template for op in templates)
    assert template.count("st.shared.b32") == 16
    assert "${b32:operand.src.r112}" in template
    assert "${b32:operand.src.r127}" in template
    assert "${b32:operand.src.r111}" not in template
