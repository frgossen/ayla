"""Tests for lowering compatible moves with STMatrix stores."""
import textwrap

from ayla.base.ir import find_ops_of_type
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
    templates = find_ops_of_type(module, PTXTemplateOp)
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
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(
        op.template for op in templates if "stmatrix.sync" in op.template
    )
    assert template.count("stmatrix.sync") == 8


def test_writer_predicate_uses_column_owner_from_tile_chain():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() block_dim=[512, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %c2: index = index_constant() attrs={value: 2}
            %row: index = index_div(%wgx: index, %c2: index)
            %col: index = index_mod(%wgx: index, %c2: index)
            %s: smem<128x128xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %s_rows: smem<64x128xbf16 strides=[128, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<128x128xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %row: index, 0)
            %r: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s_rows: smem<64x128xbf16 strides=[128, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">, %col: index) attrs={writer_wg: 1}
          }
        }"""))

    module = lower_to_ptx_template(module)
    template = next(
        op.template for op in find_ops_of_type(module, PTXTemplateOp)
        if "stmatrix.sync" in op.template
    )
    assert "${b32:operand.writer.val}, 1" in template
    assert "shr.u32 ${b32:st_wg}" not in template


def test_column_owner_also_predicates_scalar_fallback():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%writer: index) block_dim=[512, 1, 1] {
            %s: smem<64x128xbf16> = alloc() attrs={id: 0}
            %r: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x128xbf16>, %r: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">, %writer: index) attrs={writer_wg: 1}
          }
        }"""))

    module = lower_to_ptx_template(module)
    template = "".join(
        op.template for op in find_ops_of_type(module, PTXTemplateOp)
    )
    assert "${b32:operand.writer.val}, 1" in template
    assert "@${pred:st_writer} st.shared.b32" in template


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
    templates = find_ops_of_type(module, PTXTemplateOp)
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
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = "".join(op.template for op in templates)
    assert template.count("st.shared.b32") == 16
    assert "${b32:operand.src.r112}" in template
    assert "${b32:operand.src.r127}" in template
    assert "${b32:operand.src.r111}" not in template
