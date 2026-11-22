"""Tests for lowering compatible moves with STMatrix stores."""

import textwrap

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    PTXTemplateOp,
    gpu_dialect,
)
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("half", [0, 1])
def test_owned_warp_store_selects_columns_from_a_larger_packed_root(dtype, half):
    module = gpu_dialect.parse_module(
        textwrap.dedent(f"""\
        module {{
          kernel @k() block_dim=[128, 1, 1] {{
            %s: smem<8x56x{dtype}> = alloc() attrs={{id: 0}}
            %r: wgrmem<64x168x{dtype} warps=4 layout="wgmma_layout_0"> = alloc() attrs={{id: 1}}
            %v: wgrmem<8x56x{dtype} warps=1 layout="wgmma_warp"> = tile(%r: wgrmem<64x168x{dtype} warps=4 layout="wgmma_layout_0">, {half}, 1) attrs={{index_steps: [8, 56]}}
            move(%s: smem<8x56x{dtype}>, %v: wgrmem<8x56x{dtype} warps=1 layout="wgmma_warp">)
          }}
        }}""")
    )
    template = "".join(op.template for op in find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp
    ))
    assert template.count("stmatrix.sync.aligned.m8n8.x4") == 1
    assert template.count("stmatrix.sync.aligned.m8n8.x2") == 1
    assert template.count("stmatrix.sync.aligned.m8n8.x1") == 1
    # N56 starts halfway through a WGMMA N16 block. Selection must use
    # the owned warp's packed pairs rather than whole-warpgroup groups.
    first = 14 + half
    assert f"${{b32:operand.src.r{first}}}" in template
    assert f"${{b32:operand.src.r{first - 2}}}" not in template


def test_lowers_128b_block_major_move_with_stmatrix():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")
    )

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(op.template for op in templates if "stmatrix.sync" in op.template)
    assert template.count("stmatrix.sync") == 16
    assert "and.b32 ${b32:st_swizzle_mask}, ${b32:st_addr_off}, 896;" in template
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in template


def test_lowers_128b_block_major_column_slice():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 1)
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %r_slice: wgrmem<64x128xbf16 warps=4 layout="wgmma_layout_0"> = tile(%r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 1)
            move(%slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r_slice: wgrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")
    )

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(op.template for op in templates if "stmatrix.sync" in op.template)
    assert template.count("stmatrix.sync") == 8
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in template
    assert "${b32:operand.src.r32}" in template
    assert "${b32:operand.src.r63}" in template
    assert "${b32:operand.src.r31}" not in template


def test_lowers_legacy_attribute_column_slice():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%s: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 1)
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%slice: smem<64x128xbf16 strides=[256, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={wgmma_col_start: 128}
          }
        }""")
    )

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = next(op.template for op in templates if "stmatrix.sync" in op.template)
    assert template.count("stmatrix.sync") == 8


def test_unswizzled_move_uses_native_matrix_stores():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x256xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x256xbf16>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")
    )

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = "".join(op.template for op in templates)
    assert template.count("stmatrix.sync.aligned.m8n8.x4") == 16
    assert "st.shared.b32" not in template
    assert "${b32:st_row}, 256" in template
    assert "st_swizzle_mask" not in template


def test_unswizzled_fragment_tail_uses_x1_matrix_stores():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x40xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x40xbf16>, %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")
    )  # noqa: E501

    templates = find_ops_of_type(
        lower_to_ptx_template(module),
        PTXTemplateOp,
    )
    template = "".join(op.template for op in templates)
    assert template.count("stmatrix.sync.aligned.m8n8.x4") == 2
    assert template.count("stmatrix.sync.aligned.m8n8.x1") == 2
    assert "st.shared.b32" not in template
    assert "${b32:operand.src.r9}" in template


def test_n8_fragment_scalar_store_supports_global_memory():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k(%g: gmem<64x40xbf16>) block_dim=[128, 1, 1] {
            %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            move(%g: gmem<64x40xbf16>, %r: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")
    )  # noqa: E501

    templates = find_ops_of_type(
        lower_to_ptx_template(module),
        PTXTemplateOp,
    )
    template = "".join(op.template for op in templates)
    assert template.count("st.global.b32") == 10
    assert "${b32:operand.fragment.r9}" in template


def test_scalar_store_uses_root_registers_for_nested_wgrmem_view():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[128, 1, 1] {
            %s: smem<64x64xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<128x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %rows: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = tile(%r: wgrmem<128x256xbf16 warps=4 layout="wgmma_layout_0">, 1, 0)
            %tail: wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0"> = tile(%rows: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 3)
            move(%s: smem<64x64xbf16>, %tail: wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")
    )

    module = lower_to_ptx_template(module)
    templates = find_ops_of_type(module, PTXTemplateOp)
    template = "".join(op.template for op in templates)
    assert template.count("st.shared.b32") == 16
    assert "${b32:operand.fragment.r112}" in template
    assert "${b32:operand.fragment.r127}" in template
    assert "${b32:operand.fragment.r111}" not in template


def test_matrix_gather_honors_writer_warpgroup():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k() block_dim=[256, 1, 1] {
            %s: smem<64x128xbf16> = alloc() attrs={id: 0}
            %r: wgrmem<64x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            move(%s: smem<64x128xbf16>, %r: wgrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">) attrs={writer_wg: 1}
          }
        }""")
    )
    templates = find_ops_of_type(lower_to_ptx_template(module), PTXTemplateOp)
    template = "".join(op.template for op in templates)
    stores = [line for line in template.splitlines() if "stmatrix.sync" in line]
    assert len(stores) == 8
    assert all(line.startswith("@${pred:st_writer} ") for line in stores)
    assert "${b32:st_wg}, 1;" in template


@pytest.mark.parametrize(
    ("dtype", "strides", "swizzle", "width"),
    [
        ("f32", [1, 64], "128B", 64),
        ("bf16", [1, 64], None, 64),
        ("bf16", [1, 64], "64B", 64),
        ("bf16", [1, 64], "128B", 32),
        ("bf16", [1, 80], "128B", 64),
    ],
)
def test_unsupported_transposed_store_fails_verification(
    dtype,
    strides,
    swizzle,
    width,
):
    swizzle_attr = f' swizzle="{swizzle}"' if swizzle else ""
    smem = (
        f"smem<64x16x{dtype} strides={strides}{swizzle_attr} "
        f'attrs={{physical_layout: {{block_width: {width}, order: "block_major"}}}}>'
    )
    fragment = f'wgrmem<64x16x{dtype} warps=4 layout="wgmma_layout_0">'
    module = gpu_dialect.parse_module(
        textwrap.dedent(f"""\
        module {{
          kernel @k() block_dim=[128, 1, 1] {{
            %s: {smem} = alloc() attrs={{id: 0}}
            %r: {fragment} = alloc() attrs={{id: 1}}
            move(%s: {smem}, %r: {fragment})
          }}
        }}""")
    )
    with pytest.raises(ValueError, match="transposed WGMMA fragment store"):
        verify_gpu_dialect(module)


def test_transposed_store_groups_trailing_pairs_for_each_m_tile():
    smem = (
        'smem<128x104xbf16 strides=[1, 128] swizzle="128B" '
        'attrs={physical_layout: {block_width: 64, order: "block_major"}}>'
    )
    fragment = 'wgrmem<128x104xbf16 warps=4 layout="wgmma_layout_0">'
    module = gpu_dialect.parse_module(
        textwrap.dedent(f"""\
        module {{
          kernel @k() block_dim=[128, 1, 1] {{
            %s: {smem} = alloc() attrs={{id: 0}}
            %r: {fragment} = alloc() attrs={{id: 1}}
            move(%s: {smem}, %r: {fragment})
          }}
        }}""")
    )
    templates = find_ops_of_type(lower_to_ptx_template(module), PTXTemplateOp)
    template = next(op.template for op in templates if "stmatrix.sync" in op.template)
    assert template.count("stmatrix.sync.aligned.m8n8.x4.trans") == 12
    assert template.count("stmatrix.sync.aligned.m8n8.x2.trans") == 2
    for register in range(52):
        assert template.count(f"${{b32:operand.src.r{register}}}") == 1


def test_stmatrix_preserves_typed_layout_inside_shared_arena():
    module = gpu_dialect.parse_module("""module {
      kernel @arena() block_dim=[128, 1, 1] {
        %storage: smem<33024xbf16> = alloc() attrs={id: 0}
        %matrix: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = reinterpret(%storage: smem<33024xbf16>) attrs={byte_offset: 128}
        %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
        move(%matrix: smem<64x256xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, %r: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
      }
    }""")
    verify_gpu_dialect(module)
    lowered = lower_to_ptx_template(module)
    templates = find_ops_of_type(lowered, PTXTemplateOp)
    store = next(op.template for op in templates if "stmatrix.sync" in op.template)
    assert store.count("stmatrix.sync") == 16
    assert "mad.lo.u32 ${b32:st_off}, ${b32:st_row}, 64" in store
