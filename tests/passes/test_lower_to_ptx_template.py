"""Tests for the lower_to_ptx_template pass."""

import textwrap

import pytest

from ayla.base.ir import find_ops, find_ops_of_type
from ayla.gpu_dialect import (
    IndexConstantOp,
    PtrAddOp,
    PTXTemplateOp,
    ReinterpretOp,
    ReshapeOp,
    ToPtrOp,
    gpu_dialect,
)
from ayla.passes.folding import Folding
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from tests.testing_utils import assert_contains_in_order


def _lower_barrier(scope: str) -> str:
    """Lower a single ``barrier`` op of the given scope and return its PTX."""
    num_threads = ", num_threads: 128" if scope == "block_leader" else ""
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            barrier() attrs={{scope: "{scope}"{num_threads}}}
          }}
        }}"""))
    lowered = lower_to_ptx_template(module)
    return "".join(
        t.attrs["template"]
        for t in find_ops_of_type(lowered, PTXTemplateOp)
    )


@pytest.mark.parametrize("scope, expected", [
    ("thread", ""),                # program order, no barrier
    ("block", "bar.sync 0;\n"),    # full-CTA barrier
    ("cluster", textwrap.dedent("""\
        barrier.cluster.arrive.release;
        barrier.cluster.wait.acquire;
    """)),
])
def test_lower_barrier(scope, expected):
    assert _lower_barrier(scope) == expected


def test_lower_barrier_block_leader_is_named_barrier():
    # Warp 0 (t0) waits on named barrier 1; every other warp just arrives.
    assert _lower_barrier("block_leader") == textwrap.dedent("""\
        mov.u32 ${b32:bl_tid}, %tid.x;
        setp.lt.u32 ${pred:bl_w0}, ${b32:bl_tid}, 32;
        @${pred:bl_w0} bar.sync 1, 128;
        @!${pred:bl_w0} bar.arrive 1, 128;
        """)


def test_lower_smem_arena_and_reinterpret():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %arena: smem<6144xbf16> = alloc() attrs={allocation_alignment: 1024, id: 0}
            %reinterpret: smem<32x64xbf16> = reinterpret(%arena: smem<6144xbf16>) attrs={byte_offset: 8192}
          }
        }"""))

    lower_to_ptx_template(module)
    templates = [
        op.attrs["template"]
        for op in find_ops_of_type(module, PTXTemplateOp)
    ]
    assert sum(".shared" in template for template in templates) == 1
    assert ".shared .align 1024" in templates[0]
    assert "[12288]" in templates[0]
    assert len(templates) == 1
    [reinterpret] = find_ops_of_type(module, PtrAddOp)
    assert isinstance(reinterpret.ptr.definition, ToPtrOp)
    assert isinstance(reinterpret.byte_offset.definition, IndexConstantOp)
    assert reinterpret.byte_offset.definition.value == 8192
    assert not find_ops_of_type(module, ReinterpretOp)


def test_lower_zero_offset_reinterpret_is_elided():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %arena: smem<64x64xbf16> = alloc() attrs={id: 0}
            %reinterpret: smem<32x64xbf16> = reinterpret(%arena: smem<64x64xbf16>) attrs={byte_offset: 0}
          }
        }"""))

    lower_to_ptx_template(module)

    assert not find_ops(module, lambda op: isinstance(
        op, (ReinterpretOp, ToPtrOp, PtrAddOp),
    ))


def test_lower_reshape_is_elided():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%arg: gmem<2x3x4xf32>) {
            %view: gmem<6x4xf32> = reshape(%arg: gmem<2x3x4xf32>)
          }
        }"""))

    lower_to_ptx_template(module)

    assert not find_ops(module, lambda op: isinstance(op, ReshapeOp))


def test_lower_tile_uses_the_reinterpreted_view_strides():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %arena: smem<4096xbf16> = alloc() attrs={id: 0}
            %view: smem<64x64xbf16> = reinterpret(%arena: smem<4096xbf16>) attrs={byte_offset: 0}
            %tile: smem<64x16xbf16 strides=[64, 1]> = tile(%view: smem<64x64xbf16>, 0, 1)
          }
        }"""))

    lower_to_ptx_template(module)
    Folding().run(module)

    [tile] = find_ops(module, lambda op: isinstance(op, PtrAddOp))
    assert isinstance(tile.byte_offset.definition, IndexConstantOp)
    assert tile.byte_offset.definition.value == 32


def test_lower_barrier_unsupported_scope_raises():
    with pytest.raises(NotImplementedError):
        _lower_barrier("grid")


def test_lower_tma_load_with_l2_cache_hint():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @load(%desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %x: index, %y: index, %z: index = block_ids()
            %smem: smem<64x64xf32> = alloc()
            %lock: lock<mbar> = alloc()
            async(%lock: lock<mbar>) tma_move(%smem: smem<64x64xf32>, %desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %x: index, %y: index) attrs={l2_cache_hint: "evict_last"}
          }
        }"""))
    lower_to_ptx_template(module)
    ptx = "".join(
        op.attrs["template"]
        for op in find_ops_of_type(module, PTXTemplateOp)
    )
    assert ptx.count("createpolicy.fractional.L2::evict_last") == 1
    assert "mbarrier::complete_tx::bytes.L2::cache_hint" in ptx


def test_lower_tma_store_with_l2_cache_hint():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @store(%desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %x: index, %y: index, %z: index = block_ids()
            %smem: smem<64x64xf32> = alloc()
            %lock: lock<bulk> = alloc()
            async(%lock: lock<bulk>) tma_move(%desc: tma_desc<arg_idx=0 gmem<64x64xf32> tile=64x64>, %smem: smem<64x64xf32>, %x: index, %y: index) attrs={l2_cache_hint: "evict_first"}
          }
        }"""))
    lower_to_ptx_template(module)
    ptx = "".join(
        op.attrs["template"]
        for op in find_ops_of_type(module, PTXTemplateOp)
    )
    assert ptx.count("createpolicy.fractional.L2::evict_first") == 1
    assert "global.shared::cta.bulk_group.L2::cache_hint" in ptx


def test_lower_move_gmem_to_smem():
    """Cooperative load: move(smem, gmem) lowers to a cooperative loop."""
    module_str = textwrap.dedent(
        """\
        module {
          kernel @load_tile(%a: gmem<4096x4096xbf16>) grid_dim=[64, 64, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %a_tile: gmem<64x16xbf16 strides=[4096, 1]> = tile(%a: gmem<4096x4096xbf16>, %by: index, %bx: index)
            %smem_a: smem<64x16xbf16> = alloc() attrs={id: 0}
            move(%smem_a: smem<64x16xbf16>, %a_tile: gmem<64x16xbf16 strides=[4096, 1]>) collective=xyz.block
          }
        }"""
    )  # noqa: E501

    module = gpu_dialect.parse(module_str)
    module = lower_to_async(module)
    lowered = lower_to_ptx_template(module)
    templates = find_ops_of_type(lowered, PTXTemplateOp)
    template = templates[-1].attrs["template"]

    assert_contains_in_order(
        template,
        # No bar.sync in the move itself: insert_syncs emits any needed barrier
        # before this op (the smem WAR/RAW is modeled there).
        textwrap.dedent(
            """\
            mov.u32 ${b32:co_idx}, %tid.x;
            mov.u32 ${b32:co_ntid}, %ntid.x;
            ${label:co_idx_loop}:
            setp.ge.u32 ${pred:co_idx_done}, ${b32:co_idx}, 16;
            @${pred:co_idx_done} bra ${label:co_end};
            mad.wide.u32 ${b64:co_gpar}, ${b32:co_idx}, 2, ${b64:operand.src.ptr};
            mul.lo.u32 ${b32:co_spar}, ${b32:co_idx}, 2;
            add.u32 ${b32:co_spar}, ${b32:co_spar}, ${b32:operand.dst.ptr};
            add.u64 ${b64:co_gaddr}, ${b64:co_gpar}, 0;
            add.u32 ${b32:co_saddr}, ${b32:co_spar}, 0;
            ld.global.b16 ${b16:co_tmp}, [${b64:co_gaddr}];
            st.shared.b16 [${b32:co_saddr}], ${b16:co_tmp};
            add.u64 ${b64:co_gaddr}, ${b64:co_gpar}, 8192;
            add.u32 ${b32:co_saddr}, ${b32:co_spar}, 32;
            ld.global.b16 ${b16:co_tmp}, [${b64:co_gaddr}];
            st.shared.b16 [${b32:co_saddr}], ${b16:co_tmp};
            """
        ),
        textwrap.dedent(
            """\
            add.u32 ${b32:co_idx}, ${b32:co_idx}, ${b32:co_ntid};
            bra ${label:co_idx_loop};
            ${label:co_end}:
            """
        ),
    )
    assert template.count("ld.global.b16") == 64, "64 rows unrolled"


def test_lower_elementwise_sigmoid_mma():
    """Lower every register of a valid mma.sync m16n8 accumulator.

    The hardware exponential is base 2, so exp(-x) is ex2 of -x*log2(e)
    (0fBFB8AA3B). dst doubles as the scratch register -- src is read by the
    first instruction and not after, which is what lets dst be src.
    """
    w = 'wrmem<16x8xf32 layout="mma">'
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {{
            %a: {w} = alloc() attrs={{id: 0}}
            %b: {w} = alloc() attrs={{id: 1}}
            elementwise sigmoid(%b: {w}, %a: {w})
          }}
        }}"""))

    lowered = lower_to_ptx_template(module)
    template = find_ops_of_type(lowered, PTXTemplateOp)[-1].attrs["template"]

    expected = "".join(
        textwrap.dedent(f"""\
            mul.f32 ${{f32:operand.dst.r{reg}}}, ${{f32:operand.src.r{reg}}}, 0fBFB8AA3B;
            ex2.approx.f32 ${{f32:operand.dst.r{reg}}}, ${{f32:operand.dst.r{reg}}};
            add.f32 ${{f32:operand.dst.r{reg}}}, ${{f32:operand.dst.r{reg}}}, 0f3F800000;
            rcp.approx.f32 ${{f32:operand.dst.r{reg}}}, ${{f32:operand.dst.r{reg}}};
            """)
        for reg in range(4)
    )
    assert template == expected


def test_lower_sigmoid_with_bf16_output_uses_ftz_phase_schedule():
    w = 'wrmem<16x8xf32 layout="mma">'
    b = 'wrmem<16x8xbf16 layout="mma">'
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {{
            %a: {w} = alloc() attrs={{id: 0}}
            %b: {b} = alloc() attrs={{id: 1}}
            elementwise sigmoid(%b: {b}, %a: {w})
          }}
        }}"""))

    [template] = [
        op.attrs["template"]
        for op in find_ops_of_type(
            lower_to_ptx_template(module), PTXTemplateOp,
        )
        if "ex2." in op.attrs["template"]
    ]
    assert "ex2.approx.ftz.f32" in template
    assert "rcp.approx.ftz.f32" in template


def test_lower_sigmoid_can_convert_its_result_to_f16():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %src: wrmem<16x8xf32 layout="mma"> = alloc() attrs={id: 0}
            %dst: wrmem<16x8xf16 layout="mma"> = alloc() attrs={id: 1}
            elementwise sigmoid(%dst: wrmem<16x8xf16 layout="mma">, %src: wrmem<16x8xf32 layout="mma">)
          }
        }"""))  # noqa: E501

    template = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )[-1].attrs["template"]
    assert "ex2.approx.f32" in template
    assert "ex2.approx.ftz.f32" not in template
    assert template.count("cvt.rn.f16x2.f32") == 2


def test_lower_elementwise_convert_uses_root_registers_for_wrmem_view():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            %dst: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %src_tail: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = tile(%src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, 0, 3)
            %dst_tail: wrmem<64x64xbf16 warps=4 layout="wgmma_layout_0"> = tile(%dst: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 3)
            elementwise convert(%dst_tail: wrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">, %src_tail: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    lowered = lower_to_ptx_template(module)
    template = find_ops_of_type(
        lowered, PTXTemplateOp,
    )[-1].attrs["template"]
    assert template.count("cvt.rn.bf16x2.f32") == 16
    assert "operand.dst.r48" in template
    assert "operand.src.r97" in template
    assert "operand.dst.r47" not in template


def test_lower_elementwise_uses_root_registers_for_wrmem_view():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            %tail: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = tile(%src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, 0, 3)
            elementwise sigmoid(%tail: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %tail: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
          }
        }"""))

    lowered = lower_to_ptx_template(module)
    template = find_ops_of_type(
        lowered, PTXTemplateOp,
    )[-1].attrs["template"]
    assert template.count("mul.f32") == 32
    assert template.count("ex2.approx.f32") == 32
    assert template.count("add.f32") == 32
    assert template.count("rcp.approx.f32") == 32
    assert "operand.dst.r96" in template
    assert "operand.dst.r95" not in template
    assert_contains_in_order(template, """
        mul.f32 ${f32:operand.dst.r96}, ${f32:operand.src.r96}, 0fBFB8AA3B;
        ex2.approx.f32 ${f32:operand.dst.r96}, ${f32:operand.dst.r96};
        add.f32 ${f32:operand.dst.r96}, ${f32:operand.dst.r96}, 0f3F800000;
        rcp.approx.f32 ${f32:operand.dst.r96}, ${f32:operand.dst.r96};
        mul.f32 ${f32:operand.dst.r97}, ${f32:operand.src.r97}, 0fBFB8AA3B;
    """)
