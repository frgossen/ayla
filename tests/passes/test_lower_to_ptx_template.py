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
    ("leader_to_cluster", textwrap.dedent("""\
        barrier.cluster.arrive.release;
        barrier.cluster.wait.acquire;
    """)),
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


def test_lower_cluster_to_leader_barrier_uses_its_lock():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[2, 1, 1] block_dim=[128, 1, 1] cluster_dim=[2, 1, 1] {
            %lock: lock<cluster_mbar> = alloc()
            barrier(%lock: lock<cluster_mbar>) attrs={scope: "cluster_to_leader"}
          }
        }"""))

    lower_to_ptx_template(module)
    ptx = "".join(
        op.attrs["template"]
        for op in find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    )
    assert ".shared .align 8 .b64 ${smem:bar}[1];" in ptx
    assert (
        "mbarrier.init.shared::cta.b64 [${b32:result.dst.val}], 2;"
        in ptx
    )
    assert_contains_in_order(
        ptx,
        "fence.mbarrier_init.release.cluster;",
        "bar.sync 0;",
        "barrier.cluster.arrive.release;",
        "barrier.cluster.wait.acquire;",
        "mapa.shared::cluster.u32 ${b32:remote_empty}, ${b32:operand.sync.val}, ${b32:cluster_rank};",
        "mbarrier.arrive.shared::cluster.b64 _, [${b32:remote_empty}];",
        "mbarrier.arrive.shared::cta.b64 ${b64:cluster_state}, [${b32:operand.sync.val}];",
        "mbarrier.try_wait.acquire.cluster.shared::cta.b64 ${pred:ready}, [${b32:operand.sync.val}], ${b64:cluster_state};",
    )


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


def test_lower_nested_tile_uses_root_block_major_coordinates():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %root: smem<128x128xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = alloc() attrs={id: 0}
            %rows: smem<64x128xbf16 strides=[128, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%root: smem<128x128xbf16 swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 1, 0)
            %inner: smem<64x16xbf16 strides=[128, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}> = tile(%rows: smem<64x128xbf16 strides=[128, 1] swizzle="128B" attrs={physical_layout: {block_width: 64, order: "block_major"}}>, 0, 5)
          }
        }"""))

    lower_to_ptx_template(module)
    Folding().run(module)

    ptr_adds = find_ops_of_type(module, PtrAddOp)
    assert len(ptr_adds) == 2
    offsets = [op.byte_offset.definition for op in ptr_adds]
    assert all(isinstance(offset, IndexConstantOp) for offset in offsets)
    assert sum(offset.value for offset in offsets) == 24608


def test_grid_barrier_uses_last_arriver_to_release_waiters():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%sync: gmem<2xf32>) grid_dim=[8, 1, 1] block_dim=[128, 1, 1] {
            barrier(%sync: gmem<2xf32>) attrs={scope: "grid"}
          }
        }"""))
    lower_to_ptx_template(module)
    template = find_ops(
        module, lambda op: isinstance(op, PTXTemplateOp),
    )[0].attrs["template"]
    assert template.count("bar.sync 0;") == 2
    assert "atom.acq_rel.gpu.global.inc.u32" in template
    assert "setp.eq.u32 ${pred:grid_last}, ${b32:grid_ticket}, 7;" in template
    assert "cp.async.bulk.wait_group" not in template
    assert "grid_count_wait" not in template


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


def test_lower_tma_multicast_uses_one_cluster_issuer_and_cluster_wait():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @load(%desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) grid_dim=[2, 1, 1] block_dim=[32, 1, 1] cluster_dim=[2, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %smem: smem<16x16xf32> = alloc() attrs={id: 0}
            %lock: lock<mbar> = alloc() attrs={id: 1}
            async(%lock: lock<mbar>) collective=xyz.cluster tma_move(%smem: smem<16x16xf32>, %desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={l2_cache_hint: "evict_last", multicast: True}
            await(%lock: lock<mbar>)
          }
        }"""))

    lower_to_ptx_template(module)

    ptx = "".join(
        op.attrs["template"]
        for op in find_ops(module, lambda op: isinstance(op, PTXTemplateOp))
    )
    leader = "mov.u32 ${b32:cluster_rank}, %cluster_ctarank;"
    multicast = (
        "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
    )
    arrive = "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"
    wait = "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
    assert_contains_in_order(
        ptx,
        "fence.mbarrier_init.release.cluster;",
        "bar.sync 0;",
        "barrier.cluster.arrive.release;",
        "barrier.cluster.wait.acquire;",
    )
    assert (
        ptx.index("barrier.cluster.wait.acquire;")
        < ptx.index(leader)
        < ptx.index(multicast)
        < ptx.index(arrive)
        < ptx.index(wait)
    )
    assert ptx.count(multicast) == 1
    assert (
        "[${b32:operand.lock.val}], 3, "
        "${b64:operand.cache_policy0.ptr};"
    ) in ptx


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
    w = 'wgrmem<16x8xf32 layout="mma">'
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
    w = 'wgrmem<16x8xf32 layout="mma">'
    b = 'wgrmem<16x8xbf16 layout="mma">'
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
    assert "${f32:sigmoid_result0}" in template
    assert "${f32:sigmoid_result1}" in template
    assert "sigmoid_result2" not in template
    assert not any(
        line.startswith("mul.f32 ${f32:operand.src")
        for line in template.splitlines()
    )


def test_lower_sigmoid_can_convert_its_result_to_f16():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %src: wgrmem<16x8xf32 layout="mma"> = alloc() attrs={id: 0}
            %dst: wgrmem<16x8xf16 layout="mma"> = alloc() attrs={id: 1}
            elementwise sigmoid(%dst: wgrmem<16x8xf16 layout="mma">, %src: wgrmem<16x8xf32 layout="mma">)
          }
        }"""))  # noqa: E501

    template = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )[-1].attrs["template"]
    assert "ex2.approx.f32" in template
    assert "ex2.approx.ftz.f32" not in template
    assert template.count("cvt.rn.f16x2.f32") == 2


def test_lower_elementwise_convert_uses_root_registers_for_wgrmem_view():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %src: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            %dst: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %src_tail: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = tile(%src: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, 0, 3)
            %dst_tail: wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0"> = tile(%dst: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, 0, 3)
            elementwise convert(%dst_tail: wgrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">, %src_tail: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
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


def test_lower_elementwise_convert_supports_n8_wgrmem_fragment():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %src: wgrmem<64x80xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            %dst: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            %src_tail: wgrmem<64x40xf32 warps=4 layout="wgmma_layout_0"> = tile(%src: wgrmem<64x80xf32 warps=4 layout="wgmma_layout_0">, 0, 1)
            elementwise convert(%dst: wgrmem<64x40xbf16 warps=4 layout="wgmma_layout_0">, %src_tail: wgrmem<64x40xf32 warps=4 layout="wgmma_layout_0">)
          }
        }"""))  # noqa: E501

    template = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )[-1].attrs["template"]
    assert template.count("cvt.rn.bf16x2.f32") == 10
    assert "operand.dst.r9" in template
    assert "operand.src.r39" in template


@pytest.mark.parametrize("n", range(8, 257, 8))
def test_lower_elementwise_supports_every_wgmma_n_extent(n):
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %src: wgrmem<64x{n}xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={{id: 0}}
            %dst: wgrmem<64x{n}xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={{id: 1}}
            elementwise convert(%dst: wgrmem<64x{n}xbf16 warps=4 layout="wgmma_layout_0">, %src: wgrmem<64x{n}xf32 warps=4 layout="wgmma_layout_0">)
          }}
        }}"""))  # noqa: E501

    template = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )[-1].attrs["template"]
    assert template.count("cvt.rn.bf16x2.f32") == n // 4


def test_lower_elementwise_uses_root_registers_for_wgrmem_view():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %src: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0}
            %tail: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = tile(%src: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, 0, 3)
            elementwise sigmoid(%tail: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %tail: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
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


def test_lower_elementwise_mul():
    w = 'wgrmem<16x8xf32 layout="mma">'
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %lhs: {w} = alloc() attrs={{id: 0}}
            %rhs: {w} = alloc() attrs={{id: 1}}
            elementwise mul(%lhs: {w}, %lhs: {w}, %rhs: {w})
          }}
        }}"""))

    lowered = lower_to_ptx_template(module)
    template = find_ops(
        lowered, lambda o: isinstance(o, PTXTemplateOp)
    )[-1].attrs["template"]
    assert template == "".join(
        f"mul.f32 ${{f32:operand.dst.r{reg}}}, "
        f"${{f32:operand.lhs.r{reg}}}, ${{f32:operand.rhs.r{reg}}};\n"
        for reg in range(4)
    )


def test_lower_elementwise_silu():
    w = 'wgrmem<16x8xf32 layout="mma">'
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %a: {w} = alloc() attrs={{id: 0}}
            elementwise silu(%a: {w}, %a: {w})
          }}
        }}"""))

    lowered = lower_to_ptx_template(module)
    template = find_ops(
        lowered, lambda op: isinstance(op, PTXTemplateOp)
    )[-1].attrs["template"]

    assert "${f32:silu_tmp0}" in template
    assert "silu_tmp1" not in template
    assert template.count("ex2.approx.f32") == 4
    assert template.count("mul.f32 ${f32:operand.dst") == 4


def test_lower_elementwise_silu_can_opt_out_of_minimum_register_schedule():
    w = 'wgrmem<16x8xf32 layout="mma">'
    b = 'wgrmem<16x8xbf16 layout="mma">'
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %src: {w} = alloc() attrs={{id: 0}}
            %dst: {b} = alloc() attrs={{id: 1}}
            elementwise silu(%dst: {b}, %src: {w}) attrs={{ilp: True}}
          }}
        }}"""))

    lowered = lower_to_ptx_template(module)
    template = find_ops(
        lowered, lambda op: isinstance(op, PTXTemplateOp)
    )[-1].attrs["template"]

    assert template.count("ex2.approx.ftz.f32") == 4
    assert template.count("rcp.approx.ftz.f32") == 4
    assert "ex2.approx.f32" not in template
    assert "${f32:silu_result3}" in template
    assert template.endswith(textwrap.dedent("""\
        cvt.rn.bf16x2.f32 ${b32:operand.dst.r0}, ${f32:silu_result1}, ${f32:silu_result0};
        cvt.rn.bf16x2.f32 ${b32:operand.dst.r1}, ${f32:silu_result3}, ${f32:silu_result2};
        """))


def test_lower_elementwise_silu_minimizes_temporary_live_ranges_by_default():
    src = 'wgrmem<16x8xf32 layout="mma">'
    dst = 'wgrmem<16x8xbf16 layout="mma">'
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @k() grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {{
            %src: {src} = alloc() attrs={{id: 0}}
            %dst: {dst} = alloc() attrs={{id: 1}}
            elementwise silu(%dst: {dst}, %src: {src})
          }}
        }}"""))

    template = find_ops_of_type(
        lower_to_ptx_template(module), PTXTemplateOp,
    )[-1].attrs["template"]
    assert template.count("ex2.approx.ftz.f32") == 4
    assert "silu_tmp" not in template
    assert "${f32:silu_result0}" in template
    assert "${f32:silu_result1}" in template
    assert "silu_result2" not in template
    assert template.count("cvt.rn.bf16x2.f32") == 2
    assert not any(
        line.startswith("mul.f32 ${f32:operand.src")
        for line in template.splitlines()
    )
