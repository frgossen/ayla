"""Tests for the lower_to_ptx_template pass."""

import textwrap

import pytest

from ayla.gpu_dialect import (
    gpu_dialect, PTXTemplateOp, KernelOp, EXPLICIT_BARRIERS_ATTR,
)
from ayla.base.ir import find_op, find_ops
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.testing import assert_contains_in_order


@pytest.mark.parametrize("explicit_barriers", [False, True])
def test_explicit_barriers_gates_upfront_tma_bar_sync(explicit_barriers):
    """Codegen emits an upfront bar.sync before a TMA load (the smem-reuse WAR),
    unless the kernel is marked ``explicit_barriers`` — then insert_syncs owns
    the barriers and codegen must not double up.

    Input is a post-``lower_to_async``/``lower_to_tma_move`` async TMA load, so
    this exercises ``lower_to_ptx_template`` in isolation."""
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%a: gmem_tensor<128x64xbf16>, %a_desc: tma_desc<arg_idx=0 gmem_tensor<128x64xbf16> tile=128x64 swizzle="128B">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %sa: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %c0: index = index_constant() attrs={value: 0}
            %0: lock<mbar> = alloc()
            async(%0: lock<mbar>) tma_move(%sa: smem_tensor<128x64xbf16 swizzle="128B">, %a_desc: tma_desc<arg_idx=0 gmem_tensor<128x64xbf16> tile=128x64 swizzle="128B">, %c0: index, %c0: index) attrs={tma: True}
            await(%0: lock<mbar>)
          }
        }"""))  # noqa: E501
    if explicit_barriers:
        find_op(module, lambda o: isinstance(o, KernelOp)).attrs[EXPLICIT_BARRIERS_ATTR] = True

    lowered = lower_to_ptx_template(module)
    [tma] = [
        t.attrs["template"]
        for t in find_ops(lowered, lambda o: isinstance(o, PTXTemplateOp))
        if "cp.async.bulk.tensor" in t.attrs["template"]
    ]

    # Elect thread 0, issue the bulk-tensor copy, arrive on the mbarrier.
    expected = textwrap.dedent("""\
        mov.u32 ${b32:tid_x}, %tid.x;
        mov.u32 ${b32:tid_y}, %tid.y;
        or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_y};
        mov.u32 ${b32:tid_z}, %tid.z;
        or.b32 ${b32:tid_x}, ${b32:tid_x}, ${b32:tid_z};
        setp.ne.s32 ${pred:not_t0}, ${b32:tid_x}, 0;
        @${pred:not_t0} bra ${label:after_load};
        cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [${b32:operand.dst0.ptr}], [${b64:operand.desc0.ptr}, {${b32:operand.idx0_1.val}, ${b32:operand.idx0_0.val}}], [${b32:operand.lock.val}];
        mov.u32 ${b32:exp_bytes}, 16384;
        mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [${b32:operand.lock.val}], ${b32:exp_bytes};
        ${label:after_load}:
        """)  # noqa: E501
    if not explicit_barriers:
        # Codegen owns the smem-reuse WAR: bar.sync before issuing the load.
        expected = "bar.sync 0;\n" + expected

    assert tma == expected


def test_lower_move_gmem_to_smem():
    """Cooperative load: move(smem, gmem) lowers to a cooperative loop."""
    module_str = textwrap.dedent(
        """\
        module {
          kernel @load_tile(%a: gmem_tensor<4096x4096xbf16>) grid_dim=[64, 64, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %a_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<4096x4096xbf16>, %by: index, %bx: index)
            %smem_a: smem_tensor<64x16xbf16> = alloc()
            move(%smem_a: smem_tensor<64x16xbf16>, %a_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]>) attrs={collective: "block"}
          }
        }"""
    )  # noqa: E501

    module = gpu_dialect.parse(module_str)
    module = lower_to_async(module)
    lowered = lower_to_ptx_template(module)
    templates = find_ops(lowered, lambda op: isinstance(op, PTXTemplateOp))
    template = templates[-1].attrs["template"]

    assert_contains_in_order(
        template,
        # Sync upfront (before the cooperative loop): prior readers finish
        # before this load overwrites the smem tile.
        "bar.sync 0;\n",
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
