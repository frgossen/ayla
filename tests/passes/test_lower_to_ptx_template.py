"""Tests for the lower_to_ptx_template pass."""

import textwrap

from ayla.gpu_dialect import gpu_dialect, PTXTemplateOp
from ayla.base.ir import find_ops
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.testing import assert_contains_in_order


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


def test_lower_move_wrmem_wgmma_to_smem():
    """wgmma_layout_0 wrmem → non-swizzled smem stages via st.shared.b32."""
    module_str = textwrap.dedent(
        """\
        module {
          kernel @stage(%c: gmem_tensor<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %acc: wrmem_tensor<64x64xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            %smem: smem_tensor<64x64xbf16> = alloc()
            move(%smem: smem_tensor<64x64xbf16>, %acc: wrmem_tensor<64x64xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""
    )  # noqa: E501

    module = gpu_dialect.parse(module_str)
    lowered = lower_to_ptx_template(module)
    template = find_ops(
        lowered, lambda op: isinstance(op, PTXTemplateOp)
    )[-1].attrs["template"]

    # 32-bit smem base computed from the per-thread dynamic offset.
    assert_contains_in_order(
        template,
        "add.u32 ${b32:base}, ${b32:operand.dst.ptr}, ${b32:operand.offset.val};\n",
        # reg 0: (row 0, col 0)   -> byte 0
        "st.shared.b32 [${b32:base}+0], ${b32:operand.src.r0};\n",
        # reg 1: (row 8, col 0)   -> 8 * (64*2) = 1024
        "st.shared.b32 [${b32:base}+1024], ${b32:operand.src.r1};\n",
        # reg 2: (row 0, col 8)   -> 8 * 2 = 16
        "st.shared.b32 [${b32:base}+16], ${b32:operand.src.r2};\n",
        # reg 3: (row 8, col 8)   -> 1024 + 16 = 1040
        "st.shared.b32 [${b32:base}+1040], ${b32:operand.src.r3};\n",
    )
    # 64x64 wgmma_layout_0 over 4 warps = 32 elems/thread = 16 b32 regs.
    assert template.count("st.shared.b32") == 16
    assert "st.global" not in template


def test_lower_move_wrmem_wgmma_to_compact_smem_n_slice():
    """A 64-column packed stage selects the matching source registers."""
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @stage_slice() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %acc: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            %smem: smem_tensor<64x64xbf16> = alloc()
            move(%smem: smem_tensor<64x64xbf16>, %acc: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={wgmma_n_offset: 64}
          }
        }"""))
    lowered = lower_to_ptx_template(module)
    template = find_ops(
        lowered, lambda op: isinstance(op, PTXTemplateOp)
    )[-1].attrs["template"]
    assert template.count("st.shared.b32") == 16
    assert "operand.src.r16" in template
    assert "operand.src.r0" not in template


def test_bulk_store_await_rendezvous_before_smem_reuse():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @wait() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %lock: lock<bulk> = alloc()
            await(%lock: lock<bulk>)
          }
        }"""))
    lowered = lower_to_ptx_template(module)
    template = find_ops(
        lowered, lambda op: isinstance(op, PTXTemplateOp)
    )[-1].attrs["template"]
    assert_contains_in_order(
        template,
        "cp.async.bulk.wait_group 0;\n",
        "bar.sync 0;\n",
    )
