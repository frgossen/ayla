"""Tests for the lower_to_ptx_template pass."""

import textwrap

from ayla.gpu_dialect import gpu_dialect, PTXTemplateOp
from ayla.ir import find_ops
from ayla.lower_to_async import lower_to_async
from ayla.lower_to_ptx_template import lower_to_ptx_template
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
            bar.sync 0;
            """
        ),
    )
    assert template.count("ld.global.b16") == 64, "64 rows unrolled"
