"""Tests for the lower_to_ptx_template pass."""

import textwrap

import pytest

from ayla.gpu_dialect import gpu_dialect, PTXTemplateOp
from ayla.base.ir import find_ops
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.testing import assert_contains_in_order


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
        for t in find_ops(lowered, lambda o: isinstance(o, PTXTemplateOp))
    )


@pytest.mark.parametrize("scope, expected", [
    ("thread", ""),                # program order, no barrier
    ("block", "bar.sync 0;\n"),    # full-CTA barrier
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


def test_lower_barrier_unsupported_scope_raises():
    with pytest.raises(NotImplementedError):
        _lower_barrier("grid")


def test_lower_move_gmem_to_smem():
    """Cooperative load: move(smem, gmem) lowers to a cooperative loop."""
    module_str = textwrap.dedent(
        """\
        module {
          kernel @load_tile(%a: gmem_tensor<4096x4096xbf16>) grid_dim=[64, 64, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %a_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<4096x4096xbf16>, %by: index, %bx: index)
            %smem_a: smem_tensor<64x16xbf16> = alloc() attrs={id: 0}
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
