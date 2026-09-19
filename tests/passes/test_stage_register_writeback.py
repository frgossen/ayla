"""Tests for materializing semantic register writebacks."""
import textwrap

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import AllocOp, MoveOp, gpu_dialect
from ayla.passes.stage_register_writeback import stage_register_writeback


def test_ignores_direct_register_writeback():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @writeback(%src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, %dst: gmem<64x256xbf16>) {
            move(%dst: gmem<64x256xbf16>, %src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={label: "mm_dst_wrmem"}
          }
        }"""))

    original = repr(module)
    stage_register_writeback(module, label_filter="mm_dst_wrmem")
    assert repr(module) == original


def test_materializes_sliced_double_buffered_tma_store():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @writeback(%src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, %dst: gmem<64x256xbf16>) {
            move(%dst: gmem<64x256xbf16>, %src: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread] attrs={l2_cache_hint: "evict_first", label: "mm_dst_wrmem", tma: True}
          }
        }"""))

    stage_register_writeback(
        module,
        writeback_slices=2,
        writeback_buffers=2,
        unsafe_reuse_smem=True,
        label_filter="mm_dst_wrmem",
    )

    allocs = find_ops_of_type(module, AllocOp)
    assert len(allocs) == 3
    assert [op.attrs.get("reuse_group") for op in allocs] == [
        None, "mm_writeback", "mm_writeback",
    ]
    moves = find_ops_of_type(module, MoveOp)
    assert [op.attrs.get("label") for op in moves] == [
        "mm_dst_smem", "mm_dst_tma", "mm_dst_smem", "mm_dst_tma",
    ]
    assert [op.attrs.get("wgmma_col_start") for op in moves[::2]] == [0, 128]
    assert all(op.attrs.get("tma") is True for op in moves[1::2])
    assert all(
        op.attrs.get("l2_cache_hint") == "evict_first"
        for op in moves[1::2]
    )


@pytest.mark.parametrize(
    ("slices", "buffers"),
    [(0, 1), (2, 3)],
)
def test_rejects_invalid_staging_counts(slices: int, buffers: int):
    module = gpu_dialect.parse("module {}")
    with pytest.raises(ValueError):
        stage_register_writeback(
            module,
            writeback_slices=slices,
            writeback_buffers=buffers,
        )
