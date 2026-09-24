"""Tests for materializing semantic register writebacks."""
import textwrap

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    BarrierOp,
    CollectiveAxis,
    CollectiveLevel,
    MoveOp,
    TileOp,
    gpu_dialect,
)
from ayla.passes.insert_syncs import insert_syncs
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_tma_move import lower_to_tma_move
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


def test_multi_wg_rows_and_columns_keep_their_staging_ownership():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @writeback(%src: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">, %dst: gmem<128x256xbf16>) block_dim=[512, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %c2: index = index_constant() attrs={value: 2}
            %row: index = index_div(%wgx: index, %c2: index)
            %col: index = index_mod(%wgx: index, %c2: index)
            %rows: gmem<64x256xbf16 strides=[256, 1]> = tile(%dst: gmem<128x256xbf16>, %row: index, 0)
            %tile: gmem<64x128xbf16 strides=[256, 1]> = tile(%rows: gmem<64x256xbf16 strides=[256, 1]>, 0, %col: index)
            move(%tile: gmem<64x128xbf16 strides=[256, 1]>, %src: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "mm_dst_wrmem", tma: True}
          }
        }"""))

    stage_register_writeback(
        module, writeback_slices=2, label_filter="mm_dst_wrmem",
    )

    moves = find_ops_of_type(module, MoveOp)
    gathers, stores = moves[::2], moves[1::2]
    assert len(gathers) == len(stores) == 2
    assert all(len(move.indices) == 1 for move in gathers)
    assert all(move.indices[0].name == "col" for move in gathers)
    assert [move.attrs["writer_wg"] for move in gathers] == [0, 1]
    assert all(
        isinstance(move.dst.definition, TileOp)
        and move.dst.definition.indices[0].name == "row"
        and move.dst.type.sizes == [64, 128]
        for move in gathers
    )
    assert all(
        all(level >= CollectiveLevel.BLOCK for level in move.collective.values())
        for move in stores
    )
    assert all(
        move.collective[CollectiveAxis.X] == CollectiveLevel.BLOCK
        for move in stores
    )


def test_four_row_warp_groups_sync_before_block_wide_tma_store():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @writeback(%src: wrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">, %dst: gmem<256x64xbf16>) block_dim=[512, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %rows: gmem<64x64xbf16> = tile(%dst: gmem<256x64xbf16>, %wgx: index, 0)
            move(%rows: gmem<64x64xbf16>, %src: wrmem<64x64xbf16 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "mm_dst_wrmem", tma: True}
          }
        }"""))

    stage_register_writeback(module, label_filter="mm_dst_wrmem")
    [gather, store] = find_ops_of_type(module, MoveOp)
    assert gather.dst.type.sizes == [64, 64]
    assert isinstance(gather.dst.definition, TileOp)
    assert gather.dst.definition.indices[0].name == "wgx"
    assert all(
        level >= CollectiveLevel.BLOCK for level in store.collective.values()
    )

    lower_to_tma_move(module)
    lower_to_async(module)
    insert_syncs(module)
    [barrier] = find_ops_of_type(module, BarrierOp)
    assert barrier.scope == "block_leader"
    assert barrier.attrs["num_threads"] == 512


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
