"""Tests for materializing semantic register writebacks."""
import textwrap

import pytest

from ayla.base.ir import Value, find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    BarrierOp,
    CollectiveAxis,
    CollectiveLevel,
    IfOp,
    IndexConstantOp,
    IndexEqOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    gpu_dialect,
)
from ayla.passes.insert_syncs import insert_syncs
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_tma_move import lower_to_tma_move
from ayla.passes.stage_register_writeback import stage_register_writeback
from ayla.ptx_codegen import make_ptx_kernel


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
          kernel @writeback(%src: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">, %dst: gmem<128x256xbf16>) grid_dim=[1, 1, 1] block_dim=[512, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %c2: index = index_constant() attrs={value: 2}
            %row: index = index_div(%wgx: index, %c2: index)
            %col: index = index_mod(%wgx: index, %c2: index)
            %rows: gmem<64x256xbf16 strides=[256, 1]> = tile(%dst: gmem<128x256xbf16>, %row: index, 0)
            %tile: gmem<64x128xbf16 strides=[256, 1]> = tile(%rows: gmem<64x256xbf16 strides=[256, 1]>, 0, %col: index)
            move(%tile: gmem<64x128xbf16 strides=[256, 1]>, %src: wrmem<64x128xbf16 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={dst_tiling_scheme: 7, label: "mm_dst_wrmem", tma: True}
          }
        }"""))

    stage_register_writeback(
        module,
        writeback_slices=2,
        physical_layout={
            "swizzle": "128B",
            "block_width": 64,
            "order": "block_major",
        },
        label_filter="mm_dst_wrmem",
    )

    moves = find_ops_of_type(module, MoveOp)
    gathers = [move for move in moves if not move.attrs.get("tma")]
    stores = [move for move in moves if move.attrs.get("tma")]
    assert len(gathers) == len(stores) == 2
    conditions = find_ops_of_type(module, IndexEqOp)
    assert all(condition.lhs.name == "col" for condition in conditions)
    slice_ids = []
    for condition in conditions:
        definition = condition.rhs.definition
        assert isinstance(definition, IndexConstantOp)
        slice_ids.append(definition.value)
    assert slice_ids == [0, 1]
    guards = find_ops_of_type(module, IfOp)
    assert [guard.condition for guard in guards] == [
        condition.result() for condition in conditions
    ]
    assert [guard.body.ops for guard in guards] == [[move] for move in gathers]
    for move in gathers:
        tile = move.dst.definition
        assert isinstance(tile, TileOp)
        row = tile.indices[0]
        assert isinstance(row, Value) and row.name == "row"
        assert isinstance(move.dst.type, SmemTensorType)
        assert move.dst.type.sizes == [64, 128]
    assert all(
        move.collective[CollectiveAxis.X] == CollectiveLevel.WG
        for move in stores
    )
    ptx = make_ptx_kernel(module).ptx
    assert ptx.count("setp.eq.u32") == 2
    assert ptx.count("stmatrix.sync") == 16


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
    assert isinstance(gather.dst.type, SmemTensorType)
    assert gather.dst.type.sizes == [64, 64]
    tile = gather.dst.definition
    assert isinstance(tile, TileOp)
    row = tile.indices[0]
    assert isinstance(row, Value) and row.name == "wgx"
    assert store.collective[CollectiveAxis.X] == CollectiveLevel.WG

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
