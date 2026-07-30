"""Tests for round-robin output-store staging."""
import textwrap

import pytest

from ayla.base.ir import find_ops
from ayla.gpu_dialect import AllocOp, MoveOp, SmemTensorType, TileOp, gpu_dialect
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_tma_move import lower_to_tma_move
from ayla.passes.schedule_async import schedule_async
from ayla.passes.stage_store import stage_store


def _store_module(num_stores: int = 2):
    stores = "\n".join(
        f"""
        %c{i}: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<{128 * num_stores}x256xbf16>, {i}, 0)
        %d{i}: gmem_tensor<64x256xbf16> = tile(%c{i}: gmem_tensor<128x256xbf16>, %wgx: index, 0)
        %a{i}: wrmem_tensor<64x256xbf16 warps=4 layout=\"wgmma_layout_0\"> = alloc()
        move(%d{i}: gmem_tensor<64x256xbf16>, %a{i}: wrmem_tensor<64x256xbf16 warps=4 layout=\"wgmma_layout_0\">) attrs={{label: \"store_iter{i}\"}}
        """
        for i in range(num_stores)
    )
    return gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @store(%c: gmem_tensor<{128 * num_stores}x256xbf16>) {{
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            {stores}
          }}
        }}"""))


def _stage_allocs(module):
    return find_ops(module, lambda op: (
        isinstance(op, AllocOp)
        and isinstance(op.result().type, SmemTensorType)
    ))


def test_stage_store_uses_one_arena_for_two_stores():
    module = _store_module()
    stage_store(module, label_filter=r"store_iter\d+", num_buffers=1)
    assert len(_stage_allocs(module)) == 1
    tma_stores = find_ops(module, lambda op: (
        isinstance(op, MoveOp) and op.attrs.get("tma") is True
    ))
    assert len(tma_stores) == 2
    assert tma_stores[0].src is tma_stores[1].src


def test_stage_store_round_robins_two_arenas_at_distance_two():
    module = _store_module(3)
    stage_store(module, label_filter=r"store_iter\d+", num_buffers=2)
    assert len(_stage_allocs(module)) == 2
    tma_stores = find_ops(module, lambda op: (
        isinstance(op, MoveOp) and op.attrs.get("tma") is True
    ))
    assert tma_stores[0].src is tma_stores[2].src
    assert tma_stores[0].src is not tma_stores[1].src


def test_two_arenas_delay_first_wait_until_after_second_store():
    module = _store_module()
    stage_store(module, label_filter=r"store_iter\d+", num_buffers=2)
    module = schedule_async(lower_to_tma_move(lower_to_async(module)))
    block = module.body.entry_block.ops[0].body.entry_block
    first_wait = next(i for i, op in enumerate(block.ops) if op.op_id == "await")
    last_tma = max(
        i for i, op in enumerate(block.ops)
        if op.op_id == "async"
    )
    assert last_tma < first_wait


def test_stage_store_rejects_invalid_buffer_count():
    with pytest.raises(ValueError, match="num_buffers must be positive"):
        stage_store(_store_module(), r"store_iter\d+", num_buffers=0)


def test_stage_store_is_noop_without_matching_stores():
    module = _store_module()
    before = repr(module)
    stage_store(module, label_filter=r"other", num_buffers=2)
    assert repr(module) == before


def test_stage_store_shares_kernel_scoped_arenas_across_blocks():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @store(%c: gmem_tensor<256x256xbf16>) {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            for (%i: index = 0 to 1) {
              %c0: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<256x256xbf16>, 0, 0)
              %d0: gmem_tensor<64x256xbf16> = tile(%c0: gmem_tensor<128x256xbf16>, %wgx: index, 0)
              %a0: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
              move(%d0: gmem_tensor<64x256xbf16>, %a0: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "store_0"}
            }
            for (%j: index = 0 to 1) {
              %c1: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<256x256xbf16>, 1, 0)
              %d1: gmem_tensor<64x256xbf16> = tile(%c1: gmem_tensor<128x256xbf16>, %wgx: index, 0)
              %a1: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
              move(%d1: gmem_tensor<64x256xbf16>, %a1: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "store_1"}
            }
          }
        }"""))
    stage_store(module, label_filter=r"store_\d+", num_buffers=2)
    assert len(_stage_allocs(module)) == 2
    kernel_block = module.body.entry_block.ops[0].body.entry_block
    assert all(isinstance(op, AllocOp) for op in kernel_block.ops[:2])
