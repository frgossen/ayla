"""Tests for output-store pipelining."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import AllocOp, AsyncOp, AwaitOp, ConvertOp, MoveOp, SmemTensorType, gpu_dialect
from ayla.passes.pipeline_store import pipeline_store


def _store_module(num_stores=2):
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
          kernel @store(%c: gmem_tensor<{128 * num_stores}x256xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            {stores}
          }}
        }}"""))


def test_pipeline_store_stages_tma_and_delays_waits():
    module = _store_module()
    pipeline_store(module, label_filter=r"store_iter\d+")
    block = module.body.entry_block.ops[0].body.entry_block

    stage_allocs = [
        op for op in block.ops
        if isinstance(op, AllocOp)
        and isinstance(op.result().type, SmemTensorType)
    ]
    assert len(stage_allocs) == 1
    async_ops = find_ops(module, lambda op: isinstance(op, AsyncOp))
    assert len(async_ops) == 2
    assert all(len(op.body.entry_block.ops) == 1 for op in async_ops)
    assert len(find_ops(module, lambda op: isinstance(op, AwaitOp))) == 2

    first_async = next(i for i, op in enumerate(block.ops) if isinstance(op, AsyncOp))
    first_wait = next(i for i, op in enumerate(block.ops) if isinstance(op, AwaitOp))
    second_stage = next(
        i for i, op in enumerate(block.ops)
        if isinstance(op, MoveOp)
        and op.attrs.get("label") == "store_iter1_stage"
    )
    assert first_async < first_wait < second_stage


def test_pipeline_store_double_buffers_until_kernel_tail():
    module = _store_module()
    pipeline_store(module, label_filter=r"store_iter\d+", num_buffers=2)
    block = module.body.entry_block.ops[0].body.entry_block

    stage_allocs = [
        op for op in block.ops
        if isinstance(op, AllocOp)
        and isinstance(op.result().type, SmemTensorType)
    ]
    assert len(stage_allocs) == 2
    assert len(find_ops(module, lambda op: isinstance(op, AsyncOp))) == 2
    assert len(find_ops(module, lambda op: isinstance(op, AwaitOp))) == 2

    last_async = max(i for i, op in enumerate(block.ops) if isinstance(op, AsyncOp))
    first_wait = min(i for i, op in enumerate(block.ops) if isinstance(op, AwaitOp))
    assert last_async < first_wait


def test_pipeline_store_double_buffer_waits_at_distance_two():
    module = _store_module(3)
    pipeline_store(module, label_filter=r"store_iter\d+", num_buffers=2)
    block = module.body.entry_block.ops[0].body.entry_block

    first_wait = next(i for i, op in enumerate(block.ops) if isinstance(op, AwaitOp))
    second_stage = next(
        i for i, op in enumerate(block.ops)
        if isinstance(op, MoveOp)
        and op.attrs.get("label") == "store_iter1_stage"
    )
    third_stage = next(
        i for i, op in enumerate(block.ops)
        if isinstance(op, MoveOp)
        and op.attrs.get("label") == "store_iter2_stage"
    )
    assert second_stage < first_wait < third_stage


def test_pipeline_store_overlaps_convert_with_load_and_stage_with_wgmma():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @overlap(%x: gmem_tensor<64x16xbf16>, %b: smem_tensor<256x16xbf16>, %c: gmem_tensor<256x256xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %c0: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<256x256xbf16>, 0, 0)
            %d0: gmem_tensor<64x256xbf16> = tile(%c0: gmem_tensor<128x256xbf16>, %wgx: index, 0)
            %f0: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            %o: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            convert(%o: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">, %f0: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">)
            move(%d0: gmem_tensor<64x256xbf16>, %o: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "store_iter0"}
            %s: smem_tensor<64x16xbf16> = alloc()
            move(%s: smem_tensor<64x16xbf16>, %x: gmem_tensor<64x16xbf16>) attrs={label: "next_load", tma: True}
            %f1: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            matmul(%s: smem_tensor<64x16xbf16>, %b: smem_tensor<256x16xbf16>, %f1: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "next_mm"}
            %c1: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<256x256xbf16>, 1, 0)
            %d1: gmem_tensor<64x256xbf16> = tile(%c1: gmem_tensor<128x256xbf16>, %wgx: index, 0)
            convert(%o: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">, %f1: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">)
            move(%d1: gmem_tensor<64x256xbf16>, %o: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "store_iter1"}
          }
        }"""))
    pipeline_store(module, label_filter=r"store_iter\d+")
    block = module.body.entry_block.ops[0].body.entry_block

    convert0 = [op for op in block.ops if isinstance(op, ConvertOp)][0]
    next_load = next(op for op in block.ops if isinstance(op, AsyncOp)
                     and op.attrs.get("label") == "next_load_async")
    next_mm = next(op for op in block.ops if isinstance(op, AsyncOp)
                   and op.attrs.get("label") == "next_mm_async")
    next_mm_wait = next(op for op in block.ops if isinstance(op, AwaitOp)
                        and op.lock is next_mm.lock)
    stage0 = next(op for op in block.ops if isinstance(op, MoveOp)
                  and op.attrs.get("label") == "store_iter0_stage")

    assert block.ops.index(next_load) < block.ops.index(convert0)
    assert block.ops.index(next_mm) < block.ops.index(stage0)
    assert block.ops.index(stage0) < block.ops.index(next_mm_wait)
