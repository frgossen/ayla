"""Tests for output-store pipelining."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import AllocOp, AsyncOp, AwaitOp, MoveOp, SmemTensorType, gpu_dialect
from ayla.passes.pipeline_store import pipeline_store


def test_pipeline_store_stages_tma_and_delays_waits():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @store(%c: gmem_tensor<256x256xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %c0: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<256x256xbf16>, 0, 0)
            %d0: gmem_tensor<64x256xbf16> = tile(%c0: gmem_tensor<128x256xbf16>, %wgx: index, 0)
            %a0: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            move(%d0: gmem_tensor<64x256xbf16>, %a0: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "store_iter0"}
            %c1: gmem_tensor<128x256xbf16> = tile(%c: gmem_tensor<256x256xbf16>, 1, 0)
            %d1: gmem_tensor<64x256xbf16> = tile(%c1: gmem_tensor<128x256xbf16>, %wgx: index, 0)
            %a1: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            move(%d1: gmem_tensor<64x256xbf16>, %a1: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "store_iter1"}
          }
        }"""))
    pipeline_store(module, label_filter=r"store_iter\d+")
    block = module.body.entry_block.ops[0].body.entry_block

    stage_allocs = [
        op for op in block.ops
        if isinstance(op, AllocOp)
        and isinstance(op.result().type, SmemTensorType)
    ]
    assert len(stage_allocs) == 4
    assert len(find_ops(module, lambda op: isinstance(op, AsyncOp))) == 8
    assert len(find_ops(module, lambda op: isinstance(op, AwaitOp))) == 2

    first_async = next(i for i, op in enumerate(block.ops) if isinstance(op, AsyncOp))
    first_wait = next(i for i, op in enumerate(block.ops) if isinstance(op, AwaitOp))
    second_stage = next(
        i for i, op in enumerate(block.ops)
        if isinstance(op, MoveOp)
        and op.attrs.get("label") == "store_iter1_stage_s0"
    )
    assert first_async < first_wait < second_stage
