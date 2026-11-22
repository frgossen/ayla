"""Queued completion must preserve refill data and event dependencies."""

import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import AsyncOp, AwaitOp, gpu_dialect
from ayla.passes.queue_async_tail import QueueAsyncTail


def skeleton(dst="a0", waited="input1"):
    return gpu_dialect.parse_module(f"""module {{
      kernel @test(%g: gmem<64x16xbf16>) {{
        %a0: smem<64x16xbf16> = alloc() attrs={{id: 1}}
        %a1: smem<64x16xbf16> = alloc() attrs={{id: 2}}
        %b0: smem<64x16xbf16> = alloc() attrs={{id: 3}}
        %b1: smem<64x16xbf16> = alloc() attrs={{id: 4}}
        %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc()
        %old: lock<wgmma> = alloc()
        %next: lock<wgmma> = alloc()
        %input0: lock<mbar> = alloc()
        %input1: lock<mbar> = alloc()
        async(%input1: lock<mbar>) attrs={{label: "prefill"}}
          move(%a1: smem<64x16xbf16>, %g: gmem<64x16xbf16>)
        async(%old: lock<wgmma>) attrs={{label: "old"}}
          matmul(%a0: smem<64x16xbf16>, %b0: smem<64x16xbf16>, %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
        await(%old: lock<wgmma>)
        async(%input0: lock<mbar>) attrs={{label: "refill"}}
          move(%{dst}: smem<64x16xbf16>, %g: gmem<64x16xbf16>)
        await(%{waited}: lock<mbar>)
        async(%next: lock<wgmma>) attrs={{label: "next"}}
          matmul(%a1: smem<64x16xbf16>, %b1: smem<64x16xbf16>, %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
        await(%next: lock<wgmma>)
      }}
    }}""")


def test_queue_retains_full_final_drain_and_retires_before_refill():
    module = skeleton()
    QueueAsyncTail().run(module)
    ops = module.body.ops[0].body.ops
    issues = {op.attrs.get("label"): op for op in find_ops_of_type(module, AsyncOp)}
    waits = [
        op for op in find_ops_of_type(module, AwaitOp) if op.lock.type.kind == "wgmma"
    ]
    assert waits[0].attrs["pending"] == 1 and not waits[-1].attrs.get("pending")
    assert ops.index(issues["next"]) < ops.index(waits[0]) < ops.index(issues["refill"])
    before = str(module)
    QueueAsyncTail().run(module)
    assert str(module) == before


@pytest.mark.parametrize("dst,waited", [("a1", "input1"), ("a0", "input0")])
def test_queue_does_not_move_refill_after_its_reader_or_await(dst, waited):
    module = skeleton(dst, waited)
    before = str(module)
    QueueAsyncTail().run(module)
    assert str(module) == before
