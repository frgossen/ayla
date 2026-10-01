"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

from ayla.analysis.sync import SyncAnalysis
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AsyncOp,
    AwaitOp,
    ModuleOp,
    gpu_dialect,
)
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.schedule_async import schedule_async


def test_bulk_await_sinks_until_shared_source_is_overwritten():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%g: gmem<4x4xf32>, %a: rmem<1x1xf32>, %b: rmem<1x1xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %s: smem<4x4xf32> = alloc() attrs={id: 0}
            %r: rmem<4x4xf32> = alloc() attrs={id: 1}
            %acc: rmem<1x1xf32> = alloc() attrs={id: 2, init: 0.0}
            move(%g: gmem<4x4xf32>, %s: smem<4x4xf32>) attrs={label: "store", tma: true}
            matmul(%a: rmem<1x1xf32>, %b: rmem<1x1xf32>, %acc: rmem<1x1xf32>) attrs={label: "next_compute"}
            move(%s: smem<4x4xf32>, %r: rmem<4x4xf32>) attrs={label: "reuse"}
          }
        }"""))
    assert isinstance(module, ModuleOp)
    text = repr(schedule_async(lower_to_async(module)))
    store = text.index('label: "store_async"')
    compute = text.index('label: "next_compute"')
    await_store = text.index('label: "store_await"')
    reuse = text.index('label: "reuse"')
    assert store < compute < await_store < reuse


def test_partial_wait_completes_only_its_listed_lock():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @k(%a: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %c1: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) {
            %l0: lock<wgmma> = alloc()
            %l1: lock<wgmma> = alloc()
            async(%l0: lock<wgmma>) matmul(%a: smem<64x16xbf16>, %a: smem<64x16xbf16>, %c0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
            async(%l1: lock<wgmma>) matmul(%b: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c1: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
            await(%l0: lock<wgmma>) attrs={pending: 1}
          }
        }"""))
    issues = find_ops_of_type(module, AsyncOp)
    [wait] = find_ops_of_type(module, AwaitOp)
    analysis = SyncAnalysis()
    analysis.analyze(module)

    pending = {
        effect.pending_await
        for effects in analysis.live_out(wait).concurrent_memory_effects.values()
        for effect in effects
    }
    assert issues[0].lock not in pending
    assert issues[1].lock in pending
