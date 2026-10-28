"""Tests for scheduling asynchronous work around independent operations."""

import textwrap

from typing import cast

from ayla.gpu_dialect import AllocOp, KernelOp, AsyncOp, ModuleOp, gpu_dialect
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.schedule_async import schedule_async


from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import AwaitOp
from ayla.analysis.sync import SyncAnalysis


def test_bulk_await_sinks_until_shared_source_is_overwritten():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @test(%g: gmem<4x4xf32>, %a: rmem<1x1xf32>, %b: rmem<1x1xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %s: smem<4x4xf32> = alloc() attrs={id: 0}
            %r: rmem<4x4xf32> = alloc() attrs={id: 1}
            %acc: rmem<1x1xf32> = alloc() attrs={id: 2}
            %acc_init: rmem<f32> = constant 0.0
            elementwise fill(%acc: rmem<1x1xf32>, %acc_init: rmem<f32>)
            move(%g: gmem<4x4xf32>, %s: smem<4x4xf32>) attrs={label: "store", tma: true}
            matmul(%a: rmem<1x1xf32>, %b: rmem<1x1xf32>, %acc: rmem<1x1xf32>) attrs={label: "next_compute"}
            move(%s: smem<4x4xf32>, %r: rmem<4x4xf32>) attrs={label: "reuse"}
          }
        }""")
    )
    assert isinstance(module, ModuleOp)
    text = repr(schedule_async(lower_to_async(module)))
    store = text.index('label: "store_async"')
    compute = text.index('label: "next_compute"')
    await_store = text.index('label: "store_await"')
    reuse = text.index('label: "reuse"')
    assert store < compute < await_store < reuse


def test_partial_wait_completes_only_its_listed_lock():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k(%a: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %c1: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) {
            %l0: lock<wgmma> = alloc()
            %l1: lock<wgmma> = alloc()
            async(%l0: lock<wgmma>) matmul(%a: smem<64x16xbf16>, %a: smem<64x16xbf16>, %c0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
            async(%l1: lock<wgmma>) matmul(%b: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c1: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
            await(%l0: lock<wgmma>) attrs={pending: 1}
          }
        }""")
    )
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


def test_hoisted_loads_leave_lock_allocations_together():
    # Each load rises to just below its own allocations; the allocations
    # then return to the front so one barrier publishes both locks.
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @k(%g: gmem<16x16xf32>, %r: rmem<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %s0: smem<16x16xf32> = alloc() attrs={id: 0}
            %l0: lock<mbar> = alloc() attrs={id: 1}
            %s1: smem<16x16xf32> = alloc() attrs={id: 2}
            %l1: lock<mbar> = alloc() attrs={id: 3}
            %acc: rmem<16x16xf32> = alloc() attrs={id: 4}
            %zero: rmem<f32> = constant 0.0
            elementwise fill(%acc: rmem<16x16xf32>, %zero: rmem<f32>)
            move(%r: rmem<16x16xf32>, %r: rmem<16x16xf32>)
            async(%l0: lock<mbar>) move(%s0: smem<16x16xf32>, %g: gmem<16x16xf32>) attrs={tma: True}
            async(%l1: lock<mbar>) move(%s1: smem<16x16xf32>, %g: gmem<16x16xf32>) attrs={tma: True}
            await(%l0: lock<mbar>)
            await(%l1: lock<mbar>)
          }
        }""")
    )

    schedule_async(module)

    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    ops = kernel.body.ops
    assert sorted(
        cast(int, op.attrs["id"]) for op in ops[:4] if isinstance(op, AllocOp)
    ) == [0, 1, 2, 3]
    # The loads still rise above the initializing allocation.
    first_load = next(i for i, op in enumerate(ops) if isinstance(op, AsyncOp))
    acc = next(
        i
        for i, op in enumerate(ops)
        if op.op_id == "elementwise" and op.attrs.get("fn") == "fill"
    )
    assert first_load < acc


def test_register_load_hoists_behind_async_loads():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @test(%g: gmem<4x4xf32>, %h: gmem<4x4xf32>, %a: rmem<1x1xf32>, %b: rmem<1x1xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %s: smem<4x4xf32> = alloc() attrs={id: 0}
            %r: rmem<4x4xf32> = alloc() attrs={id: 1}
            %acc: rmem<1x1xf32> = alloc() attrs={id: 2, init: 0.0}
            matmul(%a: rmem<1x1xf32>, %b: rmem<1x1xf32>, %acc: rmem<1x1xf32>) attrs={label: "compute"}
            move(%r: rmem<4x4xf32>, %h: gmem<4x4xf32>) attrs={label: "register_load"}
            move(%s: smem<4x4xf32>, %g: gmem<4x4xf32>) attrs={label: "shared_load", tma: true}
          }
        }""")
    )
    assert isinstance(module, ModuleOp)
    text = repr(schedule_async(lower_to_async(module)))
    shared_load = text.index('label: "shared_load_async"')
    register_load = text.index('label: "register_load"')
    compute = text.index('label: "compute"')
    assert shared_load < register_load < compute
