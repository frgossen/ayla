"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

from ayla.gpu_dialect import AsyncOp, ModuleOp, gpu_dialect
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


def test_hoisted_loads_leave_lock_allocations_together():
    # Each load rises to just below its own allocations; the allocations
    # then return to the front so one barrier publishes both locks.
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @k(%g: gmem<16x16xf32>, %r: rmem<16x16xf32>) grid_dim=[1, 1, 1] block_dim=[32, 1, 1] {
            %s0: smem<16x16xf32> = alloc() attrs={id: 0}
            %l0: lock<mbar> = alloc() attrs={id: 1}
            %s1: smem<16x16xf32> = alloc() attrs={id: 2}
            %l1: lock<mbar> = alloc() attrs={id: 3}
            %acc: rmem<16x16xf32> = alloc() attrs={id: 4, init: 0.0}
            move(%r: rmem<16x16xf32>, %r: rmem<16x16xf32>)
            async(%l0: lock<mbar>) move(%s0: smem<16x16xf32>, %g: gmem<16x16xf32>) attrs={tma: True}
            async(%l1: lock<mbar>) move(%s1: smem<16x16xf32>, %g: gmem<16x16xf32>) attrs={tma: True}
            await(%l0: lock<mbar>)
            await(%l1: lock<mbar>)
          }
        }"""))

    schedule_async(module)

    ops = module.body.ops[0].body.ops
    assert sorted(op.attrs.get("id") for op in ops[:4]) == [0, 1, 2, 3]
    # The loads still rise above the initializing allocation.
    first_load = next(i for i, op in enumerate(ops) if isinstance(op, AsyncOp))
    acc = next(i for i, op in enumerate(ops) if op.attrs.get("init") is not None)
    assert first_load < acc
