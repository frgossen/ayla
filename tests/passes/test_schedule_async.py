"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
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
    text = repr(schedule_async(lower_to_async(module)))
    store = text.index('label: "store_async"')
    compute = text.index('label: "next_compute"')
    await_store = text.index('label: "store_await"')
    reuse = text.index('label: "reuse"')
    assert store < compute < await_store < reuse


def test_conditional_is_an_async_scheduling_boundary():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%g0: gmem<4x4xf32>, %g1: gmem<4x4xf32>) grid_dim=[2, 1, 1] block_dim=[128, 1, 1] {
            %s0: smem<4x4xf32> = alloc() attrs={id: 0}
            %s1: smem<4x4xf32> = alloc() attrs={id: 1}
            %l0: lock<mbar> = alloc() attrs={id: 2}
            %l1: lock<mbar> = alloc() attrs={id: 3}
            %bx: index, %by: index, %bz: index = block_ids()
            %c1: index = index_constant() attrs={value: 1}
            %condition: pred = index_lt(%bx: index, %c1: index)
            async(%l0: lock<mbar>) attrs={label: "before_async"} move(%s0: smem<4x4xf32>, %g0: gmem<4x4xf32>) attrs={tma: True}
            await(%l0: lock<mbar>) attrs={label: "before_await"}
            if(%condition: pred) {
              %guarded: index = index_constant() attrs={value: 7}
            }
            async(%l1: lock<mbar>) attrs={label: "after_async"} move(%s1: smem<4x4xf32>, %g1: gmem<4x4xf32>) attrs={tma: True}
            await(%l1: lock<mbar>) attrs={label: "after_await"}
          }
        }"""))  # noqa: E501

    text = repr(schedule_async(module))

    assert text.index('label: "before_await"') < text.index("if(")
    assert text.index("if(") < text.index('label: "after_async"')
