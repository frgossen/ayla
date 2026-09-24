"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.schedule_async import schedule_async


def test_pre_lowered_tma_load_hoists_across_independent_compute():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>, %b: gmem<4x4xf32>, %w: smem<4x4xf32>, %out: rmem<4x4xf32>) {
            %first: smem<4x4xf32> = alloc() attrs={id: 0}
            %second: smem<4x4xf32> = alloc() attrs={id: 1}
            move(%first: smem<4x4xf32>, %a: gmem<4x4xf32>) attrs={label: "first", tma: true}
            matmul(%first: smem<4x4xf32>, %w: smem<4x4xf32>, %out: rmem<4x4xf32>) attrs={label: "compute"}
            move(%second: smem<4x4xf32>, %b: gmem<4x4xf32>) attrs={label: "second", tma: true}
          }
        }"""))  # noqa: E501

    text = repr(schedule_async(module))
    assert text.index('label: "second"') < text.index('label: "compute"')


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
