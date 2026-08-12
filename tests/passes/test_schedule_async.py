"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.schedule_async import schedule_async


def test_bulk_await_sinks_until_shared_source_is_overwritten():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%g: gmem_tensor<4x4xf32>, %a: rmem_tensor<1x1xf32>, %b: rmem_tensor<1x1xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %s: smem_tensor<4x4xf32> = alloc()
            %r: rmem_tensor<4x4xf32> = alloc()
            %acc: rmem_tensor<1x1xf32> = alloc() attrs={init: 0.0}
            move(%g: gmem_tensor<4x4xf32>, %s: smem_tensor<4x4xf32>) attrs={label: "store", tma: true}
            matmul(%a: rmem_tensor<1x1xf32>, %b: rmem_tensor<1x1xf32>, %acc: rmem_tensor<1x1xf32>) attrs={label: "next_compute"}
            move(%s: smem_tensor<4x4xf32>, %r: rmem_tensor<4x4xf32>) attrs={label: "reuse"}
          }
        }"""))
    text = repr(schedule_async(lower_to_async(module)))
    store = text.index('label: "store_async"')
    compute = text.index('label: "next_compute"')
    await_store = text.index('label: "store_await"')
    reuse = text.index('label: "reuse"')
    assert store < compute < await_store < reuse
