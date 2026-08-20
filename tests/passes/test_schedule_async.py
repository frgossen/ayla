"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.ptx_codegen import PtxLoweringPipeline
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.lower_to_tma_move import LowerToTmaMove, lower_to_tma_move
from ayla.passes.schedule_async import ScheduleAsync, schedule_async


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


def test_lowered_tma_address_math_hoists_without_hoisting_the_load():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%g: gmem_tensor<128x128xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s: smem_tensor<64x64xbf16 swizzle="128B"> = alloc()
            %rhs: smem_tensor<64x64xbf16 swizzle="128B"> = alloc()
            %reuse: rmem_tensor<64x64xbf16> = alloc()
            %acc: wrmem_tensor<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            %wlock: lock<wgmma> = alloc()
            %c1: index = index_constant() attrs={value: 1}
            %tile: gmem_tensor<64x64xbf16 strides=[128, 1]> = tile(%g: gmem_tensor<128x128xbf16>, 0, %c1: index)
            async(%wlock: lock<wgmma>) attrs={label: "compute_async"} matmul(%s: smem_tensor<64x64xbf16 swizzle="128B">, %rhs: smem_tensor<64x64xbf16 swizzle="128B">, %acc: wrmem_tensor<64x64xf32 warps=4 layout="wgmma_layout_0">)
            await(%wlock: lock<wgmma>) attrs={label: "compute_await"}
            move(%s: smem_tensor<64x64xbf16 swizzle="128B">, %reuse: rmem_tensor<64x64xbf16>) attrs={label: "stage_reuse"}
            %tlock: lock<mbar> = alloc()
            async(%tlock: lock<mbar>) attrs={label: "load_async"} move(%s: smem_tensor<64x64xbf16 swizzle="128B">, %tile: gmem_tensor<64x64xbf16 strides=[128, 1]>) attrs={tma: true}
            await(%tlock: lock<mbar>) attrs={label: "load_await"}
          }
        }"""))

    text = repr(schedule_async(lower_to_tma_move(module)))
    address = text.index("index_mul(%c1")
    compute_await = text.index('label: "compute_await"')
    stage_reuse = text.index('label: "stage_reuse"')
    tma_lock = text.index("%tlock: lock<mbar> = alloc()")
    load = text.index('label: "load_async"')

    assert address < compute_await < stage_reuse < tma_lock < load


def test_ptx_pipeline_schedules_coordinates_immediately_after_tma_lowering():
    passes = PtxLoweringPipeline().passes
    lower_index = next(
        i for i, compiler_pass in enumerate(passes)
        if isinstance(compiler_pass, LowerToTmaMove)
    )
    schedule = passes[lower_index + 1]

    assert isinstance(schedule, ScheduleAsync)
    assert not schedule.sink_awaits
