"""Tests for the lower_to_async pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.lower_to_async import lower_to_async


def test_sync_move_stays_as_move():
    """Non-TMA move stays as MoveOp (not converted to async)."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %smem: smem<128x64xf32> = alloc() attrs={id: 0}
            move(%smem: smem<128x64xf32>, %a: gmem<4096x4096xf32>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    lowered = lower_to_async(module)
    text = repr(lowered)

    assert text == module_str


def test_tma_to_async_move_await():
    """TMA-eligible move becomes async { move } + await with a lock."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %smem: smem<128x64xf32> = alloc() attrs={id: 0}
            move(%smem: smem<128x64xf32>, %a: gmem<4096x4096xf32>) attrs={tma: True}
          }
        }""")

    module = gpu_dialect.parse(module_str)
    lowered = lower_to_async(module)
    text = repr(lowered)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %smem: smem<128x64xf32> = alloc() attrs={id: 0}
            %0: lock<mbar> = alloc() attrs={id: 0}
            async(%0: lock<mbar>) move(%smem: smem<128x64xf32>, %a: gmem<4096x4096xf32>) attrs={tma: True}
            await(%0: lock<mbar>)
          }
        }""")
    assert text == expected


def test_sync_matmul_stays_as_matmul():
    """Non-wgmma matmul stays as MatmulOp."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %lhs: rmem<2x2xf32> = alloc() attrs={id: 0}
            %rhs: rmem<2x2xf32> = alloc() attrs={id: 1}
            %dst: rmem<2x2xf32> = alloc() attrs={id: 2}
            matmul(%lhs: rmem<2x2xf32>, %rhs: rmem<2x2xf32>, %dst: rmem<2x2xf32>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    lowered = lower_to_async(module)
    text = repr(lowered)

    assert text == module_str


def test_wgmma_matmul_to_async_await():
    """wgmma-eligible matmul becomes async { matmul } + await with a lock."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
          }
        }""")

    module = gpu_dialect.parse(module_str)
    lowered = lower_to_async(module)
    text = repr(lowered)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %0: lock<wgmma> = alloc() attrs={id: 0}
            async(%0: lock<wgmma>) collective=[xy.block, z.thread] matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<128x16xbf16 swizzle="128B">, %c: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            await(%0: lock<wgmma>)
          }
        }""")
    assert text == expected
