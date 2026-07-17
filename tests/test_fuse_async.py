"""Tests for the fuse_async pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.fuse_async import fuse_async


def test_fuse_same_accumulator():
    """Matmuls writing to the same accumulator fuse — wgmma guarantees
    ordering within a commit group."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc()
            async(%l0: lock<wgmma>) matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            await(%l0: lock<wgmma>)
            %l1: lock<wgmma> = alloc()
            async(%l1: lock<wgmma>) matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            await(%l1: lock<wgmma>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = fuse_async(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc()
            %l1: lock<wgmma> = alloc()
            async(%l0: lock<wgmma>) {
              matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            }
            await(%l0: lock<wgmma>)
          }
        }""")
    assert text == expected


def test_fuse_tma_loads():
    """Consecutive TMA loads (mbar) fuse onto one mbarrier + one await."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<128x4096xbf16>, %b: gmem_tensor<128x4096xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s0: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %s1: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %c0: index = index_constant() attrs={value: 0}
            %a0: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<128x4096xbf16>, 0, %c0: index)
            %b0: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem_tensor<128x4096xbf16>, 0, %c0: index)
            %l0: lock<mbar> = alloc()
            async(%l0: lock<mbar>) move(%s0: smem_tensor<128x64xbf16 swizzle="128B">, %a0: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem", nc: True, tma: True}
            await(%l0: lock<mbar>)
            %l1: lock<mbar> = alloc()
            async(%l1: lock<mbar>) move(%s1: smem_tensor<128x64xbf16 swizzle="128B">, %b0: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem", nc: True, tma: True}
            await(%l1: lock<mbar>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = fuse_async(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<128x4096xbf16>, %b: gmem_tensor<128x4096xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s0: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %s1: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %c0: index = index_constant() attrs={value: 0}
            %a0: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<128x4096xbf16>, 0, %c0: index)
            %b0: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem_tensor<128x4096xbf16>, 0, %c0: index)
            %l0: lock<mbar> = alloc()
            %l1: lock<mbar> = alloc()
            async(%l0: lock<mbar>) {
              move(%s0: smem_tensor<128x64xbf16 swizzle="128B">, %a0: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem", nc: True, tma: True}
              move(%s1: smem_tensor<128x64xbf16 swizzle="128B">, %b0: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem", nc: True, tma: True}
            }
            await(%l0: lock<mbar>)
          }
        }""")
    assert text == expected


def test_fuse_different_accumulators():
    """Matmuls writing to different accumulators can fuse."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c0: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">, %c1: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc()
            async(%l0: lock<wgmma>) matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c0: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            await(%l0: lock<wgmma>)
            %l1: lock<wgmma> = alloc()
            async(%l1: lock<wgmma>) matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c1: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            await(%l1: lock<wgmma>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = fuse_async(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c0: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">, %c1: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %l0: lock<wgmma> = alloc()
            %l1: lock<wgmma> = alloc()
            async(%l0: lock<wgmma>) {
              matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c0: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              matmul(%a: smem_tensor<128x16xbf16 swizzle="128B">, %b: smem_tensor<128x16xbf16 swizzle="128B">, %c1: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            }
            await(%l0: lock<wgmma>)
          }
        }""")
    assert text == expected
