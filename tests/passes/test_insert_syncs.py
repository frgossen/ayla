"""Tests for the cross-thread sync-barrier insertion pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.insert_syncs import insert_syncs


def test_war_on_shared_buffer_gets_barrier():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<128x4096xbf16>, %b: gmem_tensor<128x4096xbf16>, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %sa: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %sb: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %ak: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<128x4096xbf16>, 0, %k: index)
              %bk: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem_tensor<128x4096xbf16>, 0, %k: index)
              %lm: lock<mbar> = alloc()
              async(%lm: lock<mbar>) {
                move(%sa: smem_tensor<128x64xbf16 swizzle="128B">, %ak: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
                move(%sb: smem_tensor<128x64xbf16 swizzle="128B">, %bk: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
              }
              await(%lm: lock<mbar>)
              %lw: lock<wgmma> = alloc()
              async(%lw: lock<wgmma>) matmul(%sa: smem_tensor<128x64xbf16 swizzle="128B">, %sb: smem_tensor<128x64xbf16 swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              await(%lw: lock<wgmma>)
            }
          }
        }"""))  # noqa: E501

    insert_syncs(module)

    print(repr(module))

    # A block-scoped barrier is inserted before each async region that reuses a
    # shared buffer: before the TMA loads (WAR on %sa/%sb vs. the prior matmul's
    # reads) and before the matmul (RAW on the freshly loaded %sa/%sb).
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<128x4096xbf16>, %b: gmem_tensor<128x4096xbf16>, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] attrs={explicit_barriers: True} {
            %sa: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %sb: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %ak: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<128x4096xbf16>, 0, %k: index)
              %bk: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%b: gmem_tensor<128x4096xbf16>, 0, %k: index)
              %lm: lock<mbar> = alloc()
              barrier() attrs={scope: "block"}
              async(%lm: lock<mbar>) {
                move(%sa: smem_tensor<128x64xbf16 swizzle="128B">, %ak: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
                move(%sb: smem_tensor<128x64xbf16 swizzle="128B">, %bk: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
              }
              await(%lm: lock<mbar>)
              %lw: lock<wgmma> = alloc()
              barrier() attrs={scope: "block"}
              async(%lw: lock<wgmma>) matmul(%sa: smem_tensor<128x64xbf16 swizzle="128B">, %sb: smem_tensor<128x64xbf16 swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              await(%lw: lock<wgmma>)
            }
          }
        }""")  # noqa: E501
