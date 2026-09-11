"""The pranjalssh_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from .registry import register_gemm


@register_gemm
def pranjalssh_4096_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @pranjalssh_4096(%a: gmem<4096x4096xbf16>, %bt: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 32, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %smem_a: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %smem_b: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 2, init: 0.0}
            %a_block: gmem<128x4096xbf16 strides=[4096, 1]> = tile(%a: gmem<4096x4096xbf16>, %by: index, 0)
            %bt_block: gmem<128x4096xbf16 strides=[4096, 1]> = tile(%bt: gmem<4096x4096xbf16>, %bx: index, 0)
            for (%k: index = 0 to 64) {
              %a_tile: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %k: index)
              %bt_tile: gmem<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %k: index)
              move(%smem_a: smem<128x64xbf16 swizzle="128B">, %a_tile: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              move(%smem_b: smem<128x64xbf16 swizzle="128B">, %bt_tile: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              for (%ki: index = 0 to 4) {
                %a_inner: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a: smem<128x64xbf16 swizzle="128B">, 0, %ki: index)
                %b_inner: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b: smem<128x64xbf16 swizzle="128B">, 0, %ki: index)
                matmul(%a_inner: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
              }
            }
            %acc_bf: wrmem<128x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 3}
            elementwise(%acc_bf: wrmem<128x128xbf16 warps=4 layout="wgmma_layout_0">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) attrs={fn: "convert"}
            %c_block: gmem<128x128xbf16 strides=[4096, 1]> = tile(%c: gmem<4096x4096xbf16>, %by: index, %bx: index)
            move(%c_block: gmem<128x128xbf16 strides=[4096, 1]>, %acc_bf: wrmem<128x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse(module_str)


# Async variant of pranjalssh_4096: issues both TMA loads concurrently before
# awaiting them, and uses async_matmul + await (the natural wgmma form).
