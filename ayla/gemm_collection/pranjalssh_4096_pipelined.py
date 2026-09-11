"""The pranjalssh_4096_pipelined GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from .registry import register_gemm


@register_gemm
def pranjalssh_4096_pipelined_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @pranjalssh_4096_pipelined(%a: gmem<4096x4096xbf16>, %bt: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 32, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %smem_a0: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %smem_b0: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            %smem_a1: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 2}
            %smem_b1: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 3}
            %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 4, init: 0.0}
            %a_block: gmem<128x4096xbf16 strides=[4096, 1]> = tile(%a: gmem<4096x4096xbf16>, %by: index, 0)
            %bt_block: gmem<128x4096xbf16 strides=[4096, 1]> = tile(%bt: gmem<4096x4096xbf16>, %bx: index, 0)
            %a_tile_pro: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, 0)
            %bt_tile_pro: gmem<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, 0)
            %lock_a0_pro: lock<mbar> = alloc() attrs={id: 5}
            async(%lock_a0_pro: lock<mbar>) move(%smem_a0: smem<128x64xbf16 swizzle="128B">, %a_tile_pro: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            %lock_b0_pro: lock<mbar> = alloc() attrs={id: 6}
            async(%lock_b0_pro: lock<mbar>) move(%smem_b0: smem<128x64xbf16 swizzle="128B">, %bt_tile_pro: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            await(%lock_a0_pro: lock<mbar>)
            await(%lock_b0_pro: lock<mbar>)
            for (%k: index = 0 to 62 step 2) {
              %c1: index = index_constant() attrs={value: 1}
              %c2: index = index_constant() attrs={value: 2}
              %k1: index = index_add(%k: index, %c1: index)
              %k2: index = index_add(%k: index, %c2: index)
              %a_tile_1: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %k1: index)
              %bt_tile_1: gmem<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %k1: index)
              %lock_a1: lock<mbar> = alloc() attrs={id: 7}
              async(%lock_a1: lock<mbar>) move(%smem_a1: smem<128x64xbf16 swizzle="128B">, %a_tile_1: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              %lock_b1: lock<mbar> = alloc() attrs={id: 8}
              async(%lock_b1: lock<mbar>) move(%smem_b1: smem<128x64xbf16 swizzle="128B">, %bt_tile_1: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              for (%ki0: index = 0 to 4) {
                %a_inner_0: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a0: smem<128x64xbf16 swizzle="128B">, 0, %ki0: index)
                %b_inner_0: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b0: smem<128x64xbf16 swizzle="128B">, 0, %ki0: index)
                matmul(%a_inner_0: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_0: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) attrs={collective: {x: "block", y: "block", z: "thread"}}
              }
              %a_tile_next: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %k2: index)
              %bt_tile_next: gmem<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %k2: index)
              %lock_a0_next: lock<mbar> = alloc() attrs={id: 9}
              async(%lock_a0_next: lock<mbar>) move(%smem_a0: smem<128x64xbf16 swizzle="128B">, %a_tile_next: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              %lock_b0_next: lock<mbar> = alloc() attrs={id: 10}
              async(%lock_b0_next: lock<mbar>) move(%smem_b0: smem<128x64xbf16 swizzle="128B">, %bt_tile_next: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              await(%lock_a1: lock<mbar>)
              await(%lock_b1: lock<mbar>)
              for (%ki1: index = 0 to 4) {
                %a_inner_1: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a1: smem<128x64xbf16 swizzle="128B">, 0, %ki1: index)
                %b_inner_1: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b1: smem<128x64xbf16 swizzle="128B">, 0, %ki1: index)
                matmul(%a_inner_1: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_1: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) attrs={collective: {x: "block", y: "block", z: "thread"}}
              }
              await(%lock_a0_next: lock<mbar>)
              await(%lock_b0_next: lock<mbar>)
            }
            %c63: index = index_constant() attrs={value: 63}
            %a_tile_e1: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %c63: index)
            %bt_tile_e1: gmem<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem<128x4096xbf16 strides=[4096, 1]>, 0, %c63: index)
            %lock_a1_e: lock<mbar> = alloc() attrs={id: 11}
            async(%lock_a1_e: lock<mbar>) move(%smem_a1: smem<128x64xbf16 swizzle="128B">, %a_tile_e1: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            %lock_b1_e: lock<mbar> = alloc() attrs={id: 12}
            async(%lock_b1_e: lock<mbar>) move(%smem_b1: smem<128x64xbf16 swizzle="128B">, %bt_tile_e1: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            for (%ki0e: index = 0 to 4) {
              %a_inner_0e: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a0: smem<128x64xbf16 swizzle="128B">, 0, %ki0e: index)
              %b_inner_0e: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b0: smem<128x64xbf16 swizzle="128B">, 0, %ki0e: index)
              matmul(%a_inner_0e: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_0e: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) attrs={collective: {x: "block", y: "block", z: "thread"}}
            }
            await(%lock_a1_e: lock<mbar>)
            await(%lock_b1_e: lock<mbar>)
            for (%ki1e: index = 0 to 4) {
              %a_inner_1e: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a1: smem<128x64xbf16 swizzle="128B">, 0, %ki1e: index)
              %b_inner_1e: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b1: smem<128x64xbf16 swizzle="128B">, 0, %ki1e: index)
              matmul(%a_inner_1e: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_1e: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) attrs={collective: {x: "block", y: "block", z: "thread"}}
            }
            %acc_bf: wrmem<128x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 13}
            elementwise(%acc_bf: wrmem<128x128xbf16 warps=4 layout="wgmma_layout_0">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) attrs={fn: "convert"}
            %c_block: gmem<128x128xbf16 strides=[4096, 1]> = tile(%c: gmem<4096x4096xbf16>, %by: index, %bx: index)
            move(%c_block: gmem<128x128xbf16 strides=[4096, 1]>, %acc_bf: wrmem<128x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse(module_str)
