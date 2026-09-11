"""The loop_4x4_matmul_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from .registry import register_gemm


@register_gemm
def loop_4x4_matmul_4096_module():
    return gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @loop_4x4_matmul_4096(%a: gmem<4096x4096xf32>, %b: gmem<4096x4096xf32>, %c: gmem<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[16, 16, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem<64x4096xf32> = tile(%a: gmem<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem<64x4096xf32> = tile(%b: gmem<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem<64x64xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem<4x4096xf32> = tile(%a_block: gmem<64x4096xf32>, %ty: index, 0)
            %b_thread: gmem<4x4096xf32> = tile(%b_block: gmem<64x4096xf32>, %tx: index, 0)
            %c_thread: gmem<4x4xf32 strides=[4096, 1]> = tile(%c_block: gmem<64x64xf32 strides=[4096, 1]>, %ty: index, %tx: index)
            %c_rmem: rmem<4x4xf32> = alloc() attrs={id: 0, init: 0.0}
            for (%k: index = 0 to 1024) {
              %a_k: gmem<4x4xf32 strides=[4096, 1]> = tile(%a_thread: gmem<4x4096xf32>, 0, %k: index)
              %b_k: gmem<4x4xf32 strides=[4096, 1]> = tile(%b_thread: gmem<4x4096xf32>, 0, %k: index)
              %a_rmem: rmem<4x4xf32> = alloc() attrs={id: 1}
              move(%a_rmem: rmem<4x4xf32>, %a_k: gmem<4x4xf32 strides=[4096, 1]>) attrs={nc: true}
              %b_rmem: rmem<4x4xf32> = alloc() attrs={id: 2}
              move(%b_rmem: rmem<4x4xf32>, %b_k: gmem<4x4xf32 strides=[4096, 1]>) attrs={nc: true}
              matmul(%a_rmem: rmem<4x4xf32>, %b_rmem: rmem<4x4xf32>, %c_rmem: rmem<4x4xf32>)
            }
            move(%c_thread: gmem<4x4xf32 strides=[4096, 1]>, %c_rmem: rmem<4x4xf32>)
          }
        }"""))  # noqa: E501
