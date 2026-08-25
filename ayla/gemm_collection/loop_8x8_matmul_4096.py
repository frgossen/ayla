"""The loop_8x8_matmul_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from .registry import register


# Based on https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_3.cuh
# BM=128, BN=128, BK=64, 128 threads.
@register
def loop_8x8_matmul_4096_module():
    return gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @loop_8x8_matmul_4096(%a: gmem_tensor<4096x4096xf32>, %b: gmem_tensor<4096x4096xf32>, %c: gmem_tensor<4096x4096xf32>) grid_dim=[32, 32, 1] block_dim=[16, 16, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem_tensor<128x4096xf32> = tile(%a: gmem_tensor<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem_tensor<128x4096xf32> = tile(%b: gmem_tensor<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem_tensor<128x128xf32 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem_tensor<8x4096xf32> = tile(%a_block: gmem_tensor<128x4096xf32>, %ty: index, 0)
            %b_thread: gmem_tensor<8x4096xf32> = tile(%b_block: gmem_tensor<128x4096xf32>, %tx: index, 0)
            %c_thread: gmem_tensor<8x8xf32 strides=[4096, 1]> = tile(%c_block: gmem_tensor<128x128xf32 strides=[4096, 1]>, %ty: index, %tx: index)
            %c_rmem: rmem_tensor<8x8xf32> = alloc() attrs={init: 0.0}
            for (%k: index = 0 to 4096) {
              %a_k: gmem_tensor<8x1xf32 strides=[4096, 1]> = tile(%a_thread: gmem_tensor<8x4096xf32>, 0, %k: index)
              %b_k: gmem_tensor<8x1xf32 strides=[4096, 1]> = tile(%b_thread: gmem_tensor<8x4096xf32>, 0, %k: index)
              %a_rmem: rmem_tensor<8x1xf32> = alloc()
              move(%a_rmem: rmem_tensor<8x1xf32>, %a_k: gmem_tensor<8x1xf32 strides=[4096, 1]>) attrs={nc: true}
              %b_rmem: rmem_tensor<8x1xf32> = alloc()
              move(%b_rmem: rmem_tensor<8x1xf32>, %b_k: gmem_tensor<8x1xf32 strides=[4096, 1]>) attrs={nc: true}
              matmul(%a_rmem: rmem_tensor<8x1xf32>, %b_rmem: rmem_tensor<8x1xf32>, %c_rmem: rmem_tensor<8x8xf32>)
            }
            move(%c_thread: gmem_tensor<8x8xf32 strides=[4096, 1]>, %c_rmem: rmem_tensor<8x8xf32>)
          }
        }"""))  # noqa: E501
