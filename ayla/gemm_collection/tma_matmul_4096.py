"""The tma_matmul_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from .registry import register_gemm


@register_gemm
def tma_matmul_4096_module():
    """4096x4096 f32 matmul using TMA loads.

    Same blocking as naive_matmul_4096_loop (64x64 blocks, 32x32 threads,
    2x2 per thread) but loads A/B tiles into smem via TMA before the inner
    K-loop reads them.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @tma_matmul_4096(%a: gmem<4096x4096xf32>, %b: gmem<4096x4096xf32>, %c: gmem<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index = block_id() attrs={axis: "x"}
            %cy: index = block_id() attrs={axis: "y"}
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem<64x4096xf32> = tile(%a: gmem<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem<64x4096xf32> = tile(%b: gmem<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem<64x64xf32 strides=[4096, 1]> = tile(%c: gmem<4096x4096xf32>, %cx: index, %cy: index)
            %c_thread: gmem<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem<64x64xf32 strides=[4096, 1]>, %ty: index, %tx: index)
            %c_thread_rmem: rmem<2x2xf32> = alloc() attrs={id: 0, init: 0.0}
            for (%ko: index = 0 to 64) {
              %a_ko: gmem<64x64xf32 strides=[4096, 1]> = tile(%a_block: gmem<64x4096xf32>, 0, %ko: index)
              %b_ko: gmem<64x64xf32 strides=[4096, 1]> = tile(%b_block: gmem<64x4096xf32>, 0, %ko: index)
              %a_ko_smem: smem<64x64xf32> = alloc() attrs={id: 1}
              move(%a_ko_smem: smem<64x64xf32>, %a_ko: gmem<64x64xf32 strides=[4096, 1]>) attrs={tma: True}
              %b_ko_smem: smem<64x64xf32> = alloc() attrs={id: 2}
              move(%b_ko_smem: smem<64x64xf32>, %b_ko: gmem<64x64xf32 strides=[4096, 1]>) attrs={tma: True}
              for (%ki: index = 0 to 64) {
                %a_ki: smem<2x1xf32 strides=[64, 1]> = tile(%a_ko_smem: smem<64x64xf32>, %ty: index, %ki: index)
                %b_ki: smem<2x1xf32 strides=[64, 1]> = tile(%b_ko_smem: smem<64x64xf32>, %tx: index, %ki: index)
                %a_ki_rmem: rmem<2x1xf32> = alloc() attrs={id: 3}
                move(%a_ki_rmem: rmem<2x1xf32>, %a_ki: smem<2x1xf32 strides=[64, 1]>)
                %b_ki_rmem: rmem<2x1xf32> = alloc() attrs={id: 4}
                move(%b_ki_rmem: rmem<2x1xf32>, %b_ki: smem<2x1xf32 strides=[64, 1]>)
                matmul(%a_ki_rmem: rmem<2x1xf32>, %b_ki_rmem: rmem<2x1xf32>, %c_thread_rmem: rmem<2x2xf32>)
              }
            }
            move(%c_thread: gmem<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem<2x2xf32>)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse_module(module_str)
