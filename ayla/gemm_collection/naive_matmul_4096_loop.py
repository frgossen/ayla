"""The naive_matmul_4096_loop GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call


def naive_matmul_4096_loop_module():
    """Naive per-thread 4096x4096 f32 matmul using tile + move + fma."""
    module_str = textwrap.dedent("""\
        module {
          kernel @naive_matmul_4096_loop(%a: gmem_tensor<4096x4096xf32>, %b: gmem_tensor<4096x4096xf32>, %c: gmem_tensor<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem_tensor<64x4096xf32> = tile(%a: gmem_tensor<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem_tensor<64x4096xf32> = tile(%b: gmem_tensor<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem_tensor<64x64xf32 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem_tensor<2x4096xf32> = tile(%a_block: gmem_tensor<64x4096xf32>, %tx: index, 0)
            %b_thread: gmem_tensor<2x4096xf32> = tile(%b_block: gmem_tensor<64x4096xf32>, %ty: index, 0)
            %c_thread: gmem_tensor<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem_tensor<64x64xf32 strides=[4096, 1]>, %tx: index, %ty: index)
            %c_thread_rmem: rmem_tensor<2x2xf32> = alloc() attrs={init: 0.0}
            for (%k: index = 0 to 4096) {
              %a_k: gmem_tensor<2x1xf32 strides=[4096, 1]> = tile(%a_thread: gmem_tensor<2x4096xf32>, 0, %k: index)
              %b_k: gmem_tensor<2x1xf32 strides=[4096, 1]> = tile(%b_thread: gmem_tensor<2x4096xf32>, 0, %k: index)
              %a_k_rmem: rmem_tensor<2x1xf32> = alloc()
              move(%a_k_rmem: rmem_tensor<2x1xf32>, %a_k: gmem_tensor<2x1xf32 strides=[4096, 1]>) attrs={nc: true}
              %b_k_rmem: rmem_tensor<2x1xf32> = alloc()
              move(%b_k_rmem: rmem_tensor<2x1xf32>, %b_k: gmem_tensor<2x1xf32 strides=[4096, 1]>) attrs={nc: true}
              matmul(%a_k_rmem: rmem_tensor<2x1xf32>, %b_k_rmem: rmem_tensor<2x1xf32>, %c_thread_rmem: rmem_tensor<2x2xf32>)
            }
            move(%c_thread: gmem_tensor<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem_tensor<2x2xf32>)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse(module_str)


def _register_naive_matmul_4096_loop() -> None:
    module = naive_matmul_4096_loop_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


_register_naive_matmul_4096_loop()
