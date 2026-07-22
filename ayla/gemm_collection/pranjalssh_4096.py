"""The pranjalssh_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call


def pranjalssh_4096_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @pranjalssh_4096(%a: gmem_tensor<4096x4096xbf16>, %bt: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[32, 32, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %smem_a: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %smem_b: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            %a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<4096x4096xbf16>, %by: index, 0)
            %bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]> = tile(%bt: gmem_tensor<4096x4096xbf16>, %bx: index, 0)
            for (%k: index = 0 to 64) {
              %a_tile: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %k: index)
              %bt_tile: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %k: index)
              move(%smem_a: smem_tensor<128x64xbf16 swizzle="128B">, %a_tile: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              move(%smem_b: smem_tensor<128x64xbf16 swizzle="128B">, %bt_tile: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              for (%ki: index = 0 to 4) {
                %a_inner: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki: index)
                %b_inner: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki: index)
                matmul(%a_inner: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              }
            }
            %acc_bf: wrmem_tensor<128x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            convert(%acc_bf: wrmem_tensor<128x128xbf16 warps=4 layout="wgmma_layout_0">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            %c_block: gmem_tensor<128x128xbf16 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xbf16>, %by: index, %bx: index)
            move(%c_block: gmem_tensor<128x128xbf16 strides=[4096, 1]>, %acc_bf: wrmem_tensor<128x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse(module_str)


def _register_pranjalssh_4096() -> None:
    module = pranjalssh_4096_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


# Async variant of pranjalssh_4096: issues both TMA loads concurrently before
# awaiting them, and uses async_matmul + await (the natural wgmma form).


_register_pranjalssh_4096()
