"""A small collection of pre-built GEMM (matrix multiply) kernels.

Each entry exposes:
- a `*_TEMPLATE` PTX-template string for the inner-most ptx_template op,
- a `*_module()` builder returning a parsed `ModuleOp` ready for codegen,
- a `*_torch_op()` builder returning a registered ``torch.ops`` overload.

The intended use:

    import ayla.gemm_collection  # registers all ops

    torch.ops.ayla.naive_matmul_4096(a, b)  # returns c
"""
import textwrap

import torch

from .canonicalize import canonicalize
from .fuse_async import fuse_async
from .gpu_dialect import gpu_dialect, PTXTemplateOp, TmaDescType
from .ir import Value
from .licm import licm
from .lower_to_async import lower_to_async
from .pipeline_loop import pipeline_loop
from .promote_buffers import promote_buffers
from .ptx_codegen import make_ptx_kernel
from .ptx_runtime import make_ptx_launcher
from .schedule_async import schedule_async
from .tile_matmul import tile_matmul
from .torch_custom_call import OperandKind, make_torch_custom_call
from .unroll_loop import unroll_loop
from ._fastcu_matmul_5_1_ptx import FASTCU_MATMUL_5_1_TEMPLATE



# ---------------------------------------------------------------------------
# V0b: Same naive 4096x4096 f32 matmul as V0, but written in high-level IR
# (tile/alloc/load/matmul/store + sequential K-loop) rather than as one
# hand-written ptx_template. Lowering pass + codegen produce the kernel.
# ---------------------------------------------------------------------------
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


def tma_matmul_4096_module():
    """4096x4096 f32 matmul using TMA loads.

    Same blocking as naive_matmul_4096_loop (64x64 blocks, 32x32 threads,
    2x2 per thread) but loads A/B tiles into smem via TMA before the inner
    K-loop reads them.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @tma_matmul_4096(%a: gmem_tensor<4096x4096xf32>, %b: gmem_tensor<4096x4096xf32>, %c: gmem_tensor<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[32, 32, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem_tensor<64x4096xf32> = tile(%a: gmem_tensor<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem_tensor<64x4096xf32> = tile(%b: gmem_tensor<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem_tensor<64x64xf32 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xf32>, %cx: index, %cy: index)
            %c_thread: gmem_tensor<2x2xf32 strides=[4096, 1]> = tile(%c_block: gmem_tensor<64x64xf32 strides=[4096, 1]>, %ty: index, %tx: index)
            %c_thread_rmem: rmem_tensor<2x2xf32> = alloc() attrs={init: 0.0}
            for (%ko: index = 0 to 64) {
              %a_ko: gmem_tensor<64x64xf32 strides=[4096, 1]> = tile(%a_block: gmem_tensor<64x4096xf32>, 0, %ko: index)
              %b_ko: gmem_tensor<64x64xf32 strides=[4096, 1]> = tile(%b_block: gmem_tensor<64x4096xf32>, 0, %ko: index)
              %a_ko_smem: smem_tensor<64x64xf32> = alloc()
              move(%a_ko_smem: smem_tensor<64x64xf32>, %a_ko: gmem_tensor<64x64xf32 strides=[4096, 1]>) attrs={tma: True}
              %b_ko_smem: smem_tensor<64x64xf32> = alloc()
              move(%b_ko_smem: smem_tensor<64x64xf32>, %b_ko: gmem_tensor<64x64xf32 strides=[4096, 1]>) attrs={tma: True}
              for (%ki: index = 0 to 64) {
                %a_ki: smem_tensor<2x1xf32 strides=[64, 1]> = tile(%a_ko_smem: smem_tensor<64x64xf32>, %ty: index, %ki: index)
                %b_ki: smem_tensor<2x1xf32 strides=[64, 1]> = tile(%b_ko_smem: smem_tensor<64x64xf32>, %tx: index, %ki: index)
                %a_ki_rmem: rmem_tensor<2x1xf32> = alloc()
                move(%a_ki_rmem: rmem_tensor<2x1xf32>, %a_ki: smem_tensor<2x1xf32 strides=[64, 1]>)
                %b_ki_rmem: rmem_tensor<2x1xf32> = alloc()
                move(%b_ki_rmem: rmem_tensor<2x1xf32>, %b_ki: smem_tensor<2x1xf32 strides=[64, 1]>)
                matmul(%a_ki_rmem: rmem_tensor<2x1xf32>, %b_ki_rmem: rmem_tensor<2x1xf32>, %c_thread_rmem: rmem_tensor<2x2xf32>)
              }
            }
            move(%c_thread: gmem_tensor<2x2xf32 strides=[4096, 1]>, %c_thread_rmem: rmem_tensor<2x2xf32>)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse(module_str)


def _register_tma_matmul_4096() -> None:
    module = tma_matmul_4096_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


def loop_4x4_matmul_4096_module():
    return gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @loop_4x4_matmul_4096(%a: gmem_tensor<4096x4096xf32>, %b: gmem_tensor<4096x4096xf32>, %c: gmem_tensor<4096x4096xf32>) grid_dim=[64, 64, 1] block_dim=[16, 16, 1] {
            %cx: index, %cy: index, %cz: index = block_ids()
            %tx: index, %ty: index, %tz: index = thread_ids()
            %a_block: gmem_tensor<64x4096xf32> = tile(%a: gmem_tensor<4096x4096xf32>, %cx: index, 0)
            %b_block: gmem_tensor<64x4096xf32> = tile(%b: gmem_tensor<4096x4096xf32>, %cy: index, 0)
            %c_block: gmem_tensor<64x64xf32 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xf32>, %cx: index, %cy: index)
            %a_thread: gmem_tensor<4x4096xf32> = tile(%a_block: gmem_tensor<64x4096xf32>, %ty: index, 0)
            %b_thread: gmem_tensor<4x4096xf32> = tile(%b_block: gmem_tensor<64x4096xf32>, %tx: index, 0)
            %c_thread: gmem_tensor<4x4xf32 strides=[4096, 1]> = tile(%c_block: gmem_tensor<64x64xf32 strides=[4096, 1]>, %ty: index, %tx: index)
            %c_rmem: rmem_tensor<4x4xf32> = alloc() attrs={init: 0.0}
            for (%k: index = 0 to 1024) {
              %a_k: gmem_tensor<4x4xf32 strides=[4096, 1]> = tile(%a_thread: gmem_tensor<4x4096xf32>, 0, %k: index)
              %b_k: gmem_tensor<4x4xf32 strides=[4096, 1]> = tile(%b_thread: gmem_tensor<4x4096xf32>, 0, %k: index)
              %a_rmem: rmem_tensor<4x4xf32> = alloc()
              move(%a_rmem: rmem_tensor<4x4xf32>, %a_k: gmem_tensor<4x4xf32 strides=[4096, 1]>) attrs={nc: true}
              %b_rmem: rmem_tensor<4x4xf32> = alloc()
              move(%b_rmem: rmem_tensor<4x4xf32>, %b_k: gmem_tensor<4x4xf32 strides=[4096, 1]>) attrs={nc: true}
              matmul(%a_rmem: rmem_tensor<4x4xf32>, %b_rmem: rmem_tensor<4x4xf32>, %c_rmem: rmem_tensor<4x4xf32>)
            }
            move(%c_thread: gmem_tensor<4x4xf32 strides=[4096, 1]>, %c_rmem: rmem_tensor<4x4xf32>)
          }
        }"""))  # noqa: E501


def _register_loop_4x4_matmul_4096() -> None:
    module = loop_4x4_matmul_4096_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(
        launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


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


def _register_loop_8x8_matmul_4096() -> None:
    module = loop_8x8_matmul_4096_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(
        launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])





# # ---------------------------------------------------------------------------
# # V2c: 64×64 block tile, 4 warps, register-tiled 2×2 per warp (32×32).
# # Each warp does 4 ldmatrix + 4 matmul (= 8 mma.sync) per k-iteration.
# # Fully structured — no ptx_templates.
# # Grid: 64×64 (4096/64), Block: 128, K-tile: 16, smem: 4KB.
# #
# # TODO: this needs a correct wrmem layout. we need to define these uniquely and they must match between load and gemm
# # ---------------------------------------------------------------------------
# def tiled_mma_matmul_4096_module():
#     module_str = textwrap.dedent("""\
#         module {
#           kernel @tiled_mma_matmul_4096(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[64, 64, 1] block_dim=[128, 1, 1] {
#             %bx: index, %by: index, %bz: index = block_ids()
#             %tx: index, %ty: index, %tz: index = thread_ids()
#             %c32w: index = index_constant() attrs={value: 32}
#             %wid: index = index_div(%tx: index, %c32w: index)
#             %c2: index = index_constant() attrs={value: 2}
#             %warp_row: index = index_div(%wid: index, %c2: index)
#             %warp_col: index = index_mod(%wid: index, %c2: index)
#             %smem_a: smem_tensor<64x16xbf16> = alloc()
#             %smem_b: smem_tensor<64x16xbf16> = alloc()
#             %warp_a: smem_tensor<32x16xbf16 strides=[16, 1]> = tile(%smem_a: smem_tensor<64x16xbf16>, %warp_row: index, 0)
#             %warp_b: smem_tensor<32x16xbf16 strides=[16, 1]> = tile(%smem_b: smem_tensor<64x16xbf16>, %warp_col: index, 0)
#             %warp_a_top: smem_tensor<16x16xbf16 strides=[16, 1]> = tile(%warp_a: smem_tensor<32x16xbf16 strides=[16, 1]>, 0, 0)
#             %warp_a_bot: smem_tensor<16x16xbf16 strides=[16, 1]> = tile(%warp_a: smem_tensor<32x16xbf16 strides=[16, 1]>, 1, 0)
#             %warp_b_left: smem_tensor<16x16xbf16 strides=[16, 1]> = tile(%warp_b: smem_tensor<32x16xbf16 strides=[16, 1]>, 0, 0)
#             %warp_b_right: smem_tensor<16x16xbf16 strides=[16, 1]> = tile(%warp_b: smem_tensor<32x16xbf16 strides=[16, 1]>, 1, 0)
#             %a_block: gmem_tensor<64x4096xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<4096x4096xbf16>, %by: index, 0)
#             %b_block: gmem_tensor<64x4096xbf16 strides=[4096, 1]> = tile(%b: gmem_tensor<4096x4096xbf16>, %bx: index, 0)
#             %c_00: wrmem_tensor<16x16xf32 warps=1 layout="mma"> = alloc() attrs={init: 0.0}
#             %c_01: wrmem_tensor<16x16xf32 warps=1 layout="mma"> = alloc() attrs={init: 0.0}
#             %c_10: wrmem_tensor<16x16xf32 warps=1 layout="mma"> = alloc() attrs={init: 0.0}
#             %c_11: wrmem_tensor<16x16xf32 warps=1 layout="mma"> = alloc() attrs={init: 0.0}
#             for (%k: index = 0 to 256) {
#               %a_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]> = tile(%a_block: gmem_tensor<64x4096xbf16 strides=[4096, 1]>, 0, %k: index)
#               %b_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]> = tile(%b_block: gmem_tensor<64x4096xbf16 strides=[4096, 1]>, 0, %k: index)
#               move(%smem_a: smem_tensor<64x16xbf16>, %a_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]>) attrs={tma: true}
#               move(%smem_b: smem_tensor<64x16xbf16>, %b_tile: gmem_tensor<64x16xbf16 strides=[4096, 1]>) attrs={tma: true}
#               %a_top: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#               move(%a_top: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %warp_a_top: smem_tensor<16x16xbf16 strides=[16, 1]>)
#               %a_bot: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#               move(%a_bot: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %warp_a_bot: smem_tensor<16x16xbf16 strides=[16, 1]>)
#               %b_left: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#               move(%b_left: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %warp_b_left: smem_tensor<16x16xbf16 strides=[16, 1]>) attrs={trans: true}
#               %b_right: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#               move(%b_right: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %warp_b_right: smem_tensor<16x16xbf16 strides=[16, 1]>) attrs={trans: true}
#               matmul(%a_top: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %b_left: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_00: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#               matmul(%a_top: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %b_right: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_01: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#               matmul(%a_bot: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %b_left: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_10: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#               matmul(%a_bot: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %b_right: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_11: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#             }
#             %smem_c: smem_tensor<64x64xbf16> = alloc()
#             %store_warp: smem_tensor<32x32xbf16 strides=[64, 1]> = tile(%smem_c: smem_tensor<64x64xbf16>, %warp_row: index, %warp_col: index)
#             %store_00: smem_tensor<16x16xbf16 strides=[64, 1]> = tile(%store_warp: smem_tensor<32x32xbf16 strides=[64, 1]>, 0, 0)
#             %store_01: smem_tensor<16x16xbf16 strides=[64, 1]> = tile(%store_warp: smem_tensor<32x32xbf16 strides=[64, 1]>, 0, 1)
#             %store_10: smem_tensor<16x16xbf16 strides=[64, 1]> = tile(%store_warp: smem_tensor<32x32xbf16 strides=[64, 1]>, 1, 0)
#             %store_11: smem_tensor<16x16xbf16 strides=[64, 1]> = tile(%store_warp: smem_tensor<32x32xbf16 strides=[64, 1]>, 1, 1)
#             %c_00_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#             convert(%c_00_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_00: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#             move(%store_00: smem_tensor<16x16xbf16 strides=[64, 1]>, %c_00_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">)
#             %c_01_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#             convert(%c_01_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_01: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#             move(%store_01: smem_tensor<16x16xbf16 strides=[64, 1]>, %c_01_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">)
#             %c_10_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#             convert(%c_10_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_10: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#             move(%store_10: smem_tensor<16x16xbf16 strides=[64, 1]>, %c_10_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">)
#             %c_11_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma"> = alloc()
#             convert(%c_11_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">, %c_11: wrmem_tensor<16x16xf32 warps=1 layout="mma">)
#             move(%store_11: smem_tensor<16x16xbf16 strides=[64, 1]>, %c_11_bf: wrmem_tensor<16x16xbf16 warps=1 layout="mma">)
#             %c_block: gmem_tensor<64x64xbf16 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xbf16>, %by: index, %bx: index)
#             move(%c_block: gmem_tensor<64x64xbf16 strides=[4096, 1]>, %smem_c: smem_tensor<64x64xbf16>) attrs={tma: true}
#           }
#         }""")  # noqa: E501
#     return gpu_dialect.parse(module_str)


# def _register_tiled_mma_matmul_4096() -> None:
#     module = tiled_mma_matmul_4096_module()
#     kernel = make_ptx_kernel(module)
#     launcher = make_ptx_launcher(kernel)
#     make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])



# Based on https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_3.cuh
# BM=128, BN=128, BK=64, 128 threads.
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
def pranjalssh_4096_pipelined_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @pranjalssh_4096_pipelined(%a: gmem_tensor<4096x4096xbf16>, %bt: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[32, 32, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %smem_a0: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %smem_b0: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %smem_a1: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %smem_b1: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
            %a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<4096x4096xbf16>, %by: index, 0)
            %bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]> = tile(%bt: gmem_tensor<4096x4096xbf16>, %bx: index, 0)
            %a_tile_pro: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, 0)
            %bt_tile_pro: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, 0)
            %lock_a0_pro: lock<mbar> = alloc()
            async(%lock_a0_pro: lock<mbar>) move(%smem_a0: smem_tensor<128x64xbf16 swizzle="128B">, %a_tile_pro: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            %lock_b0_pro: lock<mbar> = alloc()
            async(%lock_b0_pro: lock<mbar>) move(%smem_b0: smem_tensor<128x64xbf16 swizzle="128B">, %bt_tile_pro: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            await(%lock_a0_pro: lock<mbar>)
            await(%lock_b0_pro: lock<mbar>)
            for (%k: index = 0 to 62 step 2) {
              %c1: index = index_constant() attrs={value: 1}
              %c2: index = index_constant() attrs={value: 2}
              %k1: index = index_add(%k: index, %c1: index)
              %k2: index = index_add(%k: index, %c2: index)
              %a_tile_1: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %k1: index)
              %bt_tile_1: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %k1: index)
              %lock_a1: lock<mbar> = alloc()
              async(%lock_a1: lock<mbar>) move(%smem_a1: smem_tensor<128x64xbf16 swizzle="128B">, %a_tile_1: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              %lock_b1: lock<mbar> = alloc()
              async(%lock_b1: lock<mbar>) move(%smem_b1: smem_tensor<128x64xbf16 swizzle="128B">, %bt_tile_1: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              for (%ki0: index = 0 to 4) {
                %a_inner_0: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a0: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki0: index)
                %b_inner_0: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b0: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki0: index)
                matmul(%a_inner_0: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_0: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              }
              %a_tile_next: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %k2: index)
              %bt_tile_next: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %k2: index)
              %lock_a0_next: lock<mbar> = alloc()
              async(%lock_a0_next: lock<mbar>) move(%smem_a0: smem_tensor<128x64xbf16 swizzle="128B">, %a_tile_next: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              %lock_b0_next: lock<mbar> = alloc()
              async(%lock_b0_next: lock<mbar>) move(%smem_b0: smem_tensor<128x64xbf16 swizzle="128B">, %bt_tile_next: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
              await(%lock_a1: lock<mbar>)
              await(%lock_b1: lock<mbar>)
              for (%ki1: index = 0 to 4) {
                %a_inner_1: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a1: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki1: index)
                %b_inner_1: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b1: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki1: index)
                matmul(%a_inner_1: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_1: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
              }
              await(%lock_a0_next: lock<mbar>)
              await(%lock_b0_next: lock<mbar>)
            }
            %c63: index = index_constant() attrs={value: 63}
            %a_tile_e1: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %c63: index)
            %bt_tile_e1: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%bt_block: gmem_tensor<128x4096xbf16 strides=[4096, 1]>, 0, %c63: index)
            %lock_a1_e: lock<mbar> = alloc()
            async(%lock_a1_e: lock<mbar>) move(%smem_a1: smem_tensor<128x64xbf16 swizzle="128B">, %a_tile_e1: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            %lock_b1_e: lock<mbar> = alloc()
            async(%lock_b1_e: lock<mbar>) move(%smem_b1: smem_tensor<128x64xbf16 swizzle="128B">, %bt_tile_e1: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: true}
            for (%ki0e: index = 0 to 4) {
              %a_inner_0e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a0: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki0e: index)
              %b_inner_0e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b0: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki0e: index)
              matmul(%a_inner_0e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_0e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            }
            await(%lock_a1_e: lock<mbar>)
            await(%lock_b1_e: lock<mbar>)
            for (%ki1e: index = 0 to 4) {
              %a_inner_1e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_a1: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki1e: index)
              %b_inner_1e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%smem_b1: smem_tensor<128x64xbf16 swizzle="128B">, 0, %ki1e: index)
              matmul(%a_inner_1e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_inner_1e: smem_tensor<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            }
            %acc_bf: wrmem_tensor<128x128xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
            convert(%acc_bf: wrmem_tensor<128x128xbf16 warps=4 layout="wgmma_layout_0">, %acc: wrmem_tensor<128x128xf32 warps=4 layout="wgmma_layout_0">)
            %c_block: gmem_tensor<128x128xbf16 strides=[4096, 1]> = tile(%c: gmem_tensor<4096x4096xbf16>, %by: index, %bx: index)
            move(%c_block: gmem_tensor<128x128xbf16 strides=[4096, 1]>, %acc_bf: wrmem_tensor<128x128xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }""")  # noqa: E501
    return gpu_dialect.parse(module_str)


def _register_pranjalssh_4096_pipelined() -> None:
    module = pranjalssh_4096_pipelined_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])







def auto_matmul_4096_module():
    """4096x4096 bf16 matmul built from high-level tiling passes (wgmma m64n128k16).

    Starts from a single matmul op and applies:
      1. Block-level tiling (M=128, N=128)
      2. Promote dst to wrmem (wgmma accumulator)
      3. K-tiling (K=64) sequential loop
      4. Promote lhs/rhs to smem (TMA loads)
      5. Collective thread mapping (block_dim=128)
      6. Inner K-tiling (K=16) sequential loop
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) {
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    module = tile_matmul(
        module,
        tile_dims=["M", "N"], tile_sizes=[128, 128],
        tile_levels=["block.x", "block.y"], tile_labels=[None, None],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["M"], tile_sizes=[1],
        tile_levels=["thread.x"], tile_labels=[None],
        label_filter="mm", collective=True,
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["l"],
        label_filter="mm",
    )
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = licm(module)
    module = pipeline_loop(module, label_filter_loop="k", label_filter_op="mm_(lhs|rhs)_smem_iter0", pipelining_offset=1)
    module = unroll_loop(module, unroll_factor=4, label_filter=r"l_iter\d+")
    for i in range(4):
      module = licm(module)
      module = canonicalize(module)
    module = lower_to_async(module)
    module = fuse_async(module)
    for i in range(4):
      module = licm(module)
      module = canonicalize(module)
    module = schedule_async(module)
    module = canonicalize(module)
    return module




def _register_auto_matmul_4096() -> None:
    module = auto_matmul_4096_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])




def auto_matmul_4096_wgmma256_sequential_module():
    # Persistent-kernel style: 128 blocks (just under 132 SMs on H100).
    # Each block handles 512x256 of output, sequentially iterating over
    # 4 M-tiles of 128 rows each.
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_sequential(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[8, 16, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    module = tile_matmul(
        module,
        tile_dims=["M", "N", "M", "M"], tile_sizes=[512, 256, 128, 64],
        tile_levels=["block.x", "block.y", "for", "for"], tile_labels=[None, None, "m", "mm"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["kk"],
        label_filter="mm",
    )
    module = unroll_loop(module, unroll_factor=4, label_filter=r"kk")
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    for i in range(4):
      module = licm(module)
      module = canonicalize(module)

    module = lower_to_async(module)
    module = fuse_async(module)
    for i in range(4):
      module = licm(module)
      module = canonicalize(module)


    module = pipeline_loop(module, label_filter_loop="k", label_filter_op="mm_(lhs|rhs)_smem_iter0(_await)?", pipelining_offset=1)
    module = pipeline_loop(module, label_filter_loop="mm", label_filter_op="mm_(lhs|rhs)_smem_iter0(_await)?", pipelining_offset=1)
    module = pipeline_loop(module, label_filter_loop="m", label_filter_op="mm_(lhs|rhs)_smem_iter0(_await)?", pipelining_offset=1)

    for i in range(4):
      module = licm(module)
      module = canonicalize(module)

    module = schedule_async(module)
    module = schedule_async(module)
    module = schedule_async(module)

    for i in range(4):
      module = licm(module)
      module = canonicalize(module)

    module = schedule_async(module)
    module = schedule_async(module)
    module = schedule_async(module)

    return module


def _register_auto_matmul_4096_wgmma256_sequential() -> None:
    module = auto_matmul_4096_wgmma256_sequential_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


def auto_matmul_4096_wgmma256_wg_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    module = tile_matmul(
        module,
        tile_dims=["M", "N"], tile_sizes=[128, 256],
        tile_levels=["block.x", "block.y"], tile_labels=[None, None],
        label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["M"], tile_sizes=[64],
        tile_levels=["wg.x"], tile_labels=[None],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["kk"],
        label_filter="mm",
    )
    module = unroll_loop(module, unroll_factor=4, label_filter="k")
    module = lower_to_async(module)
    # Fuse each stage's lhs+rhs TMA loads onto one mbarrier BEFORE pipelining,
    # while load and await are still adjacent (post-pipeline the await moves to
    # the loop top and can't be remapped to the fused lock).
    module = fuse_async(module)
    for i in range(4):
      module = licm(module)
      module = canonicalize(module)

    module = pipeline_loop(module, label_filter_loop="k", label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async", pipelining_offset=1)

    module = unroll_loop(module, unroll_factor=4, label_filter=r"kk_iter\d")
    module = fuse_async(module)


    for i in range(4):
      module = licm(module)
      module = canonicalize(module)



    module = schedule_async(module)
    module = canonicalize(module)

    print(module)

    return module


def _register_auto_matmul_4096_wgmma256_wg() -> None:
    module = auto_matmul_4096_wgmma256_wg_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


def fastcu_5_1_module():
    """4096x4096 bf16 matmul from pranjalssh/fast.cu ``matmul_5_1``.

    The compiled nvcc PTX for ``matmulKernel5_1<128,256,64,384,3>`` (the
    static-shared-memory variant of ``matmul_5``: wgmma ``m64n256k16``,
    BN=256, BK=64, 3-stage TMA producer/consumer pipeline, 384 threads)
    embedded verbatim as a single ``ptx_template`` op. See
    ``gen_fastcu_matmul_5_1_template.py`` for the (re-runnable) transform
    that produced ``_fastcu_matmul_5_1_ptx.py``.

    The framework generates the ``@fastcu_5_1(%a,%b,%c)`` entry plus
    the two TMA descriptors. The descriptors are deliberately cross-wired
    (``desc_a`` <- ``%b``, ``desc_b`` <- ``%a``) so that matmul_5's
    column-major C store lands as ayla's row-major ``a @ b^T`` convention.

    Shared memory is *static* (144 KB), which the driver auto-configures on
    sm_90 — so no dynamic-shared-memory launch opt-in is needed.
    """
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @fastcu_5_1(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[512, 1, 1] block_dim=[384, 1, 1] {
          }
        }"""))  # noqa: E501
    kernel = module.body.entry_block.ops[0]
    entry = kernel.body.entry_block
    a, b, c = entry.arguments
    gmem_t = a.type
    # desc_a feeds matmul_5's A path (BM=128 tile) but is built from %b
    # (arg_idx=1); desc_b feeds the B path (BN=256 tile) built from %a
    # (arg_idx=0). The swap cancels matmul_5's column-major C store so the
    # output is row-major a @ b^T (ayla's convention).
    desc_a = Value(
        name="desc_a",
        type=TmaDescType.build(
            arg_idx=1, gmem_type=gmem_t, tile_shape=[128, 64], swizzle="128B",
        ),
    )
    desc_b = Value(
        name="desc_b",
        type=TmaDescType.build(
            arg_idx=0, gmem_type=gmem_t, tile_shape=[256, 64], swizzle="128B",
        ),
    )
    entry.append_argument(desc_a)
    entry.append_argument(desc_b)
    entry.append(PTXTemplateOp.build(
        operands=[c, desc_a, desc_b],
        result_types=[],
        template=FASTCU_MATMUL_5_1_TEMPLATE,
        operand_names=["c", "desc_a", "desc_b"],
        result_names=[],
        description="fastcu_5_1",
    ))
    return module


def _register_fastcu_5_1() -> None:
    module = fastcu_5_1_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


def register_all() -> None:
    _register_naive_matmul_4096_loop()
    _register_loop_4x4_matmul_4096()
    _register_loop_8x8_matmul_4096()
    _register_tma_matmul_4096()
    # _register_tiled_mma_matmul_4096()
    _register_pranjalssh_4096()
    _register_pranjalssh_4096_pipelined()
    _register_auto_matmul_4096()
    _register_auto_matmul_4096_wgmma256_wg()
    _register_fastcu_5_1()


register_all()
