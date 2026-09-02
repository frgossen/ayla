"""The auto_matmul_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from ..passes.canonicalize import Canonicalize
from ..passes.fuse_async import fuse_async
from ..passes.licm import licm
from ..passes.lower_to_async import lower_to_async
from ..passes.multi_buffer import multi_buffer
from ..passes.pipeline_loop import pipeline_loop
from ..passes.promote_buffers import promote_buffers
from ..passes.schedule_async import schedule_async
from ..passes.tile_matmul import tile_matmul
from ..passes.unroll_loop import unroll_loop
from .registry import register_gemm


@register_gemm
def auto_matmul_4096_module():
    """4096x4096 bf16 matmul built from high-level tiling passes (wgmma m64n128k16).

    The kernel declares block_dim=128: the wgmma is issued by a whole
    warpgroup, so the threads are not something the tiling hands out.

    Starts from a single matmul op and applies:
      1. Block-level tiling (M=128, N=128)
      2. Promote dst to wrmem (wgmma accumulator)
      3. K-tiling (K=64) sequential loop
      4. Promote lhs/rhs to smem (TMA loads)
      5. Inner K-tiling (K=16) sequential loop
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) block_dim=[128, 1, 1] {
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
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
        tma=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["l"],
        label_filter="mm",
    )
    module = multi_buffer(module, depth=2, label_filter="mm_lhs_smem")
    module = multi_buffer(module, depth=2, label_filter="mm_rhs_smem")
    module = licm(module)
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = Canonicalize().run(module)
    module = licm(module)
    module = pipeline_loop(module, label_filter_loop="k", label_filter_op="mm_(lhs|rhs)_smem_iter0", pipelining_offset=1)
    module = unroll_loop(module, unroll_factor=4, label_filter=r"l_iter\d+")
    for i in range(4):
      module = licm(module)
      module = Canonicalize().run(module)
    module = lower_to_async(module)
    module = fuse_async(module)
    for i in range(4):
      module = licm(module)
      module = Canonicalize().run(module)
    module = schedule_async(module)
    module = Canonicalize().run(module)
    return module
