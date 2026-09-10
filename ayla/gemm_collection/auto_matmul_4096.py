"""The auto_matmul_4096 GEMM kernel: builder + torch registration."""
import textwrap

from ..base.pipeline import Pipeline
from ..gpu_dialect import gpu_dialect
from ..passes.canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.licm import LICM
from ..passes.lower_to_async import LowerToAsync
from ..passes.multi_buffer import MultiBuffer
from ..passes.pipeline_loop import PipelineLoop
from ..passes.promote_buffers import PromoteBuffers
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop
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
    pipeline = Pipeline()
    pipeline.add_pass(TileMatmul(
        op_dims=["M", "N"], tile_sizes=[128, 128],
        iteration_bindings=["block.x", "block.y"],
        for_loop_labels=[None, None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[64],
        iteration_bindings=["for"], for_loop_labels=["k"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=["for"], for_loop_labels=["l"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(MultiBuffer(depth=2, label_filter="mm_lhs_smem"))
    pipeline.add_pass(MultiBuffer(depth=2, label_filter="mm_rhs_smem"))
    pipeline.add_pass(LICM())
    pipeline.add_pass(UnrollLoop(unroll_factor=2, label_filter="k"))
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LICM())
    pipeline.add_pass(PipelineLoop(
        label_filter_loop="k",
        label_filter_op="mm_(lhs|rhs)_smem_iter0",
        pipelining_offset=1,
    ))
    pipeline.add_pass(UnrollLoop(
        unroll_factor=4,
        label_filter=r"l_iter\d+",
    ))
    for _ in range(2):
        pipeline.add_pass(LICM())
        pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LowerToAsync())
    pipeline.add_pass(FuseAsync())
    pipeline.add_pass(LICM())
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    return pipeline.run(module)
