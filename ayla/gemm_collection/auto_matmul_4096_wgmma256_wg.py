"""The auto_matmul_4096_wgmma256_wg GEMM kernel: builder + torch registration."""
import textwrap

from ..base.pipeline import Pipeline
from ..gpu_dialect import CollectiveAxis, CollectiveLevel, gpu_dialect
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
from ..passes.verify_gpu_dialect import VerifyGpuDialect
from ..torch_custom_call import OperandKind
from .registry import register, register_gemm


@register(
    OperandKind.R, OperandKind.R, OperandKind.RW,
    op_name="auto_matmul_4096_wgmma256_wg_out",
)
@register_gemm
def auto_matmul_4096_wgmma256_wg_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, 1] {
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) collective=xyz.grid attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse_module(module_str)
    pipeline = Pipeline()
    pipeline.add_instrumentation_pass(VerifyGpuDialect())
    pipeline.add_pass(TileMatmul(
        op_dims=["M", "N"], tile_sizes=[128, 256],
        iteration_bindings=[
            (CollectiveLevel.BLOCK, CollectiveAxis.X),
            (CollectiveLevel.BLOCK, CollectiveAxis.Y),
        ],
        for_loop_labels=[None, None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["M"], tile_sizes=[64],
        iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)],
        for_loop_labels=[None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[64],
        iteration_bindings=[None], for_loop_labels=["k"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=[None], for_loop_labels=["kk"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(MultiBuffer(depth=4, label_filter="mm_lhs_smem"))
    pipeline.add_pass(MultiBuffer(depth=4, label_filter="mm_rhs_smem"))
    pipeline.add_pass(LICM())
    pipeline.add_pass(UnrollLoop(unroll_factor=4, label_filter="k"))
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LowerToAsync())
    # Fuse each stage's lhs+rhs TMA loads onto one mbarrier BEFORE pipelining,
    # while load and await are still adjacent (post-pipeline the await moves to
    # the loop top and can't be remapped to the fused lock).
    pipeline.add_pass(FuseAsync())
    pipeline.add_pass(LICM())
    pipeline.add_pass(Canonicalize())

    pipeline.add_pass(PipelineLoop(
        label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    ))
    pipeline.add_pass(UnrollLoop(
        unroll_factor=4,
        label_filter=r"kk_iter\d",
    ))
    pipeline.add_pass(FuseAsync())

    for _ in range(2):
        pipeline.add_pass(LICM())
        pipeline.add_pass(Canonicalize())

    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    return pipeline.run(module)
