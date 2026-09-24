"""auto_matmul_4096_wgmma256_wg with a fused sigmoid: builder + registration."""
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
from ..passes.tile_elementwise import TileElementwise
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop
from ..passes.verify_gpu_dialect import VerifyGpuDialect
from ..torch_custom_call import OperandKind
from .registry import register


@register(
    OperandKind.R, OperandKind.R, OperandKind.RW,
    op_name="auto_matmul_4096_wgmma256_wg_sigmoid_out",
)
@register(OperandKind.R, OperandKind.R, OperandKind.RESULT)
def auto_matmul_4096_wgmma256_wg_sigmoid_module():
    """auto_matmul_4096_wgmma256_wg with sigmoid applied to the result.

    The sigmoid is written against the whole 4096x4096 output and never says
    where it should run. Tiling cuts it to the same tile as the matmul and
    leaves it after the K loop. The output is promoted first to block-wide
    SMEM and then, after warp-group tiling, to the WGMMA accumulator.
    Element-wise tiling adds a four-iteration, 64-column output loop only after
    those memory spaces are known.

    Matmul tiling and promotion take it along only because
    `epilogue_label_filter` names it. Without that it stays where it was
    written, reading the output back after the store -- correct, and a round
    trip.

    Explicit unrolling materializes all four 64-column output iterations. A
    follow-up can pipeline those operations once the desired overlap is
    expressed through the general loop scheduler.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg_sigmoid(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) {
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) collective=xyz.grid attrs={init: 0.0, label: "mm"}
            elementwise sigmoid(%c: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) attrs={label: "epi"}
          }
        }""")
    module = gpu_dialect.parse_module(module_str)
    pipeline = Pipeline()
    pipeline.add_instrumentation_pass(VerifyGpuDialect())
    pipeline.add_pass(TileMatmul(
        op_dims=["M", "M", "N"], tile_sizes=[1024, 128, 256],
        iteration_bindings=[
            None,
            (CollectiveLevel.BLOCK, CollectiveAxis.X),
            (CollectiveLevel.BLOCK, CollectiveAxis.Y),
        ],
        for_loop_labels=["outm", None, None],
        op_label_filter="mm", epilogue_label_filter="epi",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["dst"], tma=True,
        label_filter="mm", epilogue_label_filter="epi",
        physical_layout={
            "swizzle": "128B",
            "block_width": 64,
            "order": "block_major",
        },
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["M"], tile_sizes=[64],
        iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)],
        for_loop_labels=[None],
        op_label_filter="mm", epilogue_label_filter="epi",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wgrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
        epilogue_label_filter="epi",
    ))
    pipeline.add_pass(TileElementwise(
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
        label_filter="epi|mm_dst_convert|mm_dst_wgrmem|mm_dst_smem",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[64],
        iteration_bindings=[None], for_loop_labels=["k"],
        op_label_filter="mm", epilogue_label_filter="epi",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=[None], for_loop_labels=["kk"],
        op_label_filter="mm", epilogue_label_filter="epi",
    ))
    pipeline.add_pass(MultiBuffer(depth=3, label_filter="mm_lhs_smem"))
    pipeline.add_pass(MultiBuffer(depth=3, label_filter="mm_rhs_smem"))
    pipeline.add_pass(LICM())
    pipeline.add_pass(UnrollLoop(unroll_factor=3, label_filter="k"))
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

    pipeline.add_pass(UnrollLoop(unroll_factor=4, label_filter="outn"))
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(UnrollLoop(unroll_factor=4, label_filter="outm"))
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    return pipeline.run(module)
