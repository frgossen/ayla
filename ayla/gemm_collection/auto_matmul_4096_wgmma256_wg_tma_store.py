"""Registered variants of the configurable pipelined TMA GEMM."""

import textwrap

from ..base.pipeline import Pipeline
from ..gpu_dialect import gpu_dialect, CollectiveAxis, CollectiveLevel
from ..passes.canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.licm import LICM
from ..passes.lower_to_async import LowerToAsync
from ..passes.multi_buffer import MultiBuffer
from ..passes.pack_smem_allocs import PackSmemAllocs
from ..passes.pipeline_loop import PipelineLoop
from ..passes.promote_buffers import PromoteBuffers
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop
from .gemm import auto_matmul_4096_wgmma256_wg_tma_store_module
from .registry import register_gemm


@register_gemm
def auto_matmul_4096_wgmma256_wg_tma_store_k3_module():
    return auto_matmul_4096_wgmma256_wg_tma_store_module()


@register_gemm
def auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_module():
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_m_tiles_per_cta=2,
    )


@register_gemm
def auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2_module():
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_m_tiles_per_cta=2,
        operand_stages=3,
        output_slices=2,
        output_store_buffers=2,
    )


@register_gemm
def auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2_module():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[8, 16, 1] attrs={ptx_optimization_level: 1} {
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) collective=xyz.grid attrs={init: 0.0, label: "mm"}
          }
        }"""))
    move_attrs = {
        "lhs": {
            "l2_cache_hint": "evict_first",
            "l2_promotion": "256B",
        },
        "dst": {"l2_cache_hint": "evict_first"},
    }
    pipeline = Pipeline(dump_ir_on_change=True)
    pipeline.add_pass(TileMatmul(
        op_dims=["M", "M", "N"],
        tile_sizes=[1024, 128, 256],
        iteration_bindings=[None, (CollectiveLevel.BLOCK, CollectiveAxis.X), (CollectiveLevel.BLOCK, CollectiveAxis.Y)],
        for_loop_labels=["outm", None, None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["M"],
        tile_sizes=[64],
        iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)],
        for_loop_labels=[None],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wgrmem",
        operands=["dst"],
        swizzle=None,
        layout="wgmma_layout_0",
        warps=4,
        tma=True,
        physical_layout={
            "swizzle": "128B",
            "block_width": 64,
            "order": "block_major",
        },
        writeback_slices=2,
        writeback_buffers=2,
        move_attrs=move_attrs,
        target_dtype="f32",
        label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"],
        tile_sizes=[64],
        iteration_bindings=[None],
        for_loop_labels=["k"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="smem",
        operands=["lhs", "rhs"],
        swizzle="128B",
        layout=None,
        warps=None,
        tma=True,
        physical_layout=None,
        move_attrs=move_attrs,
        label_filter="mm",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"],
        tile_sizes=[16],
        iteration_bindings=[None],
        for_loop_labels=["kk"],
        op_label_filter="mm",
    ))
    pipeline.add_pass(MultiBuffer(depth=3, label_filter="mm_lhs_smem"))
    pipeline.add_pass(MultiBuffer(depth=3, label_filter="mm_rhs_smem"))
    pipeline.add_pass(LICM())
    pipeline.add_pass(UnrollLoop(unroll_factor=3, label_filter="k"))
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LowerToAsync())
    pipeline.add_pass(FuseAsync())
    for _ in range(4):
        pipeline.add_pass(LICM())
        pipeline.add_pass(Canonicalize())

    pipeline.add_pass(PipelineLoop(
        label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    ))
    pipeline.add_pass(UnrollLoop(
        unroll_factor=4,
        label_filter=r"kk(?:_tail\d+)?(?:_iter\d+)?",
    ))
    pipeline.add_pass(FuseAsync())
    for _ in range(4):
        pipeline.add_pass(LICM())
        pipeline.add_pass(Canonicalize())

    pipeline.add_pass(UnrollLoop(unroll_factor=4, label_filter="outm"))
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(PackSmemAllocs())
    return pipeline.run(module)
