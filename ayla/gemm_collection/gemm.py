"""Configurable GEMM module builders used by collection entries."""
import textwrap

from ..base.ir import find_ops_of_type
from ..base.pipeline import Pipeline
from ..gpu_dialect import (
    CollectiveAxis,
    CollectiveLevel,
    KernelOp,
    MatmulOp,
    ModuleOp,
    gpu_dialect,
)
from ..passes.canonicalize import Canonicalize
from ..passes.cse import CSE
from ..passes.fuse_async import FuseAsync
from ..passes.fuse_loops import FuseLoops
from ..passes.licm import LICM
from ..passes.lower_to_async import LowerToAsync
from ..passes.multi_buffer import MultiBuffer
from ..passes.pipeline_loop import PipelineLoop
from ..passes.promote_buffers import PromoteBuffers
from ..passes.schedule_async import ScheduleAsync
from ..passes.stage_register_writeback import StageRegisterWriteback
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop


def schedule_register_matmul(
    *,
    label: str,
    operand_stages: int,
    operand_k_tile: int = 64,
    rhs_tma_load_slices: int = 1,
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
    k_tail_tiles: int = 0,
) -> Pipeline:
    """Schedule a pre-tiled GEMM with an explicit register accumulator.

    The caller owns output tiling, the WRMEM destination, its epilogue, and
    any live-out moves. This pipeline only schedules the reduction and its
    operand movement. Compatible selected reductions are fused automatically.
    """
    if rhs_tma_load_slices < 1:
        raise ValueError("rhs_tma_load_slices must be positive")
    k_label = f"{loop_label_prefix}_k" if loop_label_prefix else "k"
    kk_label = f"{loop_label_prefix}_kk" if loop_label_prefix else "kk"

    pipeline = Pipeline()
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[operand_k_tile],
        iteration_bindings=[None], for_loop_labels=[k_label],
        op_label_filter=label,
        epilogue_label_filter=epilogue_label_filter,
    ))
    pipeline.add_pass(FuseLoops(
        loop_label_filter=k_label,
        body_label_filter=label,
        tail_iterations=k_tail_tiles,
    ))
    pipeline.add_pass(CSE())
    pipeline.add_pass(PromoteBuffers(
        target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None, tma=True,
        physical_layout=(
            {"block_width": 64, "order": "block_major"}
            if operand_k_tile > 64 else None
        ),
        tma_load_slices=(
            {"rhs": rhs_tma_load_slices}
            if rhs_tma_load_slices > 1 else None
        ),
        move_attrs=operand_move_attrs,
        unsafe_reuse_smem=unsafe_reuse_smem,
        label_filter=label,
    ))
    pipeline.add_pass(ScheduleAsync(
        label_filter=rf"(?:{label})_(?:lhs|rhs)_smem",
    ))
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=[None], for_loop_labels=[kk_label],
        op_label_filter=label,
        epilogue_label_filter=epilogue_label_filter,
    ))
    pipeline.add_pass(MultiBuffer(
        depth=operand_stages,
        label_filter=rf"(?:{label})_lhs_smem",
    ))
    pipeline.add_pass(MultiBuffer(
        depth=operand_stages,
        label_filter=rf"(?:{label})_rhs_smem",
    ))
    pipeline.add_pass(LICM())
    pipeline.add_pass(UnrollLoop(
        unroll_factor=operand_stages, label_filter=k_label,
    ))
    if k_tail_tiles:
        pipeline.add_pass(UnrollLoop(
            unroll_factor=max(2, k_tail_tiles),
            label_filter=rf"{k_label}_(?:first|second)_tail",
        ))
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LowerToAsync())
    pipeline.add_pass(FuseAsync())
    pipeline.add_pass(LICM())
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(PipelineLoop(
        label_filter_loop=k_label,
        label_filter_op=rf"(?:{label})_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    ))
    pipeline.add_pass(UnrollLoop(
        unroll_factor=operand_k_tile // 16,
        label_filter=rf"{kk_label}(?:_tail\d+)?(?:_iter\d+)?",
    ))
    pipeline.add_pass(FuseAsync())
    for _ in range(2):
        pipeline.add_pass(LICM())
        pipeline.add_pass(Canonicalize())
    return pipeline


def schedule_tiled_matmul(
    *,
    label: str,
    output_m_tile: int,
    output_n_tile: int,
    operand_stages: int,
    operand_k_tile: int = 64,
    warpgroup_n_tile: int | None = None,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_tma_inner_size: int | None = 128,
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
) -> Pipeline:
    """Build the post-block-tiling GEMM pipeline."""
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
    wg_dim, wg_size = (
        ("N", warpgroup_n_tile)
        if output_m_tile == 64 or output_n_tile > 256
        else ("M", 64)
    )

    pipeline = Pipeline()
    pipeline.add_pass(TileMatmul(
        op_dims=[wg_dim], tile_sizes=[wg_size],
        iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)],
        for_loop_labels=[None],
        op_label_filter=label,
        epilogue_label_filter=epilogue_label_filter,
        allow_partial_tiles=(wg_dim == "M" and output_m_tile % wg_size != 0),
    ))
    pipeline.add_pass(PromoteBuffers(
        target_type="wrmem", operands=["dst"], swizzle=None,
        layout="wgmma_layout_0", warps=4, tma=True,
        move_attrs=operand_move_attrs,
        target_dtype="f32", label_filter=label,
        epilogue_label_filter=epilogue_label_filter,
    ))
    pipeline.add_pass(StageRegisterWriteback(
        writeback_slices=output_slices,
        writeback_buffers=output_store_buffers,
        physical_layout=(
            {
                "swizzle": "128B",
                "block_width": 64,
                "order": "block_major",
            }
            if output_tma_inner_size is not None else None
        ),
        unsafe_reuse_smem=unsafe_reuse_smem,
        label_filter=f"{label}_dst_wrmem",
    ))
    pipeline.add_pass(schedule_register_matmul(
        label=label,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        # Preserve the existing three 128-row RHS loads for the tuned N=384
        # schedule; other supported schedules use one RHS load.
        rhs_tma_load_slices=(
            output_n_tile // warpgroup_n_tile
            if output_n_tile == 384 else 1
        ),
        unsafe_reuse_smem=unsafe_reuse_smem,
        operand_move_attrs=operand_move_attrs,
        epilogue_label_filter=epilogue_label_filter,
        loop_label_prefix=loop_label_prefix,
    ))
    return pipeline


def schedule_matmul(
    module: ModuleOp,
    *,
    output_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
) -> ModuleOp:
    """Apply the configurable WGMMA schedule to one concrete matmul.

    Keeping scheduling separate from module construction lets the JIT first
    specialize a dynamic template, then reuse the static collection's passes.
    """
    [kernel] = module.body.ops
    if not isinstance(kernel, KernelOp):
        raise ValueError("schedule_matmul requires exactly one kernel")
    matmuls = find_ops_of_type(module, MatmulOp)
    if len(matmuls) != 1:
        raise ValueError("schedule_matmul requires exactly one matmul")
    matmul = matmuls[0]
    m, k = matmul.lhs.type.sizes
    n, rhs_k = matmul.rhs.type.sizes
    if None in (m, n, k, rhs_k):
        raise ValueError("schedule_matmul requires static dimensions")
    if rhs_k != k:
        raise ValueError("matmul reduction dimensions must match")
    m, n, k = int(m), int(n), int(k)

    if output_tiles_per_cta not in (1, 2, 4):
        raise ValueError("output_tiles_per_cta must be 1, 2, or 4")
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if output_m_tile not in (128, 192):
        raise ValueError("output_m_tile must be 128 or 192")
    if output_m_tile == 192 and output_tiles_per_cta != 1:
        raise ValueError("M=192 currently requires one output tile per CTA")
    if output_n_tile not in (64, 192, 208, 256, 384):
        raise ValueError(
            "output_n_tile must be 64, 192, 208, 256, or 384"
        )
    if operand_k_tile not in (64, 128):
        raise ValueError("operand_k_tile must be 64 or 128")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix"):
        raise ValueError("output_store must be 'scalar' or 'stmatrix'")
    if output_store == "scalar":
        output_tma_inner_size = None
    m_blocks = (m + output_m_tile - 1) // output_m_tile
    grid_x = (m_blocks + output_tiles_per_cta - 1) // output_tiles_per_cta
    pipeline = Pipeline()
    if output_tiles_per_cta == 1:
        pipeline.add_pass(TileMatmul(
            op_dims=["M", "N"],
            tile_sizes=[output_m_tile, output_n_tile],
            iteration_bindings=[
                (CollectiveLevel.BLOCK, CollectiveAxis.X),
                (CollectiveLevel.BLOCK, CollectiveAxis.Y),
            ],
            for_loop_labels=[None, None],
            op_label_filter="mm",
            allow_partial_tiles=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        ))
    else:
        pipeline.add_pass(TileMatmul(
            op_dims=["M", "M", "N"],
            tile_sizes=[grid_x * output_m_tile, output_m_tile, 256],
            iteration_bindings=[
                None,
                (CollectiveLevel.BLOCK, CollectiveAxis.X),
                (CollectiveLevel.BLOCK, CollectiveAxis.Y),
            ],
            for_loop_labels=["outm", None, None], op_label_filter="mm",
            allow_partial_tiles=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        ))
    pipeline.add_pass(schedule_tiled_matmul(
        label="mm",
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        warpgroup_n_tile=warpgroup_n_tile,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_tma_inner_size=output_tma_inner_size,
        operand_move_attrs=operand_move_attrs,
    ))
    if output_tiles_per_cta != 1:
        pipeline.add_pass(UnrollLoop(
            unroll_factor=output_tiles_per_cta,
            label_filter="outm",
        ))
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    return pipeline.run(module)


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    ptx_optimization_level: int | None = None,
) -> ModuleOp:
    """Build and schedule a static BF16 GEMM module."""
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    problem_name = (
        str(problem_size)
        if m == n == k == problem_size
        else f"m{m}_n{n}_k{k}"
    )
    resolved_warpgroup_n_tile = warpgroup_n_tile
    if resolved_warpgroup_n_tile is None:
        resolved_warpgroup_n_tile = (
            output_n_tile if output_n_tile <= 256 else 128
        )
    name = f"auto_matmul_{problem_name}_wgmma256_wg_tma_store_k{operand_stages}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if resolved_warpgroup_n_tile not in (128, 256):
        name += f"_wg{resolved_warpgroup_n_tile}"
    if output_tiles_per_cta != 1:
        name += f"_{output_tiles_per_cta}tiles"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"

    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            matmul(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return schedule_matmul(
        module,
        output_tiles_per_cta=output_tiles_per_cta,
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        warpgroup_n_tile=resolved_warpgroup_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_tma_inner_size=output_tma_inner_size,
        output_store=output_store,
        operand_move_attrs=operand_move_attrs,
    )
