"""Configurable GEMM module builders used by collection entries."""
import textwrap
from typing import cast

from ..base.ir import find_ops_of_type
from ..base.pipeline import Pipeline
from ..gpu_dialect import (
    BarrierOp,
    CollectiveAxis,
    CollectiveLevel,
    ForOp,
    KernelOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    gpu_dialect,
)
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
) -> Pipeline:
    """Schedule a pre-tiled GEMM with an explicit register accumulator.

    The caller owns output tiling, the WGRMEM destination, its epilogue, and
    any live-out moves. This pipeline only schedules the reduction and its
    operand movement.
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
    pipeline.add_pass(TileMatmul(
        op_dims=["K"], tile_sizes=[16],
        iteration_bindings=[None], for_loop_labels=[kk_label],
        op_label_filter=label,
        epilogue_label_filter=epilogue_label_filter,
    ))
    pipeline.add_pass(MultiBuffer(
        depth=operand_stages, label_filter=f"{label}_lhs_smem",
    ))
    pipeline.add_pass(MultiBuffer(
        depth=operand_stages, label_filter=f"{label}_rhs_smem",
    ))
    pipeline.add_pass(LICM())
    pipeline.add_pass(UnrollLoop(
        unroll_factor=operand_stages, label_filter=k_label,
    ))
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LowerToAsync())
    pipeline.add_pass(FuseAsync())
    pipeline.add_pass(LICM())
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(PipelineLoop(
        label_filter_loop=k_label,
        label_filter_op=rf"{label}_(lhs|rhs)_smem_iter\d+_async",
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
        target_type="wgrmem", operands=["dst"], swizzle=None,
        layout="wgmma_layout_0", warps=4, tma=True,
        physical_layout=(
            {
                "swizzle": "128B",
                "block_width": 64,
                "order": "block_major",
            }
            if output_tma_inner_size is not None else None
        ),
        writeback_slices=output_slices,
        writeback_buffers=output_store_buffers,
        move_attrs=operand_move_attrs,
        target_dtype="f32", label_filter=label,
        epilogue_label_filter=epilogue_label_filter,
        unsafe_reuse_smem=unsafe_reuse_smem,
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
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_grid_size: int | None = None,
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
    assert isinstance(matmul, MatmulOp)
    assert isinstance(matmul.lhs.type, MemTensorType)
    assert isinstance(matmul.rhs.type, MemTensorType)
    m, k = matmul.lhs.type.sizes
    n, rhs_k = matmul.rhs.type.sizes
    if None in (m, n, k, rhs_k):
        raise ValueError("schedule_matmul requires static dimensions")
    if rhs_k != k:
        raise ValueError("matmul reduction dimensions must match")
    m, n, k = cast(tuple[int, int, int], (m, n, k))

    output_groups = (output_m_tiles_per_cta, output_n_tiles_per_cta)
    if any(group not in (1, 2, 3, 4, 6, 8, 16) for group in output_groups):
        raise ValueError(
            "output tile groups must each be 1, 2, 3, 4, 6, 8, or 16"
        )
    if output_m_tiles_per_cta > 1 and output_n_tiles_per_cta > 1:
        raise ValueError("output tiles may be grouped along only one axis")
    if output_grid_size is not None:
        if output_grid_size < 1:
            raise ValueError("output_grid_size must be positive")
        if output_groups != (1, 1):
            raise ValueError(
                "a fixed output grid and static output grouping are "
                "mutually exclusive"
            )
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if output_m_tile not in (64, 128, 192, 256):
        raise ValueError("output_m_tile must be 64, 128, 192, or 256")
    if not 48 <= output_n_tile <= 384 or output_n_tile % 8:
        raise ValueError(
            "output_n_tile must be a multiple of 8 between 48 and 384"
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
    n_blocks = (n + output_n_tile - 1) // output_n_tile
    if output_n_tiles_per_cta > 1:
        group_dim = "N"
        blocks = n_blocks
        tile_size = output_n_tile
        group_size = output_n_tiles_per_cta
        group_label = "outn"
    else:
        group_dim = "M"
        blocks = m_blocks
        tile_size = output_m_tile
        group_size = output_m_tiles_per_cta
        group_label = "outm"
    pipeline = Pipeline()
    if group_size == 1:
        if output_grid_size is not None:
            kernel.grid_dim = (output_grid_size, 1, 1)
        pipeline.add_pass(TileMatmul(
            op_dims=["M", "N"],
            tile_sizes=[output_m_tile, output_n_tile],
            iteration_bindings=[
                (CollectiveLevel.BLOCK, CollectiveAxis.X),
                (
                    CollectiveLevel.BLOCK,
                    CollectiveAxis.X if output_grid_size is not None
                    else CollectiveAxis.Y,
                ),
            ],
            for_loop_labels=[None, None],
            op_label_filter="mm",
            grid_stride_loop_label=(
                "outmn" if output_grid_size is not None else None
            ),
            allow_partial_tiles=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        ))
    else:
        grid = (blocks + group_size - 1) // group_size
        pipeline.add_pass(TileMatmul(
            op_dims=[group_dim, "M", "N"],
            tile_sizes=[
                grid * tile_size, output_m_tile, output_n_tile,
            ],
            iteration_bindings=[
                None,
                (CollectiveLevel.BLOCK, CollectiveAxis.X),
                (CollectiveLevel.BLOCK, CollectiveAxis.Y),
            ],
            for_loop_labels=[group_label, None, None], op_label_filter="mm",
            allow_partial_tiles=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
                or blocks % group_size != 0
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
    if output_grid_size is not None:
        # Promotion allocations are guarded by the partial final wave. Hoist
        # them through both that guard and the grid-stride loop before async
        # scheduling, then separate successive tiles reusing the same arenas.
        pipeline.add_pass(LICM())
        pipeline.add_pass(LICM())
        module = pipeline.run(module)
        loops = [
            loop for loop in find_ops_of_type(module, ForOp)
            if loop.attrs.get("label") == "outmn"
        ]
        logical_tiles = m_blocks * n_blocks
        if len(loops) > 1 or (
            not loops and logical_tiles > output_grid_size
        ):
            raise ValueError("expected one fixed output-grid wave loop")
        if loops:
            loops[0].body.ops.append(BarrierOp.build("block"))
        pipeline = Pipeline()
    if group_size > 1:
        pipeline.add_pass(UnrollLoop(
            unroll_factor=group_size,
            label_filter=group_label,
        ))
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    return pipeline.run(module)


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_grid_size: int | None = None,
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
    dtype: str = "bf16",
) -> ModuleOp:
    """Build and schedule a static BF16 or FP16 GEMM module."""
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if dtype not in ("bf16", "f16"):
        raise ValueError("dtype must be 'bf16' or 'f16'")
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
    if dtype != "bf16":
        name += f"_{dtype}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if resolved_warpgroup_n_tile not in (128, 256):
        name += f"_wg{resolved_warpgroup_n_tile}"
    if output_m_tiles_per_cta != 1:
        name += f"_{output_m_tiles_per_cta}tiles"
    if output_n_tiles_per_cta != 1:
        name += f"_n{output_n_tiles_per_cta}tiles"
    if output_grid_size is not None:
        name += f"_grid{output_grid_size}"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"

    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>) attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            matmul(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return schedule_matmul(
        module,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        output_n_tiles_per_cta=output_n_tiles_per_cta,
        output_grid_size=output_grid_size,
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
