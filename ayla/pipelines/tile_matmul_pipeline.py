"""Configurable GEMM module builders used by collection entries."""

import math
import re

from ..base.ir import Block, find_ops_of_type
from ..base.pipeline import Pass, Pipeline
from ..gpu_dialect import (
    CollectiveAxis,
    CollectiveLevel,
    KernelOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    AllocOp,
    AsyncOp,
    BarrierOp,
    MoveOp,
    SmemTensorType,
    root_buffers,
)
from .canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.fuse_loops import FuseLoops
from ..passes.licm import LICM
from ..passes.lower_to_async import LowerToAsync
from ..passes.multi_buffer import MultiBuffer
from ..passes.peel_loop import PeelLoop
from ..passes.pipeline_loop import PipelineLoop
from ..passes.promote_buffers import PromoteBuffers
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop


from typing import cast


class _OperandStagePeriod(Pass):
    """Use the period of the rings actually promoted in this reduction."""

    def __init__(self, label: str, stages: int, overrides: dict[str, int], unroll: UnrollLoop):
        super().__init__()
        self.label, self.stages, self.overrides, self.unroll = label, stages, overrides, unroll

    def run(self, module: ModuleOp) -> ModuleOp:
        labels = {str(op.attrs.get("label", "")) for op in find_ops_of_type(module, AllocOp)
                  if re.fullmatch(rf"(?:{self.label})_(?:lhs|rhs)_smem", str(op.attrs.get("label", "")))}
        missing = set(self.overrides) - labels
        if missing:
            raise ValueError(f"operand stage overrides have no promoted allocation: {sorted(missing)}")
        self.unroll.unroll_factor = math.lcm(*(self.overrides.get(name, self.stages) for name in labels))
        return module


class _OperandPrefetchGroups(Pass):
    """Fuse copies only when the selected pipeline gives them the same timing."""

    def __init__(self, label: str, stages: int, overrides: dict[str, int], cyclic: bool):
        super().__init__()
        self.label = label
        self.stages, self.overrides, self.cyclic = stages, overrides, cyclic

    def run(self, module: ModuleOp) -> ModuleOp:
        for issue in find_ops_of_type(module, AsyncOp):
            if len(issue.body.ops) != 1 or not isinstance(issue.body.ops[0], MoveOp):
                continue
            label = str(issue.body.ops[0].attrs.get("label", ""))
            if not re.fullmatch(rf"(?:{self.label})_(?:lhs|rhs)_smem(?:_.*)?", label):
                continue
            match = re.fullmatch(r"(.+_(?:lhs|rhs)_smem)_iter(\d+)", label)
            if match is not None:
                depth = self.overrides.get(match[1], self.stages)
                ahead = int(match[2]) < depth - int(self.cyclic)
                issue.attrs["fusion_group"] = "operand_ahead" if ahead else "operand_current"
            elif re.search(r"_(?:lhs|rhs)_smem", label):
                issue.attrs["fusion_group"] = "operand_head"
        return module


def schedule_register_matmul(
    *,
    label: str,
    operand_stages: int,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_k_tile: int = 64,
    mma_k_tile: int = 16,
    cyclic_operand_pipeline: bool = False,
    rhs_tma_load_slices: int = 1,
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
    k_tail_tiles: int | None = None,
) -> Pipeline:
    """Schedule a pre-tiled GEMM with an explicit register accumulator.

    The caller owns output tiling, the WGRMEM destination, its epilogue, and
    any live-out moves. This pipeline only schedules the reduction and its
    operand movement. ``k_tail_tiles=None`` preserves independent loops; zero
    fuses compatible selected loops with retained operands. Positive values
    peel that many outer K tiles for caller-owned asynchronous epilogues.
    ``operand_stage_overrides`` maps promoted allocation labels (for example
    ``up_rhs_smem``) to depths. Other operands keep ``operand_stages``. Copies
    with different prefetch timing retain independent completion events.
    """
    if mma_k_tile not in (16, 32) or operand_k_tile % mma_k_tile:
        raise ValueError("MMA K tile must be 16 or 32 and divide the operand K tile")
    if rhs_tma_load_slices < 1:
        raise ValueError("rhs_tma_load_slices must be positive")
    overrides = dict(operand_stage_overrides or {})
    for name, depth in overrides.items():
        if (not re.fullmatch(rf"(?:{label})_(?:lhs|rhs)_smem", name)
                or type(depth) is not int or depth < 2):
            raise ValueError("operand stage overrides require matching allocation labels and depths >= 2")
    k_label = f"{loop_label_prefix}_k" if loop_label_prefix else "k"
    kk_label = f"{loop_label_prefix}_kk" if loop_label_prefix else "kk"

    pipeline = Pipeline()
    pipeline.add_pass(
        TileMatmul(
            op_dims=["K"],
            tile_sizes=[operand_k_tile],
            iteration_bindings=[None],
            for_loop_labels=[k_label],
            op_label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
        )
    )
    if k_tail_tiles is not None:
        pipeline.add_pass(FuseLoops(k_label, label, tail_iterations=k_tail_tiles))
        if k_tail_tiles:
            pipeline.add_pass(UnrollLoop(k_tail_tiles + 1, label_filter=rf"{k_label}_(?:first|second)_tail"))
        pipeline.add_pass(Canonicalize())
        label = rf"(?:{label})(?:_(?:tail|iter)\d+)?"
    pipeline.add_pass(
        PromoteBuffers(
            target_type="smem",
            operands=["lhs", "rhs"],
            reuse_operands=k_tail_tiles is not None,
            swizzle="auto",
            layout=None,
            warps=None,
            tma=True,
            physical_layout=(
                {"block_width": 4 * mma_k_tile, "order": "block_major"}
                if operand_k_tile > 4 * mma_k_tile
                else None
            ),
            tma_load_slices=(
                {"rhs": rhs_tma_load_slices} if rhs_tma_load_slices > 1 else None
            ),
            move_attrs=operand_move_attrs,
            unsafe_reuse_smem=unsafe_reuse_smem,
            label_filter=label,
        )
    )
    pipeline.add_pass(
        TileMatmul(
            op_dims=["K"],
            tile_sizes=[mma_k_tile],
            iteration_bindings=[None],
            for_loop_labels=[kk_label],
            op_label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
        )
    )
    outer_unroll = UnrollLoop(unroll_factor=operand_stages, label_filter=k_label)
    if overrides:
        pipeline.add_pass(_OperandStagePeriod(label, operand_stages, overrides, outer_unroll))
    excluded = "|".join(re.escape(name) for name in overrides)
    exclusion = rf"(?!(?:{excluded})$)" if overrides else ""
    pipeline.add_pass(
        MultiBuffer(
            depth=operand_stages,
            label_filter=rf"{exclusion}(?:{label})_lhs_smem",
        )
    )
    pipeline.add_pass(
        MultiBuffer(
            depth=operand_stages,
            label_filter=rf"{exclusion}(?:{label})_rhs_smem",
        )
    )
    for name, depth in overrides.items():
        pipeline.add_pass(MultiBuffer(depth=depth, label_filter=re.escape(name)))
    pipeline.add_pass(LICM())
    # Unroll kk and peel the first K step, so the reduction's first WGMMA is a
    # single op the zeroed accumulator folds into as an overwrite.
    pipeline.add_pass(
        UnrollLoop(
            unroll_factor=operand_k_tile // mma_k_tile,
            label_filter=rf"{kk_label}(?:_tail\d+)?(?:_iter\d+)?",
        )
    )
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(PeelLoop(label_filter=k_label, front_iterations=1))
    pipeline.add_pass(outer_unroll)
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(LowerToAsync())
    if overrides:
        pipeline.add_pass(_OperandPrefetchGroups(label, operand_stages, overrides, cyclic_operand_pipeline))
    pipeline.add_pass(FuseAsync())
    pipeline.add_pass(LICM())
    pipeline.add_pass(Canonicalize())
    # A cyclic pipeline leaves the last stage's loads to the queued wgmma ring.
    stages = "|".join(map(str, range(operand_stages - cyclic_operand_pipeline)))
    selected = rf"{label}_(lhs|rhs)_smem_iter(?:{stages})_async"
    if overrides:
        selected = rf"(?!(?:{excluded})_iter\d+_async$)(?:{selected})"
        selected = "|".join([selected, *[
            rf"{re.escape(name)}_iter(?:{'|'.join(map(str, range(depth - cyclic_operand_pipeline)))})_async"
            for name, depth in overrides.items()
        ]])
    pipeline.add_pass(
        PipelineLoop(
            label_filter_loop=k_label,
            label_filter_op=selected,
            pipelining_offset=1,
        )
    )
    pipeline.add_pass(FuseAsync())
    for _ in range(2):
        pipeline.add_pass(LICM())
        pipeline.add_pass(Canonicalize())
    if cyclic_operand_pipeline:
        pipeline.add_pass(
            PipelineLoop(
                label_filter_loop=k_label,
                label_filter_op=rf"{label}(?:_iter\d+)+_async(?:_iter\d+)?",
                pipelining_offset=1,
            )
        )
    return pipeline


def schedule_tiled_matmul(
    *,
    label: str,
    output_m_tile: int,
    output_n_tile: int,
    operand_stages: int,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_k_tile: int = 64,
    mma_k_tile: int = 16,
    warpgroup_m_tile: int | None = None,
    cyclic_operand_pipeline: bool = False,
    warpgroup_n_tile: int | None = None,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_fragment_cols: int | None = None,
    output_fragment_swizzle: str | None = None,
    output_tma_inner_size: int | None = 128,
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
    atomic_add: bool = False,
    output_tma: bool = True,
) -> Pipeline:
    """Build the post-block-tiling GEMM pipeline."""
    if output_fragment_swizzle is not None and output_fragment_cols is None:
        raise ValueError("fragment swizzle requires fragment writeback")
    if output_fragment_cols is not None:
        output_tma_inner_size = None
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
    wg_tiles = (
        {"M": warpgroup_m_tile, "N": warpgroup_n_tile}
        if warpgroup_m_tile is not None
        else {"N": warpgroup_n_tile}
        if output_m_tile == 64 or output_n_tile > 256
        else {"M": 64}
    )

    pipeline = Pipeline()
    pipeline.add_pass(
        TileMatmul(
            op_dims=list(wg_tiles),
            tile_sizes=list(wg_tiles.values()),
            iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)] * len(wg_tiles),
            for_loop_labels=[None] * len(wg_tiles),
            op_label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
            allow_partial_tiles=(
                list(wg_tiles) == ["M"] and output_m_tile % wg_tiles["M"] != 0
            ),
        )
    )
    pipeline.add_pass(
        PromoteBuffers(
            target_type="wgrmem",
            operands=["dst"],
            swizzle=output_fragment_swizzle,
            layout="wgmma_layout_0",
            warps=4,
            tma=output_tma and not atomic_add,
            physical_layout=(
                {
                    "swizzle": "128B",
                    "block_width": 64,
                    "order": "block_major",
                }
                if output_tma_inner_size is not None
                else None
            ),
            writeback_slices=output_slices,
            writeback_buffers=output_store_buffers,
            writeback_fragment_cols=output_fragment_cols,
            move_attrs=(
                {
                    **(operand_move_attrs or {}),
                    "dst": {
                        **(operand_move_attrs or {}).get("dst", {}),
                        "atomic_add": True,
                    },
                }
                if atomic_add
                else operand_move_attrs
            ),
            target_dtype="f32",
            label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
            unsafe_reuse_smem=unsafe_reuse_smem,
        )
    )
    pipeline.add_pass(
        schedule_register_matmul(
            label=label,
            operand_stages=operand_stages,
            operand_stage_overrides=operand_stage_overrides,
            operand_k_tile=operand_k_tile,
            mma_k_tile=mma_k_tile,
            cyclic_operand_pipeline=cyclic_operand_pipeline,
            # TMA box dimensions cannot exceed 256. Wide column tiles load
            # each warpgroup's RHS separately, including the tuned N=384 case.
            rhs_tma_load_slices=(
                output_n_tile // warpgroup_n_tile if output_n_tile > 256 else 1
            ),
            unsafe_reuse_smem=unsafe_reuse_smem,
            operand_move_attrs=operand_move_attrs,
            epilogue_label_filter=epilogue_label_filter,
            loop_label_prefix=loop_label_prefix,
        )
    )
    return pipeline


def _schedule_matmul(
    module: ModuleOp,
    *,
    epilogue_label_filter: str | None = None,
    split_k_parts: int = 1,
    split_k_atomic: bool = False,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_grid_size: int | None = None,
    output_grid_axes: tuple[CollectiveAxis, CollectiveAxis] = (
        CollectiveAxis.X,
        CollectiveAxis.Y,
    ),
    output_grid_fast_axis: str = "n",
    output_reversed_dims: tuple[str, ...] = (),
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_m_tile: int | None = None,
    warpgroup_n_tile: int | None = None,
    cluster_m_ctas: int = 1,
    cluster_n_ctas: int = 1,
    operand_stages: int = 3,
    operand_stage_overrides: dict[str, int] | None = None,
    operand_k_tile: int = 64,
    cyclic_operand_pipeline: bool = False,
    warpgroup_ctas: int = 1,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_fragment_cols: int | None = None,
    output_fragment_swizzle: str | None = None,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    unsafe_reuse_smem: bool = False,
) -> ModuleOp:
    """Apply the configurable WGMMA schedule to one concrete matmul.

    Keeping scheduling separate from module construction lets the JIT first
    specialize a dynamic template, then reuse the static collection's passes.
    ``split_k_atomic=True`` partitions K across grid Z and atomically adds F32
    partials directly into C in one launch. Initialize C before launching.
    ``warpgroup_ctas=2`` pairs two independent 128-thread task programs per
    physical CTA. The even virtual grid must contain complete output waves;
    shared operand handoffs are materialized during PTX lowering. This applies
    to FP16, BF16 and FP8 and requires one warpgroup per output tile.
    """
    if type(warpgroup_ctas) is not int or warpgroup_ctas not in (1, 2):
        raise ValueError("warpgroup_ctas must be one or two")
    if warpgroup_ctas == 2 and (
        split_k_parts != 1
        or split_k_atomic
        or type(output_grid_size) is not int
        or output_grid_size < 2
        or output_grid_size % 2
        or output_m_tiles_per_cta != 1
        or output_n_tiles_per_cta != 1
        or cluster_m_ctas != 1
        or cluster_n_ctas != 1
        or unsafe_reuse_smem
        or operand_stage_overrides
    ):
        raise ValueError(
            "paired warpgroups require an even fixed grid, ungrouped WGMMA tiles "
            "and disjoint writeback"
        )
    if split_k_atomic:
        from ..passes.tile_reduce import TileReduce

        heroes = find_ops_of_type(module, MatmulOp)
        if len(heroes) != 1 or type(split_k_parts) is not int or split_k_parts < 1:
            raise ValueError("atomic partitioning requires one matmul and positive parts")
        k = cast(MemTensorType, heroes[0].lhs.type).static_shape[-1]
        if k % split_k_parts:
            raise ValueError("partition count must divide the reduction dimension")
        module = TileReduce({1: k // split_k_parts},
                            (CollectiveLevel.BLOCK, CollectiveAxis.Z)).split(module, atomic=True)
    elif split_k_parts != 1:
        raise ValueError(
            "split_k_parts requires split_k_atomic; use TileReduce.split for two phases"
        )
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
    if warpgroup_ctas == 2:
        wg_m = warpgroup_m_tile or 64
        wg_n = warpgroup_n_tile or (output_n_tile if output_n_tile <= 256 else 128)
        tasks = math.ceil(m / output_m_tile) * math.ceil(n / output_n_tile)
        if (
            wg_m != output_m_tile
            or wg_n != output_n_tile
            or tasks % cast(int, output_grid_size)
            or any(d not in (None, 1) for d in kernel.grid_dim)
            or any(d not in (None, 1) for d in kernel.cluster_dim)
        ):
            raise ValueError(
                "paired warpgroups require one warpgroup per tile "
                "and complete one-dimensional waves"
            )

    output_groups = (output_m_tiles_per_cta, output_n_tiles_per_cta)
    if any(group not in (1, 2, 3, 4, 5, 6, 8, 16) for group in output_groups):
        raise ValueError("output tile groups must each be 1, 2, 3, 4, 5, 6, 8, or 16")
    if output_m_tiles_per_cta > 1 and output_n_tiles_per_cta > 1:
        raise ValueError("output tiles may be grouped along only one axis")
    if min(cluster_m_ctas, cluster_n_ctas) < 1:
        raise ValueError("cluster dimensions must be positive")
    if cluster_m_ctas > 1 and cluster_n_ctas > 1:
        raise ValueError("only one multicast cluster axis is supported")
    if not set(output_reversed_dims) <= {"M", "N"}:
        raise ValueError("output_reversed_dims must name M and/or N")
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if output_m_tile not in (64, 128, 192, 256):
        raise ValueError("output_m_tile must be 64, 128, 192, or 256")
    if not 8 <= output_n_tile <= 384 or output_n_tile % 8:
        raise ValueError("output_n_tile must be a multiple of 8 between 8 and 384")
    if operand_k_tile not in (64, 128):
        raise ValueError("operand_k_tile must be 64 or 128")
    if cyclic_operand_pipeline and operand_stages < 2:
        raise ValueError("cyclic_operand_pipeline requires at least two operand stages")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix", "global"):
        raise ValueError("output_store must be 'scalar', 'stmatrix', or 'global'")
    if output_fragment_cols is not None and (
        type(output_fragment_cols) is not int or not 8 <= output_fragment_cols <= 256
        or output_fragment_cols % 8
        or output_store == "global" or output_slices != 1
        or not 1 <= output_store_buffers <= 8
    ):
        raise ValueError("fragment output requires aligned column partitions, 1-8 buffers and TMA writeback")
    if unsafe_reuse_smem and (
        warpgroup_m_tile not in (64, 128)
        or output_m_tile % warpgroup_m_tile
        or warpgroup_n_tile != output_n_tile
        or output_store == "global"
        or output_slices != 1 or (output_store_buffers != 1 and output_fragment_cols is None)
        or output_groups != (1, 1)
        or (cluster_m_ctas, cluster_n_ctas) != (1, 1)
        or split_k_atomic
    ):
        raise ValueError("shared compute/writeback reuse requires complete row warpgroups and one ungrouped TMA output tile")
    if output_store == "global" and (
        m % (output_m_tile * output_m_tiles_per_cta)
        or n % (output_n_tile * output_n_tiles_per_cta)
    ):
        raise ValueError("direct global stores require complete output tiles")
    # STMatrix stores 16-bit elements. F32 partials use affine scalar staging.
    if (
        output_store == "scalar"
        or cast(MemTensorType, matmul.dst.type).dtype.name == "f32"
    ):
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
    if output_grid_size is not None:
        kernel.grid_dim = (output_grid_size, 1, kernel.grid_dim[2] or 1)
    block_x = (CollectiveLevel.BLOCK, CollectiveAxis.X)
    output_bindings: list[tuple[CollectiveLevel, CollectiveAxis] | None] = (
        [block_x, block_x]
        if output_grid_size is not None
        else [
            (CollectiveLevel.BLOCK, output_grid_axes[0]),
            (CollectiveLevel.BLOCK, output_grid_axes[1]),
        ]
    )
    m_grid = (m_blocks + output_m_tiles_per_cta - 1) // output_m_tiles_per_cta
    n_grid = (n_blocks + output_n_tiles_per_cta - 1) // output_n_tiles_per_cta
    grid_by_axis = dict(zip(output_grid_axes, (m_grid, n_grid), strict=True))
    cluster_by_axis = dict(
        zip(
            output_grid_axes,
            (cluster_m_ctas, cluster_n_ctas),
            strict=True,
        )
    )
    if any(
        grid_by_axis[axis] % cluster_by_axis[axis]
        for axis in (CollectiveAxis.X, CollectiveAxis.Y)
    ):
        raise ValueError("CTA grid must divide evenly by cluster dimensions")
    if cluster_m_ctas > 1 or cluster_n_ctas > 1:
        kernel.attrs["cluster_dim"] = [
            cluster_by_axis[CollectiveAxis.X],
            cluster_by_axis[CollectiveAxis.Y],
            1,
        ]
    pipeline = Pipeline()
    # Both output dimensions are flattened onto block X for a fixed grid.
    # TileMatmul linearizes the last op dimension fastest, so preserve the
    # selected traversal order when choosing that dimension.
    if output_grid_fast_axis not in ("m", "n"):
        raise ValueError("output_grid_fast_axis must be 'm' or 'n'")
    m_fast = output_grid_size is not None and output_grid_fast_axis == "m"
    if m_fast and group_size != 1:
        raise ValueError("M-fast fixed grids do not support tile groups")
    if group_size == 1:
        pipeline.add_pass(
            TileMatmul(
                op_dims=["N", "M"] if m_fast else ["M", "N"],
                tile_sizes=(
                    [output_n_tile, output_m_tile]
                    if m_fast
                    else [output_m_tile, output_n_tile]
                ),
                iteration_bindings=output_bindings,
                for_loop_labels=[None, None],
                op_label_filter="mm",
                epilogue_label_filter=epilogue_label_filter,
                allow_partial_tiles=(m % output_m_tile != 0 or n % output_n_tile != 0),
                reversed_dims=list(output_reversed_dims),
            )
        )
    else:
        grid = (blocks + group_size - 1) // group_size
        pipeline.add_pass(
            TileMatmul(
                op_dims=[group_dim, "M", "N"],
                tile_sizes=[
                    grid * tile_size,
                    output_m_tile,
                    output_n_tile,
                ],
                iteration_bindings=[None, *output_bindings],
                for_loop_labels=[group_label, None, None],
                op_label_filter="mm",
                epilogue_label_filter=epilogue_label_filter,
                allow_partial_tiles=(
                    m % output_m_tile != 0
                    or n % output_n_tile != 0
                    or blocks % group_size != 0
                ),
                reversed_dims=list(output_reversed_dims),
            )
        )
    pipeline.add_pass(
        schedule_tiled_matmul(
            label="mm",
            mma_k_tile=32 if matmul.lhs.type.dtype.name in ("e4m3", "e5m2") else 16,
            epilogue_label_filter=epilogue_label_filter,
            output_m_tile=output_m_tile,
            output_n_tile=output_n_tile,
            operand_stages=operand_stages,
            operand_stage_overrides=operand_stage_overrides,
            operand_k_tile=operand_k_tile,
            warpgroup_m_tile=warpgroup_m_tile,
            cyclic_operand_pipeline=cyclic_operand_pipeline,
            warpgroup_n_tile=warpgroup_n_tile,
            output_slices=output_slices,
            output_store_buffers=output_store_buffers,
            output_fragment_cols=output_fragment_cols,
            output_fragment_swizzle=output_fragment_swizzle,
            output_tma_inner_size=output_tma_inner_size,
            operand_move_attrs=operand_move_attrs,
            atomic_add=matmul.attrs.get("atomic_add") is True,
            output_tma=output_store != "global",
            unsafe_reuse_smem=unsafe_reuse_smem,
        )
    )
    if group_size > 1:
        pipeline.add_pass(
            UnrollLoop(
                unroll_factor=group_size,
                label_filter=group_label,
            )
        )
    pipeline.add_pass(ScheduleAsync())
    if cyclic_operand_pipeline:
        # Peeled stages outside the queued loop keep one MMA group in flight.
        from ..passes.queue_async_tail import QueueAsyncTail

        pipeline.add_pass(QueueAsyncTail())
    pipeline.add_pass(Canonicalize())
    module = pipeline.run(module)
    if warpgroup_ctas == 2:
        if kernel.block_dim != (128, 1, 1):
            raise ValueError("paired warpgroups require a flat 128-thread task program")
        kernel.attrs["warpgroup_ctas"] = warpgroup_ctas
        kernel.attrs["warpgroup_operand_steps"] = math.ceil(k / operand_k_tile)
    if unsafe_reuse_smem:
        _fence_shared_writeback(module.body)
    return module


def _fence_shared_writeback(block: Block) -> None:
    """Retire every warpgroup's operand reads before reusing the compute arena."""
    writers = [
        i for i, op in enumerate(block.ops)
        if isinstance(op, MoveOp) and isinstance(op.dst.type, SmemTensorType)
        and any(
            isinstance(root.definition, AllocOp)
            and str(root.definition.attrs.get("reuse_group", "")).endswith("_writeback")
            for root in root_buffers(op.dst)
        )
    ]
    if writers:
        # The validated schedule puts this block inside a CTA-uniform output
        # tile guard. A faster warpgroup must not overwrite another's operands.
        block.ops.insert(writers[0], BarrierOp.build("block"))
        # TMA reads must also retire before the next tile reloads the arena.
        block.ops.append(BarrierOp.build("block", attrs={"drain_bulk": True}))
    else:
        for op in block.ops:
            for nested in op.blocks:
                _fence_shared_writeback(nested)


class TileMatmulPipeline(Pipeline):
    """Apply the unified configurable schedule to a concrete matmul."""

    def __init__(
        self, *, m: int, n: int, k: int, register_accumulator: bool = False, **config
    ) -> None:
        super().__init__()
        self.dimensions = (m, n, k)
        self.register_accumulator = register_accumulator
        self.config = config

    def run(self, module: ModuleOp) -> ModuleOp:
        if self.register_accumulator:
            return schedule_register_matmul(**self.config).run(module)
        return _schedule_matmul(module, **self.config)
