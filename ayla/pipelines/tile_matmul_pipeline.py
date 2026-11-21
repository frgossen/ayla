"""Configurable GEMM module builders used by collection entries."""

from ..base.ir import find_ops_of_type
from ..base.pipeline import Pipeline
from ..gpu_dialect import (
    BF16Type,
    F16Type,
    CollectiveAxis,
    CollectiveLevel,
    KernelOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    ReshapeOp,
    TileOp,
    root_buffer,
)
from .canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.licm import LICM
from ..passes.lower_to_async import LowerToAsync
from ..passes.multi_buffer import MultiBuffer
from ..passes.peel_loop import PeelLoop
from ..passes.pipeline_loop import PipelineLoop
from ..passes.promote_buffers import PromoteBuffers
from ..passes.reduction_range import ReductionRange
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop


from copy import deepcopy
from typing import cast
import math


def _add_unrolled_subtiles(pipeline, label, dim, size, loop_label):
    """Split each MMA into fully unrolled ``size`` subtiles along ``dim``."""
    pipeline.add_pass(TileMatmul(op_dims=[dim], tile_sizes=[size], iteration_bindings=[None],
                                 for_loop_labels=[loop_label], op_label_filter=label))
    pipeline.add_pass(UnrollLoop(unroll_factor=None, label_filter=loop_label))
    pipeline.add_pass(Canonicalize())


def schedule_register_matmul(
    *,
    label: str,
    operand_stages: int,
    operand_k_tile: int = 64,
    cyclic_operand_pipeline: bool | int = False,
    rolled_operand_pipeline: bool = False,
    rhs_tma_load_slices: int = 1,
    lhs_tma_load_slices: int = 1,
    lhs_column_major: bool = False,
    rhs_column_major: bool = False,
    lhs_smem_block_width: int = 64,
    rhs_smem_block_width: int = 64,
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
    allow_partial_k: bool = False,
    mma_n_tile: int | None = None,
    mma_m_tile: int | None = None,
    compose_m192: bool = False,
    destination_promotion: PromoteBuffers | None = None,
    cta_group: int = 1,
) -> Pipeline:
    """Schedule a pre-tiled GEMM with an explicit register accumulator.

    The caller owns output tiling, the WGRMEM destination, its epilogue, and
    any live-out moves. This pipeline only schedules the reduction and its
    operand movement.
    """
    if min(lhs_tma_load_slices,rhs_tma_load_slices) < 1:
        raise ValueError("rhs_tma_load_slices must be positive")
    k_label = f"{loop_label_prefix}_k" if loop_label_prefix else "k"
    kk_label = f"{loop_label_prefix}_kk" if loop_label_prefix else "kk"
    nn_label = f"{loop_label_prefix}_nn" if loop_label_prefix else "nn"

    pipeline = Pipeline()
    pipeline.add_pass(
        TileMatmul(
            op_dims=["K"],
            tile_sizes=[operand_k_tile],
            retain_dst_tiling_scheme=mma_m_tile is not None or compose_m192,
            iteration_bindings=[None],
            for_loop_labels=[k_label],
            op_label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
            allow_partial_tiles=allow_partial_k,
        )
    )
    pipeline.add_pass(ReductionRange(k_label))
    for operand, column_major, width in (
        ("lhs", lhs_column_major, lhs_smem_block_width),
        ("rhs", rhs_column_major, rhs_smem_block_width),
    ):
        pipeline.add_pass(
            PromoteBuffers(
                target_type="smem",
                operands=[operand],
                permutation=[1, 0] if column_major else None,
                swizzle=(f"{width * 2}B" if width >= 16 else None)
                if column_major
                else f"{min(operand_k_tile, 64) * 2}B",
                layout=None,
                warps=None,
                tma=True,
                physical_layout=(
                    {
                        "block_width": width if column_major else 64,
                        "order": "block_major",
                    }
                    if operand_k_tile > 64 or column_major
                    else None
                ),
                tma_load_slices=(
                    {operand: lhs_tma_load_slices if operand=="lhs" else rhs_tma_load_slices}
                ),
                move_attrs=operand_move_attrs,
                unsafe_reuse_smem=unsafe_reuse_smem,
                label_filter=label,
                cta_group=cta_group,
            )
        )
    if compose_m192:
        from ..passes.compose_matmul_rows import ComposeMatmulRows

        if destination_promotion is None:
            raise ValueError("M192 composition requires loop-scoped destination promotion")
        pipeline.add_pass(ComposeMatmulRows(label_filter=label))
        pipeline.add_pass(destination_promotion)
    elif mma_m_tile is not None:
        if destination_promotion is None:
            raise ValueError("M subtiles require loop-scoped destination promotion")
        _add_unrolled_subtiles(pipeline, label, "M", mma_m_tile, "mma_m")
        pipeline.add_pass(destination_promotion)
    pipeline.add_pass(
        TileMatmul(
            op_dims=["K"],
            tile_sizes=[16],
            iteration_bindings=[None],
            for_loop_labels=[kk_label],
            op_label_filter=rf"{label}(?:_iter\d+)*" if mma_m_tile is not None else label,
            epilogue_label_filter=epilogue_label_filter,
        )
    )
    if not rolled_operand_pipeline:
        pipeline.add_pass(
            MultiBuffer(
                depth=operand_stages,
                label_filter=f"{label}_lhs_smem",
            )
        )
        pipeline.add_pass(
            MultiBuffer(
                depth=operand_stages,
                label_filter=f"{label}_rhs_smem",
            )
        )
    pipeline.add_pass(LICM())
    if mma_n_tile is not None:
        # Shared operands have already been promoted at the CTA tile size.
        # MMA subtiles therefore reuse the same A allocation and one TMEM
        # accumulator rather than scheduling separate output tiles.
        _add_unrolled_subtiles(pipeline, label, "N", mma_n_tile, nn_label)
    # Unroll kk and peel the first K step, so the reduction's first WGMMA is a
    # single op the zeroed accumulator folds into as an overwrite.
    pipeline.add_pass(
        UnrollLoop(
            unroll_factor=operand_k_tile // 16,
            label_filter=rf"{kk_label}(?:_tail\d+)?(?:_iter\d+)?",
        )
    )
    pipeline.add_pass(Canonicalize())
    pipeline.add_pass(PeelLoop(label_filter=k_label, front_iterations=1))
    if not rolled_operand_pipeline:
        pipeline.add_pass(
            UnrollLoop(
                unroll_factor=operand_stages,
                label_filter=k_label,
            )
        )
    # A rolled reduction is the template of its pipeline, even when peeling
    # leaves it a single iteration.
    keep = k_label if rolled_operand_pipeline else None
    pipeline.add_pass(Canonicalize(keep_loops=keep))
    pipeline.add_pass(LowerToAsync())
    pipeline.add_pass(FuseAsync())
    pipeline.add_pass(LICM())
    pipeline.add_pass(Canonicalize(keep_loops=keep))
    if rolled_operand_pipeline:
        from .ring_schedule import LoopSchedule, PipelineSchedule
        pipeline.add_pass(PipelineLoop(k_label, operand_stages=operand_stages,
            plan=PipelineSchedule(reduction=LoopSchedule("rolled"),
                                  mma_pending=max(1, int(cyclic_operand_pipeline)))))
        return pipeline
    # A cyclic pipeline of depth d keeps d MMA batches queued: it refills the
    # last d banks after their MMA events, carried into the next iteration.
    queue_depth = int(cyclic_operand_pipeline)
    stages = "|".join(map(str, range(operand_stages - queue_depth)))
    pipeline.add_pass(
        PipelineLoop(
            label_filter_loop=k_label,
            label_filter_op=rf"{label}_(lhs|rhs)_smem_iter(?:{stages})_async",
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
                pipelining_offset=queue_depth,
            )
        )
        pipeline.add_pass(ScheduleAsync(pipeline_compute=True, mma_queue_depth=queue_depth))
    return pipeline


def schedule_tiled_matmul(
    *,
    label: str,
    lhs_column_major: bool = False,
    rhs_column_major: bool = False,
    lhs_smem_block_width: int = 64,
    rhs_smem_block_width: int = 64,
    output_m_tile: int,
    output_n_tile: int,
    operand_stages: int,
    operand_k_tile: int = 64,
    warpgroup_m_tile: int | None = None,
    cyclic_operand_pipeline: bool | int = False,
    rolled_operand_pipeline: bool = False,
    warpgroup_n_tile: int | None = None,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    tmem_writeback_register_budget: int = 32,
    tmem_writeback_warps: int = 4,
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    unsafe_reuse_smem: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    epilogue_label_filter: str | None = None,
    loop_label_prefix: str | None = None,
    accumulator: str = "wgrmem",
    output_layout: str | None = None,
    mma_m_tile: int | None = None,
    cta_group: int = 1,
) -> Pipeline:
    """Build the post-block-tiling GEMM pipeline.

    A ``"wgrmem"`` accumulator is split across warp groups, each issuing its
    own WGMMAs. A ``"tmem"`` accumulator holds the whole output tile in tensor
    memory for one thread's tcgen05 MMAs, so the tile needs no warp-group
    split; its writeback picks the register layout it drains through.
    """
    pipeline = Pipeline()
    tensor_memory = accumulator == "tmem"
    compose_m192 = tensor_memory and output_m_tile == 192
    mma_n_tile = (
        max(tile for tile in range(8, 257, 8) if output_n_tile % tile == 0)
        if tensor_memory and output_n_tile > 256 else None
    )
    if not tensor_memory:
        if warpgroup_n_tile is None:
            warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
        wg_tiles = (
            {"M": warpgroup_m_tile, "N": warpgroup_n_tile}
            if warpgroup_m_tile is not None
            else {"N": warpgroup_n_tile}
            if output_m_tile == 64 or output_n_tile > 256
            else {"M": 64}
        )
        pipeline.add_pass(
            TileMatmul(
                op_dims=list(wg_tiles),
                tile_sizes=list(wg_tiles.values()),
                iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)]
                * len(wg_tiles),
                for_loop_labels=[None] * len(wg_tiles),
                op_label_filter=label,
                epilogue_label_filter=epilogue_label_filter,
                allow_partial_tiles=(
                    list(wg_tiles) == ["M"] and output_m_tile % wg_tiles["M"] != 0
                ),
            )
        )
    destination_promotion = PromoteBuffers(
            scope_loop_label="k" if mma_m_tile is not None or compose_m192 else None,
            target_type=accumulator,
            operands=["dst"],
            swizzle=None,
            layout=output_layout or (None if tensor_memory else "wgmma_layout_0"),
            warps=tmem_writeback_warps if tensor_memory else 4,
            tma=output_store != "global",
            physical_layout=(
                {
                    "swizzle": "128B",
                    "block_width": 64,
                    "order": "block_major",
                }
                if output_tma_inner_size is not None
                or (
                    output_store == "stmatrix"
                    and output_n_tile // output_slices % 16 == 0
                )
                else None
            ),
            writeback_slices=output_slices,
            writeback_buffers=output_store_buffers,
            writeback_register_budget=tmem_writeback_register_budget,
            move_attrs=operand_move_attrs,
            target_dtype="f32",
            label_filter=rf"{label}(?:_iter\d+)*" if mma_m_tile is not None else label,
            epilogue_label_filter=epilogue_label_filter,
            unsafe_reuse_smem=unsafe_reuse_smem,
            cta_group=cta_group,
    )
    if mma_m_tile is None and not compose_m192:
        pipeline.add_pass(destination_promotion)
    pipeline.add_pass(
        schedule_register_matmul(
            label=label,
            mma_m_tile=mma_m_tile,
            compose_m192=compose_m192,
            destination_promotion=destination_promotion if mma_m_tile is not None or compose_m192 else None,
            lhs_tma_load_slices=(output_m_tile+255)//256 if mma_m_tile is not None else 1,
            operand_stages=operand_stages,
            operand_k_tile=operand_k_tile,
            cyclic_operand_pipeline=cyclic_operand_pipeline,
            rolled_operand_pipeline=rolled_operand_pipeline,
            allow_partial_k=tensor_memory,
            # Preserve the existing three 128-row RHS loads for the tuned N=384
            # schedule; other supported schedules use one RHS load.
            lhs_column_major=lhs_column_major,
            rhs_column_major=rhs_column_major,
            lhs_smem_block_width=lhs_smem_block_width,
            rhs_smem_block_width=rhs_smem_block_width,
            rhs_tma_load_slices=(
                output_n_tile // mma_n_tile
                if mma_n_tile is not None
                else output_n_tile // cast(int, warpgroup_n_tile)
                if output_n_tile == 384 and not tensor_memory
                else 1
            ),
            mma_n_tile=mma_n_tile,
            unsafe_reuse_smem=unsafe_reuse_smem,
            operand_move_attrs=operand_move_attrs,
            epilogue_label_filter=epilogue_label_filter,
            loop_label_prefix=loop_label_prefix,
            cta_group=cta_group,
        )
    )
    return pipeline


def _schedule_matmul(
    module: ModuleOp,
    *,
    output_m_tiles_per_cta: int = 1,
    batch_output_m_operands: bool = False,
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
    cross_job_pipeline: bool = False,
    cluster_m_ctas: int = 1,
    cluster_n_ctas: int = 1,
    cta_group: int | None = None,
    cluster_dim: tuple[int, int, int] | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    cyclic_operand_pipeline: bool | int = False,
    rolled_operand_pipeline: bool = False,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    masked_global_store: bool = False,
    tmem_writeback_register_budget: int = 32,
    tmem_writeback_warps: int = 4,
    reuse_tmem_allocation: bool = False,
    output_layout: str | None = None,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    multicast_operands: bool = False,
) -> ModuleOp:
    """Apply the configurable GEMM schedule to one concrete matmul.

    Keeping scheduling separate from module construction lets the JIT first
    specialize a dynamic template, then reuse the static collection's passes.
    A Blackwell target accumulates in tensor memory instead of registers.
    """
    [kernel] = module.body.ops
    if not isinstance(kernel, KernelOp):
        raise ValueError("schedule_matmul requires exactly one kernel")
    matmuls = find_ops_of_type(module, MatmulOp)
    if len(matmuls) != 1:
        raise ValueError("schedule_matmul requires exactly one matmul")
    tensor_memory = module.target == "sm_100a"
    if tensor_memory:
        from ..gemm_collection.common_warp_schedule import validate_common_warp_schedule

        validate_common_warp_schedule({"operand_move_attrs": operand_move_attrs})
        if matmuls[0].attrs.get("issuer_warp", 0) != 0 or "participant_warps" in matmuls[0].attrs:
            raise ValueError("Blackwell matmul requires a shared issuer without warp specialization")
    if rolled_operand_pipeline and not tensor_memory:
        raise ValueError("rolled operand pipeline requires the Blackwell target")
    cta_group = (2 if tensor_memory and (cluster_m_ctas,cluster_n_ctas)==(2,1) else 1) if cta_group is None else cta_group
    paired = tensor_memory and cta_group == 2
    from ..cluster_layout import ClusterLayout
    if cluster_dim is None and output_grid_size is None:
        # By default the launch cluster spans the grid axes that carry M and N,
        # as the CTA grid itself does; MapClusterTiles remaps only otherwise.
        extent = {output_grid_axes[0]: cluster_m_ctas, output_grid_axes[1]: cluster_n_ctas}
        cluster_dim = (extent.get(CollectiveAxis.X, 1), extent.get(CollectiveAxis.Y, 1), 1)
    layout = ClusterLayout(cluster_m_ctas, cluster_n_ctas, cluster_dim or (cluster_m_ctas,cluster_n_ctas,1), cta_group)
    clustered_work = output_grid_size is not None and layout.size > 1 and (
        layout.size > cta_group or layout.physical != (cta_group,1,1))
    if clustered_work and output_grid_size % layout.size:
        raise ValueError('persistent cluster grids require complete CTA rank groups')
    paired_maximum_n = 512 if output_store == "stmatrix" else 256
    if paired and (output_m_tile not in (64,128,192) or output_n_tile > paired_maximum_n or output_n_tile % 16
                   or output_m_tiles_per_cta not in (1, 2)
                   or output_n_tiles_per_cta != 1 or output_store not in ("stmatrix", "global")
                   or output_grid_axes[0] != CollectiveAxis.X):
        raise ValueError("paired MMA requires M64/M128/M192 local tiles, N16–512 with TMA stores (N16–256 with global stores), block-X pairs")
    if tensor_memory and (
        output_m_tile not in (64, 128, 192)
        or output_n_tile > 512
        or warpgroup_m_tile is not None
        or warpgroup_n_tile not in (None, output_n_tile)
        or output_store == "scalar"
    ):
        raise ValueError(
            "TMEM accumulators support 64/128/192-row tiles of at most 512 "
            "columns, without warp-group tiles or scalar stores"
        )
    if tensor_memory:
        from ..tmem_layout import TmemWarpLayout

        for native_rows in ((128, 64) if output_m_tile == 192 else (output_m_tile,)):
            TmemWarpLayout(native_rows, tmem_writeback_warps)
        kernel.block_dim = (tmem_writeback_warps * 32, 1, 1)
    if masked_global_store:
        if output_store != "global":
            raise ValueError("masked_global_store requires global output stores")
        kernel.attrs["masked_global_store"] = True
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
    if paired and output_m_tiles_per_cta == 2 and (output_m_tile != 128 or output_grid_size is not None
            or ((m + 127) // 128) % 4 or output_store != 'stmatrix' or output_n_tile > 256):
        raise ValueError('paired output frames require complete padded M512 groups and an ordinary TMA grid')
    if paired and output_store == 'global' and (m % (2*output_m_tile) or (n % output_n_tile and not masked_global_store)):
        raise ValueError("paired global stores require complete paired M and N tiles")
    if layout.size > 1 and output_store == 'global' and (m % (layout.m*output_m_tile) or (n % (layout.n*output_n_tile) and not masked_global_store)):
        raise ValueError('cluster global stores require complete logical cluster tiles')
    if paired and output_grid_size is not None and (output_grid_size % 2
            or output_grid_fast_axis != "m" or ((m + output_m_tile - 1) // output_m_tile) % 2 and not (cross_job_pipeline or clustered_work)):
        raise ValueError("paired persistent grids require even CTA and M-tile counts with M-fast traversal")

    # TMA stores permit partial output tiles. Each operand's SMEM atom must
    # divide its local tile, not the full global matrix dimension; dense TMA
    # descriptors zero-fill the final input box.
    partial_tma_outputs = tensor_memory and output_store == "stmatrix"
    lhs_atom_width = math.gcd(output_m_tile if partial_tma_outputs else m, 64)
    rhs_mma_columns = (
        max(width for width in range(8, 257, 8) if output_n_tile % width == 0)
        if tensor_memory and output_n_tile > 256 else output_n_tile
    )
    rhs_atom_width = math.gcd(
        rhs_mma_columns // cta_group if partial_tma_outputs else n, 64
    )
    for operand, atom_width, tile in (
        (matmul.lhs, lhs_atom_width, output_m_tile),
        (matmul.rhs, rhs_atom_width, output_n_tile),
    ):
        if operand.type.strides[0] == 1 and math.gcd(atom_width, tile, 64) < 16:
            raise ValueError("MN-major WGMMA requires at least 32-byte swizzle atoms")

    if batch_output_m_operands and (not tensor_memory or output_m_tile != 128
            or output_m_tiles_per_cta not in (2,4,8) or output_n_tiles_per_cta != 1
            or output_m_tiles_per_cta * output_n_tile > 512 or output_grid_size is not None
            or (cluster_m_ctas,cluster_n_ctas)!=(1,1) or output_store != "stmatrix"):
        raise ValueError("batched M operands require unclustered grouped M128 TMEM outputs and TMA stores")
    output_groups = (output_m_tiles_per_cta, output_n_tiles_per_cta)
    if any(group not in (1, 2, 3, 4, 5, 6, 8, 16) for group in output_groups):
        raise ValueError("output tile groups must each be 1, 2, 3, 4, 5, 6, 8, or 16")
    if output_m_tiles_per_cta > 1 and output_n_tiles_per_cta > 1:
        raise ValueError("output tiles may be grouped along only one axis")
    if min(cluster_m_ctas, cluster_n_ctas) < 1:
        raise ValueError("cluster dimensions must be positive")
    if not set(output_reversed_dims) <= {"M", "N"}:
        raise ValueError("output_reversed_dims must name M and/or N")
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if output_m_tile not in (64, 128, 192, 256):
        raise ValueError("output_m_tile must be 64, 128, 192, or 256")
    maximum_n = 512 if tensor_memory else 384
    if not 8 <= output_n_tile <= maximum_n or output_n_tile % 8:
        raise ValueError(f"output_n_tile must be a multiple of 8 between 8 and {maximum_n}")
    if operand_k_tile not in (16, 32, 64, 128) and not (
        tensor_memory and operand_k_tile % 64 == 0
    ):
        raise ValueError("operand_k_tile must be 16, 32, 64 or 128")
    if type(cyclic_operand_pipeline) not in (bool, int) or cyclic_operand_pipeline < 0:
        raise ValueError("cyclic_operand_pipeline must be a boolean or an MMA queue depth")
    if cyclic_operand_pipeline and operand_stages <= int(cyclic_operand_pipeline):
        raise ValueError("a cyclic operand pipeline of depth d requires more than d operand stages")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if tensor_memory or output_n_tile <= 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix", "global"):
        raise ValueError("output_store must be 'scalar', 'stmatrix', or 'global'")
    if output_store == "global":
        backing = root_buffer(matmul.dst)
        if backing is None or backing.type.strides[-1] != 1:
            raise ValueError("global output stores require a row-major backing tensor")
        if kernel.attrs.get("masked_global_store") and (
            backing.type.rank != 2
            or not isinstance(backing.type.dtype, (BF16Type, F16Type))
        ):
            raise ValueError(
                "masked global output requires unbatched Half GEMM"
            )
        m_backing, n_backing = backing.type.sizes[-2:]
        if not kernel.attrs.get("masked_global_store") and (
            m_backing % (output_m_tile * output_m_tiles_per_cta)
            or n_backing % (output_n_tile * output_n_tiles_per_cta)
        ):
            raise ValueError("global output stores require complete backing tiles")
        view = matmul.dst
        while isinstance(view.definition, (TileOp, ReshapeOp)):
            if isinstance(view.definition, TileOp) and any(
                index != 0 for index in view.definition.indices[-2:]
            ):
                raise ValueError(
                    "global output stores require zero matrix view offsets"
                )
            view = view.definition.arg
    if output_store in ("scalar", "global"):
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
        kernel.grid_dim = (output_grid_size, 1, 1)
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
    if paired:
        m_grid = -(-m_grid // 2) * 2
    n_grid = (n_blocks + output_n_tiles_per_cta - 1) // output_n_tiles_per_cta
    if layout.size > 1:
        m_grid = -(-m_grid // cluster_m_ctas) * cluster_m_ctas
        n_grid = -(-n_grid // cluster_n_ctas) * cluster_n_ctas
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
    if clustered_work:
        kernel.attrs['cluster_dim'] = [layout.size,1,1]
    if type(multicast_operands) is not bool:
        raise ValueError("multicast_operands must be a bool")
    operand_move_attrs = deepcopy(operand_move_attrs or {})
    if multicast_operands and layout.size > cta_group:
        # Clustered CTAs sharing an operand tile receive one multicast copy.
        for operand in ("lhs", "rhs"):
            masks = layout.operand_masks(operand)
            if any(m.bit_count() > 1 for m in masks):
                attrs = operand_move_attrs.setdefault(operand, {})
                if attrs.setdefault("multicast", True):
                    attrs.setdefault("multicast_cta_masks", list(masks))
                    attrs.setdefault("multicast_slices", 1)
    for operand,attrs in operand_move_attrs.items():
        if attrs.get('multicast') and 'multicast_cta_masks' not in attrs:
            attrs['multicast_cta_masks']=list(layout.operand_masks(operand))
    pipeline = Pipeline()
    # Both output dimensions are flattened onto block X for a fixed grid.
    # TileMatmul linearizes the last op dimension fastest, so preserve the
    # selected traversal order when choosing that dimension.
    if output_grid_fast_axis not in ("m", "n"):
        raise ValueError("output_grid_fast_axis must be 'm' or 'n'")
    m_fast = output_grid_size is not None and output_grid_fast_axis == "m"
    if m_fast and group_size != 1:
        raise ValueError("M-fast fixed grids do not support tile groups")
    if batch_output_m_operands:
        pipeline.add_pass(TileMatmul(op_dims=["M","N"],
            tile_sizes=[output_m_tile*group_size,output_n_tile],
            iteration_bindings=output_bindings,for_loop_labels=[None,None],
            op_label_filter="mm",allow_partial_tiles=True))
    elif group_size == 1:
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
                allow_partial_tiles=(layout.size > 1 or paired or m % output_m_tile != 0 or n % output_n_tile != 0),
                pad_to_cluster=layout.size > 1 and (output_grid_size is None or cross_job_pipeline or clustered_work),
                cluster_layout=layout if clustered_work else None,
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
            mma_m_tile=output_m_tile if batch_output_m_operands else None,
            lhs_smem_block_width=lhs_atom_width,
            rhs_smem_block_width=rhs_atom_width,
            lhs_column_major=matmul.lhs.type.strides[0] == 1,
            rhs_column_major=matmul.rhs.type.strides[0] == 1,
            output_m_tile=output_m_tile*(group_size if batch_output_m_operands else 1),
            output_n_tile=output_n_tile,
            operand_stages=operand_stages,
            operand_k_tile=operand_k_tile,
            warpgroup_m_tile=warpgroup_m_tile,
            cyclic_operand_pipeline=cyclic_operand_pipeline,
            rolled_operand_pipeline=rolled_operand_pipeline,
            warpgroup_n_tile=warpgroup_n_tile,
            output_slices=output_slices,
            output_store_buffers=output_store_buffers,
            output_tma_inner_size=output_tma_inner_size,
            output_store=output_store,
            operand_move_attrs=operand_move_attrs,
            accumulator="tmem" if tensor_memory else "wgrmem",
            tmem_writeback_register_budget=tmem_writeback_register_budget,
            tmem_writeback_warps=tmem_writeback_warps,
            output_layout=output_layout,
            cta_group=2 if paired else 1,
        )
    )
    if group_size > 1 and not batch_output_m_operands:
        pipeline.add_pass(
            UnrollLoop(
                unroll_factor=group_size,
                label_filter=group_label,
            )
        )
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    if tensor_memory and output_m_tile == 192:
        from ..passes.pack_tmem_allocs import PackTmemAllocs

        pipeline.add_pass(PackTmemAllocs(compose_rows=True, track_column_regions=True,
                                       preserve_release_barriers=False))
    if tensor_memory and group_size > 1:
        from ..passes.pack_tmem_allocs import PackTmemAllocs
        pipeline.add_pass(PackTmemAllocs())
    if reuse_tmem_allocation:
        if not tensor_memory:
            raise ValueError("TMEM allocation reuse requires a Blackwell target")
        # One pass crosses a persistent loop's guard; the second crosses the
        # loop. The resource allocation and its release always move together.
        pipeline.add_pass(LICM(reuse_tmem_allocation=True))
        pipeline.add_pass(LICM(reuse_tmem_allocation=True))
    scheduled = pipeline.run(module)
    if layout.size > 1 and (clustered_work or layout.physical != tuple(d or 1 for d in scheduled.body.ops[0].cluster_dim) or layout.m > cta_group):
        from ..passes.map_cluster_tiles import MapClusterTiles
        scheduled = MapClusterTiles(layout, output_grid_axes, persistent_grid=clustered_work).run(scheduled)
    return scheduled


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
