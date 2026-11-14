"""Blackwell GEMMs: the configurable GEMM schedule on a tcgen05 target."""

from copy import deepcopy
from typing import Any

from ..gpu_dialect import CollectiveAxis, ModuleOp, gpu_dialect
from ..pipelines.tile_matmul_pipeline import _schedule_matmul


def blackwell_tcgen05_gemm_module(
    *,
    problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    output_m_tiles_per_cta: int = 1,
    batch_output_m_operands: bool = False,
    ordered_mma_completion: bool = False,
    interleave_mma_descriptors: bool = False,
    hoist_mma_descriptors: bool = False,
    persistent_output_pipeline: bool = False,
    persistent_output_overlap: bool = True,
    persistent_output_schedule: str = "shared_ring",
    persistent_output_job_regions: bool = False,
    operand_k_tile: int = 64,
    operand_stages: int = 3,
    cyclic_operand_pipeline: bool | int = False,
    mma_issue_group_size: int = 1,
    rolled_operand_pipeline: bool = False,
    integer_index_division: bool | None = None,
    dtype: str = "bf16",
    output_grid_size: int | None = None,
    output_grid_fast_axis: str = "m",
    output_store: str = "global",
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_store_pending_groups: int = 0,
    prefetch_tma_descriptors: bool = False,
    output_layout: str | None = None,
    output_column_major: bool = False,
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    allow_partial_tiles: bool = False,
    reuse_tmem_allocation: bool = False,
    reuse_completion_locks: bool = False,
    block_gather_backend: str | None = None,
    tmem_writeback_register_budget: int = 32,
    tmem_writeback_warps: int = 4,
    hoist_invariant_operand_copies: bool = False,
    reuse_completed_operand_copies: bool = False,
    reuse_operand_copies_across_barriers: bool = False,
    warp_specific_operand_waits: bool = False,
    consumer_specific_mma_waits: bool = False,
    paired_completion_handoffs: bool = False,
    cta_group: int = 1,
    cluster_m_ctas: int | None = None,
    cluster_n_ctas: int = 1,
    cluster_dim: tuple[int, int, int] | None = None,
    multicast_operands: bool = True,
) -> ModuleOp:
    """Build ``a @ bt.T`` using common warp loops on Blackwell.

    Native schedules support paired CTAs, multicast clusters, cyclic MMA
    queues and optional descriptor interleaving/hoisting. ``mma_issue_group_size``
    changes native cyclic issue order without changing completion ownership.

    ``persistent_output_pipeline`` selects the shared operand/output ring:
    two TMEM slots have independent accumulator readiness, last-read release
    and output-staging lifetimes. ``rolled_operand_pipeline`` keeps its K loop
    compact. ``persistent_output_job_regions`` rotates whole compute/output
    jobs across participant groups within that same persistent loop.

    Output transposition uses swapped input dimensions and a column-major
    destination; callers launch with ``bt, a, c.T``. Logical cluster ranks
    are M-fast, with CTA pairs at ranks ``r`` and ``r XOR 1``; ``cluster_dim``
    selects the physical launch independently of the logical tile mapping.
    """
    cluster_m_ctas = cta_group if cluster_m_ctas is None else cluster_m_ctas
    # _schedule_matmul validates the layout; its CTA count is the physical one.
    extended_cluster = (cluster_dim or (cluster_m_ctas, cluster_n_ctas, 1)) != (cta_group, 1, 1)
    # Checks below whose only downstream counterpart is in the shared-ring passes stay here.
    if extended_cluster and (persistent_output_pipeline or batch_output_m_operands):
        raise ValueError("extended clusters require the native operand schedule")
    if type(paired_completion_handoffs) is not bool:
        raise ValueError("paired_completion_handoffs must be a boolean")
    if paired_completion_handoffs and (
        cta_group != 2
        or extended_cluster
        or not cyclic_operand_pipeline
        or rolled_operand_pipeline
        or persistent_output_pipeline
        or batch_output_m_operands
        or output_m_tiles_per_cta != 1
    ):
        raise ValueError(
            "paired completion handoffs require an ordinary native cyclic CTA pair"
        )
    if type(mma_issue_group_size) is not int or mma_issue_group_size < 1:
        raise ValueError("mma_issue_group_size must be a positive integer")
    if mma_issue_group_size > 1 and (
        not cyclic_operand_pipeline
        or rolled_operand_pipeline
        or persistent_output_pipeline
        or batch_output_m_operands
    ):
        raise ValueError(
            "MMA issue grouping requires the native cyclic operand pipeline"
        )
    if rolled_operand_pipeline and batch_output_m_operands:
        raise ValueError("rolled operands require an unbatched operand schedule")
    if (
        rolled_operand_pipeline
        and persistent_output_pipeline
        and operand_k_tile not in (64, 128)
    ):
        raise ValueError("rolled shared rings require K64 or K128 stages")
    if (
        rolled_operand_pipeline
        and int(cyclic_operand_pipeline) > 1
        and not persistent_output_pipeline
    ):
        raise ValueError(
            "an MMA queue depth above one requires the native cyclic operand pipeline"
        )
    if persistent_output_schedule != "shared_ring":
        raise ValueError("persistent_output_schedule must be shared_ring")
    if rolled_operand_pipeline and operand_stages < 2:
        raise ValueError("rolled operand pipeline requires at least two stages")
    if persistent_output_job_regions and not (
        persistent_output_pipeline
        and rolled_operand_pipeline
        and cta_group == 2
        and tmem_writeback_warps == 4
        and persistent_output_overlap
        and int(cyclic_operand_pipeline) <= 1
    ):
        raise ValueError(
            "job participant regions require an overlapping paired rolled ring and four-warp fragments"
        )
    shared_ring = persistent_output_pipeline
    if shared_ring and (
        output_grid_size is None
        or cta_group == 2
        and (output_grid_size % 2 or output_grid_fast_axis != "m")
        or output_m_tile != 128
        or output_m_tiles_per_cta != 1
        or batch_output_m_operands
        or reuse_tmem_allocation
        or warp_specific_operand_waits
        or consumer_specific_mma_waits
        or output_n_tile > 256
        or output_store != "stmatrix"
        or operand_stages < 2
        or any(
            a.get("issuer_warp", 0) or "participant_warps" in a
            for a in (operand_move_attrs or {}).values()
        )
    ):
        raise ValueError(
            "shared ring requires a persistent single-CTA TMA schedule with common participants"
        )
    if reuse_operand_copies_across_barriers and not reuse_completed_operand_copies:
        raise ValueError("barrier copy reuse requires completed operand copy reuse")
    if output_store_pending_groups and output_store != "stmatrix":
        raise ValueError("output store queues require TMA writeback")
    if type(cta_group) is not int or cta_group not in (1, 2):
        raise ValueError("cta_group must be one or two")
    if block_gather_backend is not None and block_gather_backend not in (
        "named",
        "mbar",
    ):
        raise ValueError("block_gather_backend must be named or mbar")
    m, n, k = (problem_size if dim is None else dim for dim in (m, n, k))
    if rolled_operand_pipeline and persistent_output_pipeline and k <= 2 * operand_k_tile:
        raise ValueError("rolled operand pipelines require at least three K stages")
    if cta_group == 2 and output_m_tile == 64 and operand_k_tile > 64 and k % 64:
        raise ValueError("paired M64 blocked TMA inputs require K divisible by 64")
    if (
        min(m, n, k, output_n_tile) <= 0
        or output_m_tile not in (64, 128)
        or (
            m % output_m_tile
            and not (output_store == "stmatrix" and allow_partial_tiles)
        )
        or n % 8
        or (not allow_partial_tiles and n % output_n_tile)
    ):
        raise ValueError(
            "GEMM dimensions must be positive and divide the M64/M128 and N tiles"
        )
    if (
        operand_k_tile <= 0
        or k % 16
        or (not allow_partial_tiles and k % operand_k_tile)
    ):
        raise ValueError("the K tile must divide the reduction")
    if not 1 <= operand_stages <= (k + operand_k_tile - 1) // operand_k_tile:
        raise ValueError("stages must be positive and fit the reduction")
    # B200 permits 227 KiB per CTA. Leave 1 KiB for alignment and barriers.
    operand_bytes = (
        (
            output_m_tile * (output_m_tiles_per_cta if batch_output_m_operands else 1)
            + output_n_tile // cta_group
        )
        * operand_k_tile
        * 2
        * operand_stages
    )
    if operand_bytes > 226 * 1024:
        raise ValueError("operand stages exceed Blackwell shared-memory capacity")
    if dtype not in ("bf16", "f16"):
        raise ValueError("dtype must be bf16 or f16")
    name = f"blackwell_m{m}_n{n}_k{k}_{dtype}_n{output_n_tile}_k{operand_k_tile}_s{operand_stages}"
    if output_m_tile != 128:
        name += f"_mt{output_m_tile}"
    if tmem_writeback_warps != 4:
        name += f"_w{tmem_writeback_warps}"
    if cta_group == 2:
        name += "_2cta"
    output_type = (
        f"gmem<{m}x{n}x{dtype}"
        + (f" strides=[1, {m}]" if output_column_major else "")
        + ">"
    )
    if output_column_major:
        name += "_cm"
    module = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: {output_type}) {{
            %zero: rmem<f32> = constant 0.0
            matmul(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: {output_type}, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
          }}
        }}
    """)
    if type(hoist_mma_descriptors) is not bool:
        raise ValueError("hoist_mma_descriptors must be a boolean")
    if block_gather_backend is not None:
        module.body.ops[0].attrs["block_gather_backend"] = block_gather_backend
    for attr, enabled in (
        ("reuse_completion_locks", reuse_completion_locks),
        ("ordered_mma_completion", ordered_mma_completion),
        ("interleave_mma_descriptors", interleave_mma_descriptors),
        ("hoist_mma_descriptors", hoist_mma_descriptors),
    ):
        if enabled:
            module.body.ops[0].attrs[attr] = True
    # A launch-bound grid traverses M fastest on X, so N-fast swaps the axes.
    axes = (
        (CollectiveAxis.Y, CollectiveAxis.X)
        if output_grid_size is None and output_grid_fast_axis == "n"
        else (CollectiveAxis.X, CollectiveAxis.Y)
    )
    scheduled = _schedule_matmul(
        module,
        output_grid_size=output_grid_size,
        output_grid_axes=axes,
        output_grid_fast_axis=output_grid_fast_axis,
        cluster_m_ctas=cluster_m_ctas,
        cluster_n_ctas=cluster_n_ctas,
        cluster_dim=cluster_dim,
        cta_group=cta_group,
        multicast_operands=multicast_operands,
        output_m_tile=output_m_tile,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        batch_output_m_operands=batch_output_m_operands,
        persistent_output_pipeline=persistent_output_pipeline,
        output_n_tile=output_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        cyclic_operand_pipeline=cyclic_operand_pipeline,
        rolled_operand_pipeline=rolled_operand_pipeline,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_store=output_store,
        output_layout=output_layout,
        operand_move_attrs=operand_move_attrs,
        masked_global_store=output_store == "global"
        and allow_partial_tiles
        and n % output_n_tile != 0,
        reuse_tmem_allocation=reuse_tmem_allocation,
        reuse_completion_locks=reuse_completion_locks,
        tmem_writeback_register_budget=tmem_writeback_register_budget,
        tmem_writeback_warps=tmem_writeback_warps,
    )
    if shared_ring:
        from ..passes.pipeline_shared_ring import PipelineSharedRing

        scheduled = PipelineSharedRing(
            operand_stages=operand_stages,
            overlap=persistent_output_overlap,
            rolled_operand_pipeline=rolled_operand_pipeline,
            mma_queue_depth=max(1, int(cyclic_operand_pipeline)),
            job_regions=persistent_output_job_regions,
        ).run(scheduled)
    kernel = scheduled.body.ops[0]
    if type(prefetch_tma_descriptors) is not bool:
        raise ValueError("prefetch_tma_descriptors must be a boolean")
    if prefetch_tma_descriptors:
        kernel.attrs["prefetch_tma_descriptors"] = True
    if integer_index_division is not None and type(integer_index_division) is not bool:
        raise ValueError("integer_index_division must be a boolean or None")
    if rolled_operand_pipeline if integer_index_division is None else integer_index_division:
        kernel.attrs["integer_index_division"] = True
    repeated_tmem = kernel.has_repeated_tmem_allocation
    backend = block_gather_backend or ("mbar" if repeated_tmem else "named")
    if backend == "named" and repeated_tmem:
        raise ValueError(
            "named gathers require a reusable TMEM allocation; "
            "use block_gather_backend='mbar'"
        )
    if backend != "named":
        kernel.attrs["block_gather_backend"] = backend
    if hoist_invariant_operand_copies:
        from ..passes.hoist_invariant_copies import HoistInvariantCopies

        scheduled = HoistInvariantCopies().run(scheduled)
    if reuse_completed_operand_copies:
        from ..passes.eliminate_redundant_copies import EliminateRedundantCopies

        scheduled = EliminateRedundantCopies(
            preserve_barrier_facts=reuse_operand_copies_across_barriers
        ).run(scheduled)
    # Paired K loops narrow bank waits to each CTA's issuer and retire the
    # peer's old phase read through the paired refill's completion. This
    # replaces a cluster-wide software gather every second stage. Persistent
    # and grouped multi-tile paired schedules keep their own handoffs.
    paired_relay = cta_group == 2 and not (
        extended_cluster or persistent_output_pipeline or batch_output_m_operands
    )
    if paired_relay:
        kernel.attrs["correlate_uniform_guards"] = True
        kernel.attrs["paired_phase_relay"] = True
    if paired_completion_handoffs:
        kernel.attrs["paired_completion_handoffs"] = True
    from ..passes.pipeline_cluster_banks import PipelineClusterBanks

    scheduled = PipelineClusterBanks().run(scheduled)
    if mma_issue_group_size > 1:
        from ..passes.schedule_async import ScheduleAsync

        scheduled = ScheduleAsync(compute_issue_group_size=mma_issue_group_size).run(
            scheduled
        )
    if (
        warp_specific_operand_waits or consumer_specific_mma_waits or paired_relay
    ) and not kernel.attrs.get("cluster_bank_pipeline"):
        from ..passes.schedule_warp_waits import ScheduleWarpWaits

        scheduled = ScheduleWarpWaits(
            operand_ready=warp_specific_operand_waits,
            mma_reuse=consumer_specific_mma_waits,
        ).run(scheduled)
    if output_store_pending_groups:
        from ..passes.schedule_async import ScheduleAsync

        scheduled = ScheduleAsync(
            store_pending_groups=output_store_pending_groups
        ).run(scheduled)
    return scheduled


# B200 BF16, one graph-captured launch after a 256 MiB L2 flush.
_B200_SQUARE_SCHEDULES: dict[int, dict[str, Any]] = {
    512: {"output_n_tile": 16, "operand_k_tile": 256, "operand_stages": 1},
    1024: {"output_n_tile": 64, "operand_k_tile": 256, "operand_stages": 2},
    2048: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
    },
    4096: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
        "output_grid_size": 148,
        "output_grid_fast_axis": "n",
    },
    8192: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
        "output_grid_size": 1184,
        "output_grid_fast_axis": "n",
    },
    16384: {
        "output_n_tile": 256,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "cyclic_operand_pipeline": True,
        "output_grid_fast_axis": "m",
        "operand_move_attrs": {"rhs": {"l2_cache_hint": "evict_last"}},
    },
}


def select_blackwell_square_schedule(problem_size: int) -> dict[str, Any]:
    """Return the measured B200 BF16 schedule for a supported square size.

    Measurements cover 512, 1024, 2048, 4096, 8192, and 16384. The copy allows
    callers to refine parameters without mutating the measured table.
    """
    if problem_size not in _B200_SQUARE_SCHEDULES:
        raise ValueError(f"no measured B200 BF16 square schedule for {problem_size}")
    return deepcopy(_B200_SQUARE_SCHEDULES[problem_size])


def blackwell_square_gemm_module(*, problem_size: int = 4096) -> ModuleOp:
    """Build a BF16 square GEMM using its measured B200 schedule."""
    return blackwell_tcgen05_gemm_module(
        problem_size=problem_size, **select_blackwell_square_schedule(problem_size)
    )
