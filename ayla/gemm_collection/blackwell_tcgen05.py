"""Blackwell GEMMs: the configurable GEMM schedule on a tcgen05 target."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ..gpu_dialect import CollectiveAxis, ModuleOp, gpu_dialect
from ..pipelines.tile_matmul_pipeline import _schedule_matmul
from .common_warp_schedule import validate_common_warp_schedule


def select_blackwell_gemm_schedule(
    m: int, n: int, k: int, dtype: str = "f16"
) -> dict[str, Any] | None:
    """Select an exact measured MM route, including composed M192 tiles."""
    dtype = {"f16": "float16", "bf16": "bfloat16"}.get(dtype, dtype)
    table = (
        Path(__file__).resolve().parents[1]
        / "better_benchmark_gemm_router_blackwell.json"
    )
    for route in json.loads(table.read_text())["routes"]:
        if (
            route.get("op", "mm"),
            route.get("batch", 1),
            route["dtype"],
            route["m"],
            route["n"],
            route["k"],
        ) == ("mm", 1, dtype, m, n, k):
            return deepcopy(route["configuration"])
    return None


def blackwell_tcgen05_gemm_module(
    *,
    problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    reuse_persistent_lhs: bool = False,
    reuse_persistent_rhs: bool = False,
    grouped_compute_overlap: bool = False,
    resident_phase_order_split: int | None = None,
    batch_output_m_operands: bool = False,
    ordered_mma_completion: bool = False,
    interleave_mma_descriptors: bool = False,
    compact_mma_batch: bool = False,
    hoist_mma_descriptors: bool = False,
    reuse_mma_collector_a: bool = False,
    rolled_k_unroll: int = 1,
    rolled_job_unroll: int = 1,
    peel_first_output_pair: bool = False,
    persistent_output_pipeline: bool = False,
    persistent_output_overlap: bool = True,
    persistent_output_schedule: str = "shared_ring",
    persistent_output_job_regions: bool = False,
    persistent_output_frames: int = 2,
    compute_batches_per_fragment: int = 0,
    pipeline_output_tiles: bool = False,
    carry_output_store_queue: bool = False,
    automatic_optimizations: bool = False,
    output_writeback_window: tuple[int, int] | None = None,
    operand_k_tile: int = 64,
    operand_stages: int = 3,
    cyclic_operand_pipeline: bool | int = False,
    mma_issue_group_size: int = 1,
    rolled_operand_pipeline: bool = False,
    # Earlier refills can help cold inputs but restrict compute overlap.
    rolled_refill_before_compute: bool = False,
    integer_index_division: bool | None = None,
    dtype: str = "bf16",
    output_grid_size: int | None = None,
    output_grid_fast_axis: str = "m",
    output_store: str = "global",
    output_slices: int = 1,
    output_store_buffers: int = 1,
    reuse_operand_storage_for_output: bool = False,
    reuse_all_operand_storage_for_output: bool = False,
    output_store_pending_groups: int = 0,
    prefetch_tma_descriptors: bool = False,
    preserve_prefetch_issue_order: bool = False,
    output_layout: str | None = None,
    output_column_major: bool = False,
    rhs_column_major: bool = False,
    bias: bool = False,
    batch: int | None = None,
    lhs_strides: tuple[int, ...] | None = None,
    rhs_strides: tuple[int, ...] | None = None,
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

    Explicit operand strides have two entries, or three with ``batch``.
    They override the layout defaults; ``rhs_column_major`` selects the
    default RHS matrix strides for both unbatched and batched operands.

    Native schedules support paired CTAs, multicast clusters, cyclic MMA
    queues and optional descriptor interleaving/hoisting. ``mma_issue_group_size``
    changes native cyclic issue order without changing completion ownership.

    ``output_m_tile=192`` composes native M128 and M64 updates with shared
    operands and one TMEM arena. It supports native/cyclic/rolled K schedules,
    multicast and paired CTAs, with at most 256 output columns per tile.
    Paired composition requires one output slice/buffer and enough writeback
    registers for the complete M64 component (budget 128 for N192, four warps).
    Grouped output tiles and persistent output rings are not supported for M192.

    ``persistent_output_pipeline`` selects the shared operand/output ring:
    two TMEM slots have independent accumulator readiness, last-read release
    and output-staging lifetimes. ``rolled_operand_pipeline`` keeps its K loop
    compact. ``persistent_output_job_regions`` rotates whole compute/output
    jobs across participant groups within that same persistent loop.

    Native fixed-grid schedules can opt into ``pipeline_output_tiles`` and
    ``carry_output_store_queue`` independently. ``automatic_optimizations``
    bundles allocation/event reuse, invariant copies and native store carry.
    These options leave the default schedule unchanged. Alternative output
    schedules ``pipelined``, ``paired`` and ``framed`` retain native or compact
    K pipelines; ``shared`` composes native compute with shared writeback.
    ``compute_batches_per_fragment`` bounds their future MMA issue cadence.
    Independent grouped M frames use ``grouped_compute_overlap`` instead.

    ``reuse_persistent_lhs`` / ``reuse_persistent_rhs`` retain the last K64
    panel across two complete adjacent output jobs by reversing the second
    reduction. ``resident_phase_order_split`` reverses two-job output order
    for the upper portion of a fixed grid. Both are per-schedule experiments.

    ``reuse_operand_storage_for_output`` shares output staging with a retired
    LHS buffer. ``reuse_all_operand_storage_for_output`` shares it with the
    complete retired operand arena. Both require compiler-verified completion
    of operand loads and MMA before output staging. Complete arena reuse also
    supports one fully resident native CTA pair with a drained MMA phase.

    Output transposition uses swapped input dimensions and a column-major
    destination; callers launch with ``bt, a, c.T``. Logical cluster ranks
    are M-fast, with CTA pairs at ranks ``r`` and ``r XOR 1``; ``cluster_dim``
    selects the physical launch independently of the logical tile mapping.
    """
    if reuse_persistent_lhs and reuse_persistent_rhs:
        raise ValueError("select one operand to retain across paired output jobs")
    if grouped_compute_overlap and (output_m_tiles_per_cta <= 1 or persistent_output_pipeline
                                   or rolled_operand_pipeline or batch_output_m_operands
                                   or reuse_persistent_lhs or reuse_persistent_rhs):
        raise ValueError("grouped compute overlap requires independent native M output frames")
    if reuse_persistent_lhs:
        output_n_tiles_per_cta = 2
    if reuse_persistent_rhs:
        output_m_tiles_per_cta = 2
    if (reuse_persistent_lhs or reuse_persistent_rhs) and (cta_group != 1 or output_grid_size is not None
            or output_store != "stmatrix" or operand_k_tile != 64 or output_n_tile > 256
            or not cyclic_operand_pipeline or persistent_output_pipeline or batch_output_m_operands):
        raise ValueError("retained operand banks require native paired jobs, K64 and TMA output")
    for name, enabled in (("pipeline_output_tiles", pipeline_output_tiles),
                          ("carry_output_store_queue", carry_output_store_queue),
                          ("automatic_optimizations", automatic_optimizations),
                          ("grouped_compute_overlap", grouped_compute_overlap),
                          ("reuse_persistent_lhs", reuse_persistent_lhs),
                          ("reuse_persistent_rhs", reuse_persistent_rhs)):
        if type(enabled) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if pipeline_output_tiles and (output_grid_size is None or persistent_output_pipeline
                                 or rolled_operand_pipeline or batch_output_m_operands):
        raise ValueError("native tile prefetch requires a fixed output grid and the native operand pipeline")
    if carry_output_store_queue and (output_store != "stmatrix" or output_grid_size is None
                                    or persistent_output_pipeline):
        raise ValueError("native store queue carry requires persistent TMA writeback")
    if automatic_optimizations and not persistent_output_pipeline:
        reuse_tmem_allocation = True
        reuse_completion_locks = True
        hoist_invariant_operand_copies = True
        carry_output_store_queue = output_store == "stmatrix" and output_grid_size is not None
    if pipeline_output_tiles or carry_output_store_queue:
        reuse_tmem_allocation = True
    validate_common_warp_schedule({"operand_move_attrs": operand_move_attrs})
    cluster_m_ctas = cta_group if cluster_m_ctas is None else cluster_m_ctas
    if type(reuse_operand_storage_for_output) is not bool:
        raise ValueError("reuse_operand_storage_for_output must be a boolean")
    if type(reuse_all_operand_storage_for_output) is not bool:
        raise ValueError("reuse_all_operand_storage_for_output must be a boolean")
    if reuse_operand_storage_for_output and reuse_all_operand_storage_for_output:
        raise ValueError("select one operand/output storage reuse mode")
    if reuse_operand_storage_for_output or reuse_all_operand_storage_for_output:
        single = (
            cta_group == 1 and cluster_m_ctas == 1 and cluster_n_ctas == 1
            and cluster_dim in (None, (1, 1, 1))
            and output_store_buffers == 1 and output_store_pending_groups == 0
        )
        paired = (
            reuse_all_operand_storage_for_output and cta_group == 2
            and cluster_m_ctas == 2 and cluster_n_ctas == 1
            and cluster_dim in (None, (2, 1, 1))
            and output_m_tile == 128 and operand_stages == 1
            and k <= operand_k_tile
            and (output_store_buffers, output_store_pending_groups) in ((1, 0), (2, 1))
        )
        if (
            not (single or paired)
            or output_m_tiles_per_cta != 1
            or output_grid_size is not None
            or output_store != "stmatrix"
            or rolled_operand_pipeline
            or persistent_output_pipeline
            or batch_output_m_operands
            or warp_specific_operand_waits
            or consumer_specific_mma_waits
        ):
            raise ValueError(
                "operand/output reuse requires a native single-job CTA or resident CTA pair"
            )
    # _schedule_matmul validates the layout; its CTA count is the physical one.
    if type(compact_mma_batch) is not bool:
        raise ValueError("compact_mma_batch must be a boolean")
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
    if persistent_output_schedule not in ("shared_ring", "paired", "pipelined", "framed", "shared"):
        raise ValueError("unknown persistent output schedule")
    snapshot_pipeline = persistent_output_pipeline and persistent_output_schedule in ("paired", "pipelined", "framed")
    native_output_pipeline = persistent_output_pipeline and persistent_output_schedule != "shared_ring"
    if persistent_output_schedule != "shared_ring" and not persistent_output_pipeline:
        raise ValueError("alternative output schedules require persistent_output_pipeline")
    if type(compute_batches_per_fragment) is not int or compute_batches_per_fragment < 0:
        raise ValueError("compute batches per fragment must be a nonnegative integer")
    if compute_batches_per_fragment and not (native_output_pipeline or grouped_compute_overlap):
        raise ValueError("compute cadence requires a native output overlap schedule")
    if native_output_pipeline and (output_grid_size is None or (rolled_operand_pipeline and not snapshot_pipeline)
                                  or batch_output_m_operands or output_m_tiles_per_cta != 1
                                  or output_n_tile > 256 or (output_store != "stmatrix"
                                      and persistent_output_schedule != "pipelined")
                                  or warp_specific_operand_waits or consumer_specific_mma_waits):
        raise ValueError("native output overlap requires a persistent native TMA schedule")
    if persistent_output_pipeline and persistent_output_schedule in ("paired", "framed") and (cta_group != 2 or output_m_tile != 128):
        raise ValueError("paired and framed schedules require an M128 CTA pair")
    if persistent_output_pipeline and persistent_output_schedule == "pipelined" and cta_group != 1:
        raise ValueError("pipelined snapshot requires a single CTA")
    if rolled_operand_pipeline and operand_stages < 2:
        raise ValueError("rolled operand pipeline requires at least two stages")
    if persistent_output_job_regions and not (
        persistent_output_pipeline
        and rolled_operand_pipeline
        and cta_group in (1, 2)
        and tmem_writeback_warps == 4
        and persistent_output_overlap
        and int(cyclic_operand_pipeline) < operand_stages
    ):
        raise ValueError(
            "job participant regions require an overlapping common rolled ring and four-warp fragments"
        )
    if type(persistent_output_frames) is not int or persistent_output_frames not in (1, 2):
        raise ValueError("persistent_output_frames must be one or two")
    if persistent_output_frames == 1 and not (persistent_output_pipeline and rolled_operand_pipeline and not persistent_output_overlap and not persistent_output_job_regions):
        raise ValueError("one frame requires a serial rolled shared ring without job regions")
    shared_ring = persistent_output_pipeline and persistent_output_schedule == "shared_ring"
    if output_writeback_window is not None and (
        not shared_ring or not persistent_output_overlap or persistent_output_job_regions
    ):
        raise ValueError("writeback windows require an overlapping common shared ring without job regions")
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
    if (reuse_persistent_lhs or reuse_persistent_rhs) and (
            k % 64 or m % (output_m_tile * output_m_tiles_per_cta)
            or n % (output_n_tile * output_n_tiles_per_cta) or rolled_operand_pipeline
            or rhs_column_major or bias or batch is not None or grouped_compute_overlap):
        raise ValueError("retained operands require complete paired output tiles and K64 panels")
    if cta_group == 2 and output_m_tile == 64 and operand_k_tile > 64 and k % 64:
        raise ValueError("paired M64 blocked TMA inputs require K divisible by 64")
    if (
        min(m, n, k, output_n_tile) <= 0
        or output_m_tile not in (64, 128, 192)
        or (
            m % output_m_tile
            and not (output_store == "stmatrix" and allow_partial_tiles)
        )
        or n % 8
        or (not allow_partial_tiles and n % output_n_tile)
    ):
        raise ValueError(
            "GEMM dimensions must be positive and divide the M64/M128/M192 and N tiles"
        )
    if output_m_tile == 192 and (
        output_n_tile > 256 or persistent_output_pipeline
        or batch_output_m_operands or output_m_tiles_per_cta != 1
        or output_n_tiles_per_cta != 1
    ):
        raise ValueError(
            "M192 composition requires a native tile of at most 256 columns"
        )
    if output_m_tile == 192 and cta_group == 2 and (
        output_slices != 1 or output_store_buffers != 1
        or tmem_writeback_register_budget * tmem_writeback_warps * 32 < 64 * output_n_tile
    ):
        raise ValueError(
            "paired M192 composition requires one output slice/buffer and a register "
            "budget large enough for the whole M64 writeback fragment"
        )
    if (
        operand_k_tile <= 0
        or (k % 16 and not (allow_partial_tiles and k % 8 == 0))
        or (not allow_partial_tiles and k % operand_k_tile)
    ):
        raise ValueError("the K tile must divide the reduction")
    # A rolled shared ring numbers banks across output jobs, so its stages may
    # outnumber one job's K batches; other schedules stage within one tile.
    carried_ring = rolled_operand_pipeline and persistent_output_pipeline
    if operand_stages < 1 or (
        not carried_ring
        and operand_stages > (k + operand_k_tile - 1) // operand_k_tile
    ):
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
    if rhs_column_major:
        name += "_bc"
    if bias:
        name += "_bias"
    def operand_strides(value, default, label):
        if value is None:
            return default
        if (not isinstance(value, (tuple, list)) or len(value) != len(default)
                or any(type(stride) is not int or stride < 0 for stride in value)):
            raise ValueError(f"{label} requires {len(default)} nonnegative integer strides")
        return tuple(value)

    lhs_default = (k, 1) if batch is None else (m * k, k, 1)
    rhs_spatial = (1, n) if rhs_column_major else (k, 1)
    rhs_default = rhs_spatial if batch is None else (n * k, *rhs_spatial)
    sa = operand_strides(lhs_strides, lhs_default, "lhs_strides")
    sb = operand_strides(rhs_strides, rhs_default, "rhs_strides")
    a_type = f"gmem<{m}x{k}x{dtype}" + (
        f" strides={list(sa[-2:])}" if lhs_strides is not None else ""
    ) + ">"
    b_type = f"gmem<{n}x{k}x{dtype}" + (
        f" strides={list(sb[-2:])}" if rhs_strides is not None
        else (f" strides=[1, {n}]" if rhs_column_major else "")
    ) + ">"
    bias_type = f"gmem<{m}x{n}x{dtype} strides=[0, 1]>"
    if bias and output_column_major:
        raise ValueError("bias currently requires row-major output")
    if bias and n % output_n_tile:
        raise ValueError("fused bias currently requires complete N tiles")
    accumulator = f"%bias: {bias_type}" if bias else "%zero: rmem<f32>"
    module = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @{name}({f'%bias: {bias_type}, ' if bias else ''}%a: {a_type}, %b: {b_type}, %c: {output_type}) {{
            %zero: rmem<f32> = constant 0.0
            matmul(%a: {a_type}, %b: {b_type}, %c: {output_type}, {accumulator}) collective=xyz.grid attrs={{label: "mm"}}
          }}
        }}
    """)
    if batch is not None:
        if bias or output_column_major or batch < 1:
            raise ValueError("native batching requires positive batch and an unbiased row-major output")
        def memory_type(rows, columns, strides, count=batch):
            return f"gmem<{count}x{rows}x{columns}x{dtype} strides={list(strides)}>"
        a3, a1 = memory_type(m, k, sa), memory_type(m, k, sa, 1)
        b3, b1 = memory_type(n, k, sb), memory_type(n, k, sb, 1)
        a2 = f"gmem<{m}x{k}x{dtype} strides={list(sa[-2:])}>"
        b2 = f"gmem<{n}x{k}x{dtype} strides={list(sb[-2:])}>"
        c3, c1 = f"gmem<{batch}x{m}x{n}x{dtype}>", f"gmem<1x{m}x{n}x{dtype}>"
        module = gpu_dialect.parse_module(f"""
          module attrs={{target: "sm_100a"}} {{
            kernel @{name}_batch{batch}(%a: {a3}, %b: {b3}, %c: {c3}) grid_dim=[None, None, {batch}] {{
              %bz: index = block_id.z
              %av: {a1} = tile(%a: {a3}, %bz: index, 0, 0)
              %a2: {a2} = reshape(%av: {a1})
              %bv: {b1} = tile(%b: {b3}, %bz: index, 0, 0)
              %b2: {b2} = reshape(%bv: {b1})
              %cv: {c1} = tile(%c: {c3}, %bz: index, 0, 0)
              %c2: {output_type} = reshape(%cv: {c1})
              %zero: rmem<f32> = constant 0.0
              matmul(%a2: {a2}, %b2: {b2}, %c2: {output_type}, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
            }}
          }}
        """)
    if type(rolled_job_unroll) is not int or rolled_job_unroll < 1:
        raise ValueError("rolled_job_unroll must be a positive integer")
    if type(peel_first_output_pair) is not bool:
        raise ValueError("peel_first_output_pair must be a boolean")
    if rolled_job_unroll > 1 or peel_first_output_pair:
        if not (rolled_operand_pipeline and persistent_output_pipeline
                and persistent_output_schedule == "shared_ring"):
            raise ValueError("output job loop transforms require the rolled shared ring")
        if rolled_job_unroll > 1:
            module.body.ops[0].attrs["rolled_job_unroll"] = rolled_job_unroll
        if peel_first_output_pair:
            module.body.ops[0].attrs["peel_first_output_pair"] = True
    if type(rolled_k_unroll) is not int or rolled_k_unroll < 1:
        raise ValueError("rolled_k_unroll must be a positive integer")
    if rolled_k_unroll > 1:
        if not (rolled_operand_pipeline and persistent_output_pipeline
                and persistent_output_schedule == "shared_ring"):
            raise ValueError("rolled_k_unroll requires the rolled shared ring")
        module.body.ops[0].attrs["rolled_k_unroll"] = rolled_k_unroll
    if type(reuse_mma_collector_a) is not bool:
        raise ValueError("reuse_mma_collector_a must be a boolean")
    if type(hoist_mma_descriptors) is not bool:
        raise ValueError("hoist_mma_descriptors must be a boolean")
    if type(preserve_prefetch_issue_order) is not bool:
        raise ValueError("preserve_prefetch_issue_order must be a boolean")
    if type(rolled_refill_before_compute) is not bool:
        raise ValueError("rolled_refill_before_compute must be a boolean")
    if rolled_refill_before_compute and not rolled_operand_pipeline:
        raise ValueError("rolled_refill_before_compute requires rolled_operand_pipeline")
    if block_gather_backend is not None:
        module.body.ops[0].attrs["block_gather_backend"] = block_gather_backend
    for attr, enabled in (
        ("reuse_completion_locks", reuse_completion_locks),
        ("ordered_mma_completion", ordered_mma_completion),
        ("interleave_mma_descriptors", interleave_mma_descriptors),
        ("hoist_mma_descriptors", hoist_mma_descriptors),
        ("preserve_prefetch_issue_order", preserve_prefetch_issue_order),
        ("rolled_refill_before_compute", rolled_refill_before_compute),
        ("reuse_mma_collector_a", reuse_mma_collector_a),
    ):
        if enabled:
            module.body.ops[0].attrs[attr] = True
    # A launch-bound grid traverses M fastest on X, so N-fast swaps the axes.
    axes = (
        (CollectiveAxis.Y, CollectiveAxis.X)
        if output_grid_size is None and output_grid_fast_axis == "n"
        else (CollectiveAxis.X, CollectiveAxis.Y)
    )
    if rolled_operand_pipeline and snapshot_pipeline:
        module.body.ops[0].attrs.update(compact_output_pipeline=True,
            compact_output_overlap=persistent_output_overlap,
            compact_output_cadence=compute_batches_per_fragment or 1)
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
        output_n_tiles_per_cta=output_n_tiles_per_cta,
        grouped_compute_overlap=grouped_compute_overlap,
        compute_batches_per_fragment=compute_batches_per_fragment,
        preserve_output_job_order=reuse_persistent_lhs or reuse_persistent_rhs,
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
        reuse_tmem_allocation=reuse_tmem_allocation or snapshot_pipeline,
        reuse_completion_locks=reuse_completion_locks or snapshot_pipeline,
        tmem_writeback_register_budget=tmem_writeback_register_budget,
        tmem_writeback_warps=tmem_writeback_warps,
    )
    if native_output_pipeline:
        if persistent_output_schedule == "framed":
            from ..passes.pipeline_frame_mma import PipelineFrameMma
            scheduled = PipelineFrameMma(overlap=persistent_output_overlap,
                compute_batches_per_fragment=compute_batches_per_fragment).run(scheduled)
        elif snapshot_pipeline:
            from ..passes.pipeline_paired_mma import PipelinePairedMma
            scheduled = PipelinePairedMma(overlap=persistent_output_overlap,
                convert_snapshot=persistent_output_schedule == "pipelined",
                compute_batches_per_fragment=compute_batches_per_fragment).run(scheduled)
        else:
            from ..passes.pipeline_tmem_writeback import PipelineTmemWriteback
            scheduled = PipelineTmemWriteback(overlap=persistent_output_overlap,
                shared_schedule=True, compute_batches_per_fragment=compute_batches_per_fragment or 1).run(scheduled)
        scheduled.body.ops[0].attrs["pack_shared_declarations"] = True
    elif shared_ring:
        from ..passes.pipeline_shared_ring import PipelineSharedRing

        if rolled_operand_pipeline and output_store_pending_groups:
            scheduled.body.ops[0].attrs["rolled_output_store_pending_groups"] = (
                output_store_pending_groups
            )
        scheduled = PipelineSharedRing(
            operand_stages=operand_stages,
            overlap=persistent_output_overlap,
            rolled_operand_pipeline=rolled_operand_pipeline,
            mma_queue_depth=max(1, int(cyclic_operand_pipeline)),
            job_regions=persistent_output_job_regions,
            output_writeback_window=output_writeback_window,
            frames=persistent_output_frames,
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
    if compact_mma_batch:
        kernel.attrs["compact_mma_batch"] = True
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
        extended_cluster or persistent_output_pipeline or batch_output_m_operands or grouped_compute_overlap
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
    if reuse_operand_storage_for_output or reuse_all_operand_storage_for_output:
        from ..passes.reuse_retired_operand_for_output import ReuseRetiredOperandForOutput

        scheduled = ReuseRetiredOperandForOutput(
            reuse_all_operands=reuse_all_operand_storage_for_output
        ).run(scheduled)
    if automatic_optimizations:
        from ..base.ir import find_ops_of_type
        from ..gpu_dialect import AllocOp, TmemTensorType

        for allocation in find_ops_of_type(scheduled, AllocOp):
            if isinstance(allocation.result().type, TmemTensorType):
                allocation.attrs["track_column_regions"] = True
    if pipeline_output_tiles or carry_output_store_queue:
        from ..passes.pipeline_output_tiles import CarryOutputStoreQueue, PipelineOutputTiles
        from ..passes.licm import LICM

        if pipeline_output_tiles:
            scheduled = PipelineOutputTiles().run(scheduled)
        if carry_output_store_queue:
            scheduled = LICM(hoist_queue_locks=True).run(scheduled)
            scheduled = LICM(hoist_queue_locks=True).run(scheduled)
            scheduled = CarryOutputStoreQueue(output_store_pending_groups or 7).run(scheduled)
    if reuse_persistent_lhs or reuse_persistent_rhs:
        from ..passes.reuse_persistent_operands import ReusePersistentOperands
        scheduled = ReusePersistentOperands("lhs" if reuse_persistent_lhs else "rhs").run(scheduled)
    if resident_phase_order_split is not None:
        from ..passes.reuse_persistent_operands import DesynchronizeOutputPhases
        scheduled = DesynchronizeOutputPhases(resident_phase_order_split).run(scheduled)
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
