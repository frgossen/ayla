#!/usr/bin/env python3
"""Benchmark supported real-model GEMMs and emit resumable CSV/JSON data.

Run the validator first (or use ``--validate-only``) to inspect exactly which
corpus shapes will be attempted.  The benchmark writes after every shape, so a
long run can be resumed safely with the same output JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import re
import statistics
import textwrap
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch

import benchmarks.common as benchmark_common

from benchmarks.common import (
    GemmBenchConfig,
    addmm_bench_config,
    assert_addmm_against_torch,
    assert_against_torch,
    assert_bmm_against_torch,
    bmm_bench_config,
    gemm_bench_config,
    out_parameter_addmm_op,
    out_parameter_bmm_op,
    out_parameter_op,
    print_results,
    run,
    torch_addmm_nt,
    torch_bmm_nt,
    torch_mm_nt,
)
from benchmarks.oss_gemm_corpus import (
    CORPUS,
    RAW_CSV,
    RAW_JSON,
    VALIDATION_CSV,
    VALIDATION_JSON,
    ScheduleConfig,
    Shape,
    architecture_gap_configs,
    continuous_architecture_gap_configs,
    benchmark_shapes,
    bmm_tuning_configs,
    cublas_inspired_configs,
    cache_hint_ablation_configs,
    cluster_tuning_configs,
    continuous_wgmma_configs,
    continuous_cluster_configs,
    continuous_toggle_ablation_configs,
    coarse_tile_configs,
    documented_gap_ablation_configs,
    exploration_configs,
    exact_cta_configs,
    exact_axis_ablation_configs,
    exact_cta_refinement_configs,
    fallback_config,
    gap_tuning_configs,
    grid_swap_configs,
    grid_swap_ablation_configs,
    grid_swap_refinement_configs,
    load_shapes,
    local_refinement_configs,
    local_ablation_configs,
    lhs_promotion_ablation_configs,
    multicast_ablation_configs,
    multicast_configs,
    multicast_incumbent_neighborhood_configs,
    multicast_resweep_configs,
    multicast_winner_transfer_configs,
    narrow_tile_configs,
    narrow_wg2d_configs,
    output_layout_ablation_configs,
    output_n_grouping_ablation_configs,
    output_slicing_ablation_configs,
    output_store_ablation_configs,
    pipeline_tile_configs,
    ptx_optimization_ablation_configs,
    regime,
    regime_name,
    refinement_configs,
    resource_tuning_configs,
    rhs_tma_slicing_configs,
    rhs_promotion_ablation_configs,
    smem_reuse_configs,
    small_cta_writeback_configs,
    small_cta_writeback_refinement_configs,
    small_cta_writeback_transfer_configs,
    stage_ablation_configs,
    seed_configs,
    static_loop_configs,
    tail_end_configs,
    targeted_tuning_configs,
    tile_refinement_configs,
    two_dimensional_wg_configs,
    workgroup_parallelism_configs,
    wgmma_pipeline_configs,
    wgmma_group_configs,
    wgmma_epilogue_configs,
    wgmma_feature_ablation_configs,
    validate_shapes,
    validation_row,
    write_json,
    write_rows_csv,
)


TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
IR_DTYPES = {
    "float16": "f16",
    "bfloat16": "bf16",
}
# BF16 keeps 8 mantissa bits, so long reductions round in larger steps.
MAX_ABS_ERROR = {
    "float16": 4,
    "bfloat16": 32,
}
# Names the configuration --jit-heuristic times through ayla.gemm_jit.
JIT_HEURISTIC = "jit_heuristic"


def _shapes_with_current_best(shapes, current_best):
    """Keep only shapes represented by an explicit ranking configuration."""
    return [shape for shape in shapes if current_best(shape) is not None]


def _baseline_config(shape: Shape) -> GemmBenchConfig:
    if shape.op == "bmm":
        assert shape.batch is not None
        return bmm_bench_config(
            name="Torch",
            batch=shape.batch,
            m=shape.m,
            n=shape.n,
            k=shape.k,
            compute=torch_bmm_nt,
            dtype=TORCH_DTYPES[shape.dtype],
        )
    builder = addmm_bench_config if shape.op == "addmm" else gemm_bench_config
    compute = torch_addmm_nt if shape.op == "addmm" else torch_mm_nt
    return builder(
        name="Torch", m=shape.m, n=shape.n, k=shape.k,
        compute=compute, dtype=TORCH_DTYPES[shape.dtype],
    )


def _fixed_output_unroll_factor(
    shape: Shape, schedule: ScheduleConfig,
) -> int | None:
    """Return output-wave unrolling, never reduction-loop unrolling."""
    if (
        schedule.output_cta_count is None
        or not schedule.unroll_output_tiles
    ):
        return None
    m, n = (
        (shape.n, shape.m) if schedule.transpose_output
        else (shape.m, shape.n)
    )
    logical_tiles = (
        math.ceil(m / schedule.output_m_tile)
        * math.ceil(n / schedule.output_n_tile)
    )
    waves = math.ceil(logical_tiles / schedule.output_cta_count)
    return waves if waves > 1 else None


_DEFAULT_CONFIG_TIMEOUT_S = 120.0
# How this run times configurations, recorded on every row it writes. Rows
# measured under different protocols are not comparable.
_TIMING_PROTOCOL = "inference-v2"


def _shape_fields(shape: Shape) -> dict[str, object]:
    return {
        "shape_id": shape.shape_id,
        "op": shape.op,
        "batch": shape.batch,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "dtype": shape.dtype,
        "layout": "canonical_nt",
        "frequency": shape.count,
        "suites": list(shape.suites),
        "regime": regime_name(shape),
        **regime(shape),
    }


def _result_row(
    shape: Shape,
    implementation: str,
    config: ScheduleConfig | None,
    *,
    latency_ms: float | None = None,
    tflops: float | None = None,
    torch_pct: float | None = None,
    max_abs_error: float | None = None,
    relative_rms_error: float | None = None,
    error: str | None = None,
    search_scope: str = "full_corpus_seed",
) -> dict[str, object]:
    config_values = (
        {"name": f"torch.{shape.op}"}
        if config is None else dataclasses.asdict(config)
    )
    return {
        **_shape_fields(shape),
        "implementation": implementation,
        "search_scope": search_scope,
        "configuration": config_values,
        "latency_ms": latency_ms,
        "tflops": tflops,
        "torch_pct": torch_pct,
        "correctness_max_abs": max_abs_error,
        "correctness_relative_rms": relative_rms_error,
        "status": "error" if error is not None else "ok",
        "error": error,
        "timing_protocol": _TIMING_PROTOCOL,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _jit_heuristic_config(shape: Shape) -> ScheduleConfig | None:
    """The schedule ayla.gemm_jit selects for ``shape``, recorded as a config."""
    from ayla.gemm_jit import select_gemm_schedule

    seed = select_gemm_schedule(
        shape.op, shape.dtype, shape.m, shape.n, shape.k, shape.batch or 1,
    )
    if seed is None:
        return None
    fields = {field.name for field in dataclasses.fields(ScheduleConfig)}
    return ScheduleConfig(name=JIT_HEURISTIC, **{
        key: value for key, value in seed.captured_config.items()
        if key in fields and key != "name"
    })


# CPU-only processes that build and assemble kernels for this process; set
# from --compile-workers. Zero leaves building to the GPU workers.
_COMPILE_WORKERS = benchmark_common.COMPILE_WORKERS


def _assemble_schedules(
    shape: Shape, schedules: list[ScheduleConfig], timeout_s: float | None,
) -> list[tuple | None]:
    """Each schedule's ``(PtxKernel, cubin)`` or ``(error type, message)``.

    Built in parallel off the GPUs, so a GPU worker only loads and times.
    The JIT-routed configuration gets None: it times the library's own
    compiler, in its GPU worker.
    """
    entries: list[tuple | None] = [None] * len(schedules)
    if not _COMPILE_WORKERS:
        return entries
    indices = [i for i, s in enumerate(schedules) if s.name != JIT_HEURISTIC]
    assembled = benchmark_common.assemble_kernels(
        _build_ptx_kernel, [(shape, schedules[i]) for i in indices],
        _COMPILE_WORKERS, timeout_s,
    )
    for i, entry in zip(indices, assembled, strict=True):
        entries[i] = entry
    return entries


def _make_ayla_config(
    shape: Shape, schedule: ScheduleConfig, prebuilt: tuple | None = None,
) -> tuple[GemmBenchConfig, float, float]:
    """Load one configuration's kernel and check it against Torch.

    ``prebuilt`` is the ``(PtxKernel, cubin)`` :func:`_assemble_schedules`
    made off the GPU; without it the kernel is built and JIT-compiled here.
    """
    if schedule.name == JIT_HEURISTIC:
        from ayla.gemm_jit import gemm_op

        # Time the library's own routing rather than rebuilding the schedule.
        return _bench_ayla_launcher(
            shape, schedule, gemm_op(shape.op, shape.dtype),
        )
    from ayla.ptx_runtime import make_ptx_launcher

    kernel, image = (
        (_build_ptx_kernel(shape, schedule), None)
        if prebuilt is None else prebuilt
    )
    return _bench_ayla_launcher(
        shape, schedule, make_ptx_launcher(kernel, image),
    )


def _build_ptx_kernel(shape: Shape, schedule: ScheduleConfig):
    """Schedule one configuration and lower it to PTX; needs no GPU."""
    # Keep Ayla's eagerly registered collection out of the CPU-only validation
    # path. Importing this submodule compiles the collection as a side effect.
    from ayla.gemm_collection.gemm import (
        auto_matmul_4096_wgmma256_wg_tma_store_module,
        schedule_matmul,
    )
    from ayla.base.ir import find_ancestor_of_type, find_ops_of_type
    from ayla.gpu_dialect import (
        CollectiveAxis,
        ForOp,
        gpu_dialect,
    )
    from ayla.passes.canonicalize import Canonicalize
    from ayla.passes.licm import LICM
    from ayla.passes.schedule_async import ScheduleAsync
    from ayla.passes.unroll_loop import UnrollLoop
    from ayla.ptx_codegen import make_ptx_kernel

    if schedule.unsafe_reuse_smem:
        raise ValueError("unsafe SMEM reuse is excluded from this search")
    if schedule.rhs_tma_load_slices not in (None, 1):
        raise ValueError(
            "RHS TMA slicing remains experimental and is not a seedable knob"
        )

    kwargs = schedule.builder_kwargs()
    transpose_output = bool(kwargs.pop("transpose_output"))
    # Historical searches recorded these experimental controls, but the
    # simplified queued-pipeline implementation deliberately has no separate
    # WGMMA feature knobs.  Keep accepting old result rows while compiling
    # their effective schedule through the public GEMM API.
    kwargs.pop("wgmma_group_size")
    kwargs.pop("wgmma_pending_groups")
    kwargs.pop("wgmma_epilogue_pipeline")
    if transpose_output and shape.op == "addmm":
        raise ValueError("transposed output is not implemented for addmm")
    axes = kwargs.pop("output_grid_axes")
    assert isinstance(axes, tuple)
    kwargs["output_grid_axes"] = tuple(
        CollectiveAxis[axis] for axis in axes
    )
    dtype = IR_DTYPES[shape.dtype]

    ptx_attrs = (
        ""
        if schedule.ptx_optimization_level is None
        else " attrs={ptx_optimization_level: "
        f"{schedule.ptx_optimization_level}}}"
    )

    def addmm_module():
        return gpu_dialect.parse(textwrap.dedent(f"""\
            module {{
              kernel @addmm(%bias: gmem<{shape.m}x{shape.n}x{dtype} strides=[0, 1]>, %a: gmem<{shape.m}x{shape.k}x{dtype}>, %b: gmem<{shape.n}x{shape.k}x{dtype}>, %c: gmem<{shape.m}x{shape.n}x{dtype}>){ptx_attrs} {{
                matmul(%a: gmem<{shape.m}x{shape.k}x{dtype}>, %b: gmem<{shape.n}x{shape.k}x{dtype}>, %c: gmem<{shape.m}x{shape.n}x{dtype}>, %bias: gmem<{shape.m}x{shape.n}x{dtype} strides=[0, 1]>) collective=xyz.grid attrs={{label: "mm"}}
              }}
            }}"""))

    def bmm_module():
        assert shape.batch is not None
        batch = shape.batch
        if transpose_output:
            m, n, k = shape.m, shape.n, shape.k
            c3 = f"gmem<1x{n}x{m}x{dtype} strides=[{m * n}, 1, {n}]>"
            c2 = f"gmem<{n}x{m}x{dtype} strides=[1, {n}]>"
            return gpu_dialect.parse(textwrap.dedent(f"""\
                module {{
                  kernel @bmm(%a: gmem<{batch}x{m}x{k}x{dtype}>, %bt: gmem<{batch}x{n}x{k}x{dtype}>, %c: gmem<{batch}x{n}x{m}x{dtype} strides=[{m * n}, 1, {n}]>) grid_dim=[None, None, {batch}]{ptx_attrs} {{
                    %bx: index, %by: index, %bz: index = block_ids()
                    %a3: gmem<1x{m}x{k}x{dtype}> = tile(%a: gmem<{batch}x{m}x{k}x{dtype}>, %bz: index, 0, 0)
                    %a2: gmem<{m}x{k}x{dtype}> = reshape(%a3: gmem<1x{m}x{k}x{dtype}>)
                    %bt3: gmem<1x{n}x{k}x{dtype}> = tile(%bt: gmem<{batch}x{n}x{k}x{dtype}>, %bz: index, 0, 0)
                    %bt2: gmem<{n}x{k}x{dtype}> = reshape(%bt3: gmem<1x{n}x{k}x{dtype}>)
                    %c3: {c3} = tile(%c: gmem<{batch}x{n}x{m}x{dtype} strides=[{m * n}, 1, {n}]>, %bz: index, 0, 0)
                    %c2: {c2} = reshape(%c3: {c3})
                    matmul(%bt2: gmem<{n}x{k}x{dtype}>, %a2: gmem<{m}x{k}x{dtype}>, %c2: {c2}) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
                  }}
                }}"""))
        return gpu_dialect.parse(textwrap.dedent(f"""\
            module {{
              kernel @bmm(%a: gmem<{batch}x{shape.m}x{shape.k}x{dtype}>, %bt: gmem<{batch}x{shape.n}x{shape.k}x{dtype}>, %c: gmem<{batch}x{shape.m}x{shape.n}x{dtype}>) grid_dim=[None, None, {batch}]{ptx_attrs} {{
                %bx: index, %by: index, %bz: index = block_ids()
                %a3: gmem<1x{shape.m}x{shape.k}x{dtype}> = tile(%a: gmem<{batch}x{shape.m}x{shape.k}x{dtype}>, %bz: index, 0, 0)
                %a2: gmem<{shape.m}x{shape.k}x{dtype}> = reshape(%a3: gmem<1x{shape.m}x{shape.k}x{dtype}>)
                %bt3: gmem<1x{shape.n}x{shape.k}x{dtype}> = tile(%bt: gmem<{batch}x{shape.n}x{shape.k}x{dtype}>, %bz: index, 0, 0)
                %bt2: gmem<{shape.n}x{shape.k}x{dtype}> = reshape(%bt3: gmem<1x{shape.n}x{shape.k}x{dtype}>)
                %c3: gmem<1x{shape.m}x{shape.n}x{dtype}> = tile(%c: gmem<{batch}x{shape.m}x{shape.n}x{dtype}>, %bz: index, 0, 0)
                %c2: gmem<{shape.m}x{shape.n}x{dtype}> = reshape(%c3: gmem<1x{shape.m}x{shape.n}x{dtype}>)
                matmul(%a2: gmem<{shape.m}x{shape.k}x{dtype}>, %bt2: gmem<{shape.n}x{shape.k}x{dtype}>, %c2: gmem<{shape.m}x{shape.n}x{dtype}>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
              }}
            }}"""))

    def mm_module():
        a_type = f"gmem<{shape.m}x{shape.k}x{dtype}>"
        b_type = f"gmem<{shape.n}x{shape.k}x{dtype}>"
        if transpose_output:
            c_type = (
                f"gmem<{shape.n}x{shape.m}x{dtype} strides=[1, {shape.n}]>"
            )
            operands = f"%b: {b_type}, %a: {a_type}, %c: {c_type}"
        else:
            c_type = f"gmem<{shape.m}x{shape.n}x{dtype}>"
            operands = f"%a: {a_type}, %b: {b_type}, %c: {c_type}"
        return gpu_dialect.parse(textwrap.dedent(f"""\
            module {{
              kernel @mm(%a: {a_type}, %b: {b_type}, %c: {c_type}) attrs={{ptx_optimization_level: {schedule.ptx_optimization_level}}} {{
                matmul({operands}) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
              }}
            }}"""))

    if (
        shape.op in ("addmm", "bmm")
        or transpose_output
        or schedule.output_cta_count is not None
    ):
        if shape.op == "bmm" and schedule.output_cta_count is not None:
            raise ValueError(
                "fixed output grids are not part of the initial BMM search"
            )
        schedule_kwargs = dict(kwargs)
        schedule_kwargs.pop("ptx_optimization_level", None)
        if schedule.output_cta_count is not None:
            schedule_kwargs["output_grid_size"] = schedule.output_cta_count
        module = schedule_matmul(
            (
                addmm_module()
                if shape.op == "addmm"
                else bmm_module()
                if shape.op == "bmm"
                else mm_module()
            ),
            **schedule_kwargs,
        )
        output_unroll_factor = _fixed_output_unroll_factor(shape, schedule)
        if output_unroll_factor is not None:
            module = LICM().run(module)
            module = LICM().run(module)
            outer_loops = [
                loop for loop in find_ops_of_type(module, ForOp)
                if find_ancestor_of_type(loop, ForOp) is None
            ]
            if len(outer_loops) != 1:
                raise ValueError(
                    "fixed output-grid unrolling requires exactly one outer "
                    "tile loop"
                )
            output_loop = outer_loops[0]
            output_loop.attrs["label"] = "outmn"
            module = UnrollLoop(
                unroll_factor=output_unroll_factor,
                label_filter="outmn",
            ).run(module)
            module = Canonicalize().run(module)
    else:
        module = auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=shape.m,
            n=shape.n,
            k=shape.k,
            dtype=dtype,
            **kwargs,
        )

    [kernel] = module.body.ops
    if schedule.cluster_m_ctas > 1 or schedule.cluster_n_ctas > 1:
        if schedule.output_cta_count is not None:
            cluster_axis = (
                "m" if schedule.cluster_m_ctas > 1 else "n"
            )
            kernel.attrs["fixed_grid_fast_axis"] = (
                schedule.output_tile_fast_axis
            )
            kernel.attrs["cluster_tile_axis"] = cluster_axis
            cluster = max(schedule.cluster_m_ctas, schedule.cluster_n_ctas)
            kernel.attrs["cluster_dim"] = [cluster, 1, 1]
        elif schedule.swap_output_grid_axes:
            kernel.attrs["cluster_dim"] = [
                schedule.cluster_n_ctas, schedule.cluster_m_ctas, 1,
            ]
        else:
            kernel.attrs["cluster_dim"] = [
                schedule.cluster_m_ctas, schedule.cluster_n_ctas, 1,
            ]
    if schedule.reschedule_async:
        module = ScheduleAsync().run(module)
        module = Canonicalize().run(module)
    return make_ptx_kernel(module)


def _bench_ayla_launcher(
    shape: Shape, schedule: ScheduleConfig, launcher: Any,
) -> tuple[GemmBenchConfig, float, float]:
    """Wrap ``launcher`` as a benchmark config after checking it against Torch."""
    if shape.op == "bmm":
        assert shape.batch is not None
        bench = bmm_bench_config(
            name=f"Ayla {schedule.name}",
            batch=shape.batch,
            m=shape.m,
            n=shape.n,
            k=shape.k,
            compute=out_parameter_bmm_op(launcher),
            dtype=TORCH_DTYPES[shape.dtype],
        )
        max_abs, relative_rms = assert_bmm_against_torch(
            bench,
            batch=shape.batch,
            m=shape.m,
            seed=int(shape.shape_id[-8:], 16),
            max_abs_error=MAX_ABS_ERROR[shape.dtype],
            max_relative_rms_error=2e-3,
        )
    else:
        compute = (
            out_parameter_addmm_op(launcher)
            if shape.op == "addmm" else out_parameter_op(launcher)
        )
        bench_builder = (
            addmm_bench_config if shape.op == "addmm" else gemm_bench_config
        )
        bench = bench_builder(
            name=f"Ayla {schedule.name}",
            m=shape.m, n=shape.n, k=shape.k,
            compute=compute, dtype=TORCH_DTYPES[shape.dtype],
        )
        checker = (
            assert_addmm_against_torch
            if shape.op == "addmm" else assert_against_torch
        )
        max_abs, relative_rms = checker(
            bench,
            seed=int(shape.shape_id[-8:], 16),
            max_abs_error=MAX_ABS_ERROR[shape.dtype],
            max_relative_rms_error=2e-3,
        )
    return bench, max_abs, relative_rms


def _isolated_benchmark_worker(
    shape: Shape,
    schedules: tuple[ScheduleConfig, ...],
    include_baseline: bool,
    timing: dict[str, Any],
    sender: Connection,
    max_timing_rounds: int,
    prebuilt: dict[str, tuple],
) -> None:
    """Build and time configs in a process the parent can safely terminate."""
    # Spawn starts a fresh interpreter, so the parent's timing cap is not
    # inherited through benchmark_common's module globals.
    benchmark_common._MAX_ROUNDS = max_timing_rounds
    try:
        for schedule in schedules:
            sender.send(("started", schedule.name))
            try:
                bench, max_abs, relative_rms = _make_ayla_config(
                    shape, schedule, prebuilt.get(schedule.name),
                )
                baseline = _baseline_config(shape)
                torch_result, result = run([baseline, bench], **timing)
            except Exception as error:
                sender.send(("error", schedule.name, (
                    type(error).__name__, str(error),
                )))
            else:
                if include_baseline:
                    sender.send(("baseline", {
                        "latency_ms": torch_result.median_ms,
                        "tflops": torch_result.tflops,
                    }))
                    include_baseline = False
                sender.send(("result", schedule.name, {
                    "latency_ms": result.median_ms,
                    "tflops": result.tflops,
                    "torch_pct": result.torch_mm_pct,
                    "max_abs": max_abs,
                    "relative_rms": relative_rms,
                }))
                del baseline, bench
            torch.cuda.empty_cache()
        sender.send(("done", None))
    except BaseException as error:
        # Include failures outside an individual configuration (for example a
        # CUDA initialization failure) so the parent need not infer an EOF.
        try:
            sender.send(("fatal", type(error).__name__, str(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        sender.close()


def _stop_worker(process: Any) -> None:
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _benchmark_isolated(
    shape: Shape,
    schedules: list[ScheduleConfig],
    timeout_s: float,
    timing: dict[str, Any],
) -> tuple[
    dict[str, float] | None,
    list[tuple[ScheduleConfig, dict[str, float] | None, str | None]],
]:
    """Benchmark schedules with a restartable per-configuration watchdog."""
    context = mp.get_context("spawn")
    baseline: dict[str, float] | None = None
    measurements: list[
        tuple[ScheduleConfig, dict[str, float] | None, str | None]
    ] = []
    # Configurations that fail to build or assemble never reach a GPU worker.
    prebuilt: dict[str, tuple] = {}
    buildable = []
    for schedule, entry in zip(
        schedules, _assemble_schedules(shape, schedules, timeout_s), strict=True,
    ):
        if entry is not None and isinstance(entry[1], str):
            error = f"{entry[0]}: {entry[1]}"
            print(f"  reject {schedule.name}: {error}", flush=True)
            measurements.append((schedule, None, error))
            continue
        if entry is not None:
            prebuilt[schedule.name] = entry
        buildable.append(schedule)
    schedules = buildable
    position = 0

    while position < len(schedules):
        receiver, sender = context.Pipe(duplex=False)
        remaining = tuple(schedules[position:])
        process = context.Process(
            target=_isolated_benchmark_worker,
            args=(
                shape, remaining, baseline is None, timing, sender,
                benchmark_common._MAX_ROUNDS,
                {s.name: prebuilt[s.name] for s in remaining if s.name in prebuilt},
            ),
        )
        process.start()
        sender.close()
        current: ScheduleConfig | None = None
        deadline = time.monotonic() + timeout_s
        restart = False

        try:
            while True:
                wait_s = max(0.0, deadline - time.monotonic())
                if not receiver.poll(wait_s):
                    if current is None:
                        raise RuntimeError(
                            f"Torch baseline exceeded {timeout_s:g}s timeout"
                        )
                    error = f"compile_timeout: exceeded {timeout_s:g}s"
                    print(f"  timeout {current.name}: {error}", flush=True)
                    measurements.append((current, None, error))
                    position += 1
                    restart = True
                    break

                try:
                    message = receiver.recv()
                except EOFError:
                    if current is None:
                        raise RuntimeError(
                            f"benchmark worker exited with code {process.exitcode}"
                        )
                    error = f"worker_exit: code {process.exitcode}"
                    print(f"  reject {current.name}: {error}", flush=True)
                    measurements.append((current, None, error))
                    position += 1
                    restart = True
                    break

                kind = message[0]
                if kind == "baseline":
                    baseline = message[1]
                    deadline = time.monotonic() + timeout_s
                elif kind == "started":
                    current = schedules[position]
                    if message[1] != current.name:
                        raise RuntimeError("benchmark worker schedule mismatch")
                    deadline = time.monotonic() + timeout_s
                elif kind == "result":
                    if current is None or message[1] != current.name:
                        raise RuntimeError("benchmark worker result mismatch")
                    measurements.append((current, message[2], None))
                    print(
                        f"  measured {current.name}: "
                        f"{message[2]['latency_ms']:.4f}ms",
                        flush=True,
                    )
                    position += 1
                    current = None
                    deadline = time.monotonic() + timeout_s
                elif kind == "error":
                    if current is None or message[1] != current.name:
                        raise RuntimeError("benchmark worker error mismatch")
                    error = f"{message[2][0]}: {message[2][1]}"
                    measurements.append((current, None, error))
                    print(f"  reject {current.name}: {error}", flush=True)
                    position += 1
                    # A failed CUDA launch can poison the process context, so
                    # even apparently recoverable configuration errors must
                    # continue in a fresh worker. This also prevents the
                    # worker's cleanup call from escalating an async illegal
                    # access into a fatal error for the whole shard.
                    restart = True
                    break
                elif kind == "done":
                    break
                elif kind == "fatal":
                    raise RuntimeError(f"worker {message[1]}: {message[2]}")
                else:
                    raise RuntimeError(f"unknown worker message: {kind!r}")
        finally:
            if restart or process.is_alive():
                _stop_worker(process)
            else:
                process.join(timeout=5)
            receiver.close()

        if not restart:
            break

    return baseline, measurements


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list")
    return value


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    config = row.get("configuration", {})
    name = config.get("name", "") if isinstance(config, dict) else ""
    return str(row.get("shape_id", "")), str(name)


def _schedule_from_row(row: dict[str, Any]) -> ScheduleConfig:
    config = row.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Ayla result row has no configuration mapping")
    values: dict[str, object] = {}
    for field in dataclasses.fields(ScheduleConfig):
        if field.name in config:
            values[field.name] = config[field.name]
        elif field.default is not dataclasses.MISSING:
            values[field.name] = field.default
        else:
            raise ValueError(f"configuration is missing {field.name!r}")
    return ScheduleConfig(**values)  # type: ignore[arg-type]


def _schedule_key(schedule: ScheduleConfig) -> tuple[object, ...]:
    return tuple(
        getattr(schedule, field.name)
        for field in dataclasses.fields(ScheduleConfig)
        if field.name != "name"
    )


def _load_explicit_candidates(
    path: Path,
) -> dict[str, tuple[ScheduleConfig, ...]]:
    """Load exact shape/config pairs from an analysis candidate manifest."""
    value = json.loads(path.read_text())
    entries = value.get("transfers") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise ValueError(
            f"{path}: expected a list or an object with a 'transfers' list"
        )
    by_shape: dict[str, dict[tuple[object, ...], ScheduleConfig]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: candidate {index} is not an object")
        shape_id = entry.get("target", entry.get("shape_id"))
        config = entry.get("configuration", entry.get("config"))
        if not isinstance(shape_id, str) or not isinstance(config, dict):
            raise ValueError(
                f"{path}: candidate {index} needs target/shape_id and "
                "configuration/config"
            )
        schedule = _schedule_from_row({"configuration": config})
        key = _schedule_key(schedule)
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
        schedule = dataclasses.replace(
            schedule, name=f"documented_{index:04d}_{digest}",
        )
        by_shape.setdefault(shape_id, {}).setdefault(key, schedule)
    return {
        shape_id: tuple(configs.values())
        for shape_id, configs in by_shape.items()
    }


def _adapt_transfer_schedule(
    shape: Shape, base: ScheduleConfig,
) -> ScheduleConfig | None:
    """Make a neighboring shape's winner legal for this static shape."""
    if (
        shape.m < base.output_m_tile
        or shape.n < base.output_n_tile
        or shape.k % base.operand_k_tile
        or base.output_m_tile % (base.warpgroup_m_tile or base.output_m_tile)
        or base.output_n_tile % base.warpgroup_n_tile
    ):
        return None

    m_tiles = (shape.m + base.output_m_tile - 1) // base.output_m_tile
    n_tiles = (shape.n + base.output_n_tile - 1) // base.output_n_tile
    logical_tiles = m_tiles * n_tiles
    if base.output_cta_count is None:
        if m_tiles < base.output_tiles_per_cta:
            return None
        candidate = base
    elif not base.unroll_output_tiles:
        if logical_tiles < base.output_cta_count:
            return None
        candidate = base
    else:
        divisors = [
            count for count in range(64, 265)
            if logical_tiles % count == 0
            and logical_tiles // count <= 8
            and not (shape.k > 2048 and logical_tiles // count > 2)
        ]
        if not divisors:
            return None
        cta_count = min(
            divisors,
            key=lambda count: abs(count - int(base.output_cta_count or count)),
        )
        candidate = dataclasses.replace(base, output_cta_count=cta_count)

    if (
        m_tiles % candidate.cluster_m_ctas
        or n_tiles % candidate.cluster_n_ctas
    ):
        return None
    estimated_smem = (
        2
        * (candidate.output_m_tile + candidate.output_n_tile)
        * candidate.operand_k_tile
        * candidate.operand_stages
        + 2
        * candidate.output_m_tile
        * candidate.output_n_tile
        * candidate.output_store_buffers
    )
    if not candidate.unsafe_reuse_smem and estimated_smem > 225 * 1024:
        return None

    key = _schedule_key(candidate)
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
    return dataclasses.replace(
        candidate,
        name=(
            f"xfer_m{candidate.output_m_tile}_n{candidate.output_n_tile}_"
            f"k{candidate.operand_k_tile}_s{candidate.operand_stages}_"
            f"c{candidate.output_cta_count or 0}_{digest}"
        ),
    )


def transfer_configs(
    shape: Shape,
    rows: list[dict[str, Any]],
    *,
    limit: int = 24,
) -> tuple[ScheduleConfig, ...]:
    """Transfer measured winners from nearby shapes without repeating work."""
    torch_by_shape = {
        str(row["shape_id"]): row for row in rows
        if row.get("implementation") == "torch"
        and row.get("status") == "ok"
        and row.get("latency_ms") is not None
    }
    ayla_by_shape: dict[str, list[dict[str, Any]]] = {}
    measured_here: set[tuple[object, ...]] = set()
    for row in rows:
        if row.get("implementation") != "ayla":
            continue
        try:
            schedule = _schedule_from_row(row)
        except (TypeError, ValueError):
            continue
        if str(row.get("shape_id")) == shape.shape_id:
            measured_here.add(_schedule_key(schedule))
        if row.get("status") == "ok" and row.get("latency_ms") is not None:
            ayla_by_shape.setdefault(str(row["shape_id"]), []).append(row)

    ranked: list[tuple[float, dict[str, Any]]] = []
    target_aspect = shape.m / shape.n
    for shape_id, candidates in ayla_by_shape.items():
        if shape_id == shape.shape_id:
            continue
        baseline = torch_by_shape.get(shape_id)
        if baseline is None or str(baseline.get("dtype")) != shape.dtype:
            continue
        def candidate_pct(row: dict[str, Any]) -> float:
            value = row.get("torch_pct")
            return (
                float(value) if value is not None
                else float(baseline["latency_ms"]) / float(row["latency_ms"]) * 100
            )

        best = max(candidates, key=candidate_pct)
        m, n, k = (int(best[field]) for field in ("m", "n", "k"))
        pct = candidate_pct(best)
        distance = (
            abs(math.log2(m / shape.m))
            + abs(math.log2(n / shape.n))
            + 0.7 * abs(math.log2(k / shape.k))
            + 0.5 * abs(math.log2((m / n) / target_aspect))
            + max(0.0, 98.0 - pct) / 20.0
        )
        ranked.append((distance, best))
    ranked.sort(key=lambda item: item[0])

    selected: dict[tuple[object, ...], ScheduleConfig] = {}
    for _, row in ranked:
        candidate = _adapt_transfer_schedule(shape, _schedule_from_row(row))
        if candidate is None:
            continue
        key = _schedule_key(candidate)
        if key in measured_here:
            continue
        selected.setdefault(key, candidate)
        if len(selected) == limit:
            break
    return tuple(selected.values())


def _write_raw(rows: list[dict[str, object]], json_path: Path, csv_path: Path) -> None:
    write_json(rows, json_path)
    if rows:
        write_rows_csv(rows, csv_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--validation-csv", type=Path, default=VALIDATION_CSV)
    parser.add_argument("--validation-json", type=Path, default=VALIDATION_JSON)
    parser.add_argument(
        "--skip-validation-write", action="store_true",
        help="reuse validation artifacts already produced by the validator",
    )
    parser.add_argument("--output-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--output-json", type=Path, default=RAW_JSON)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--jit-heuristic", action="store_true",
        help="time only the schedule ayla.gemm_jit selects for each shape, "
        "through the library's JIT op",
    )
    parser.add_argument(
        "--workload", choices=("mm", "addmm", "bmm", "all"), default="mm",
        help="benchmark mm, addmm, bmm, or all accepted workload identities",
    )
    parser.add_argument(
        "--shape-filter", help="regular expression matched against shape_id",
    )
    parser.add_argument(
        "--config-filter", help="regular expression matched against config name",
    )
    parser.add_argument(
        "--max-shapes", type=int,
        help="cap supported shapes after frequency sorting (for smoke runs)",
    )
    parser.add_argument(
        "--config-timeout-seconds", type=float,
        default=_DEFAULT_CONFIG_TIMEOUT_S,
        help="terminate and skip a config whose isolated build, correctness, "
        "and timing exceed this many seconds; 0 disables isolation",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="replace existing raw output instead of skipping completed shapes",
    )
    parser.add_argument(
        "--pair-each-config", action="store_true",
        help="time every candidate in its own interleaved Torch/Ayla pair",
    )
    parser.add_argument(
        "--max-timing-rounds", type=int, default=20,
        help=(
            "cap rotated CUDA-event samples per benchmark configuration; "
            "twenty already measure far more precisely than a shape's "
            "result carries over to a model's forward graph"
        ),
    )
    parser.add_argument(
        "--compile-workers", type=int,
        default=benchmark_common.COMPILE_WORKERS,
        help=(
            "CPU-only processes that build and assemble kernels ahead of "
            "the GPU worker, in total: concurrent shards share them out; "
            "0 builds and JIT-compiles in the GPU worker"
        ),
    )
    parser.add_argument(
        "--timing", choices=("inference", "cold", "warm"), default="inference",
        help=(
            "inference: time each GEMM as a model layer (hot activations, "
            "cold per-layer weights, a consumer of its output); cold: one "
            "launch after resetting L2 (needs --cold-cache-mib); warm: "
            "back-to-back launches on the same inputs"
        ),
    )
    parser.add_argument(
        "--cold-cache-mib", type=int, default=0,
        help=(
            "MiB of L2 pollution before each sample; inference timing uses at "
            "least twice the L2 size"
        ),
    )
    parser.add_argument(
        "--explore-knobs", action="store_true",
        help="run extra knob probes on the most frequent shape in each regime",
    )
    parser.add_argument(
        "--fallback-only", action="store_true",
        help="run only the conservative scalar-store coverage fallback",
    )
    parser.add_argument(
        "--refine-only", action="store_true",
        help="run only the bounded scalar N=64 refinement configurations",
    )
    parser.add_argument(
        "--tile-refine-only", action="store_true",
        help="run only the bounded M=64/128, N=128 tile refinement",
    )
    parser.add_argument(
        "--tune-only", action="store_true",
        help="run the focused tile/mainloop sweep for difficult shapes",
    )
    parser.add_argument(
        "--cublas-inspired-only", action="store_true",
        help="run configurations modeled on cuBLAS launch/SASS geometry",
    )
    parser.add_argument(
        "--resource-tune-only", action="store_true",
        help="sweep safe tile shapes and pipeline depth within the SMEM budget",
    )
    parser.add_argument(
        "--coarse-tile-only", action="store_true",
        help="sweep unmeasured output tile shapes and pipeline depths",
    )
    parser.add_argument(
        "--wg2d-only", action="store_true",
        help="sweep linearized two-dimensional workgroup tiling",
    )
    parser.add_argument(
        "--multicast-only", action="store_true",
        help="sweep cluster-2/4 operand multicast",
    )
    parser.add_argument(
        "--multicast-ablation-only", action="store_true",
        help="pair diverse cluster-2/4 multicast schedules with exact local controls",
    )
    parser.add_argument(
        "--multicast-resweep-only", action="store_true",
        help="pair incumbent/diverse tiles with cluster-2/4/8 multicast",
    )
    parser.add_argument(
        "--multicast-transfer-only", action="store_true",
        help="transfer confirmed small-K local and multicast winners",
    )
    parser.add_argument(
        "--multicast-neighborhood-only", action="store_true",
        help="co-tune clusters with each incumbent scheduling neighborhood",
    )
    parser.add_argument(
        "--pipeline-tile-only", action="store_true",
        help="refine winning tile widths with K64/K128 pipeline depths",
    )
    parser.add_argument(
        "--gap-tune-only", action="store_true",
        help="sweep missing narrow tiles, WG2D depths, and output slicing",
    )
    parser.add_argument(
        "--narrow-tile-only", action="store_true",
        help="refine N=48/56/72 tiles across M tiles and K pipelines",
    )
    parser.add_argument(
        "--bmm-tune-only", action="store_true",
        help="search compact batch-aware geometry, staging, and cache choices",
    )
    parser.add_argument(
        "--exact-cta-only", action="store_true",
        help="sweep balanced flattened output assignment at fixed CTA counts",
    )
    parser.add_argument(
        "--exact-cta-refine-only", action="store_true",
        help="refine exact-grid winners with stages and output slicing",
    )
    parser.add_argument(
        "--local-refine-only", action="store_true",
        help="search a bounded neighborhood around each shape's current best",
    )
    parser.add_argument(
        "--cluster-tune-only", action="store_true",
        help="combine narrow output tiles with cluster operand multicast",
    )
    parser.add_argument(
        "--narrow-wg2d-only", action="store_true",
        help="sweep homogeneous 2-D workgroups on narrow N tiles",
    )
    parser.add_argument(
        "--transfer-only", action="store_true",
        help="try unmeasured winners transferred from neighboring shapes",
    )
    parser.add_argument(
        "--static-loop-only", action="store_true",
        help="sweep compact near-one-wave static grid-stride schedules",
    )
    parser.add_argument(
        "--small-cta-writeback-only", action="store_true",
        help="sweep resident one-WG CTAs with fine-grained TMA writeback",
    )
    parser.add_argument(
        "--small-cta-writeback-refine-only", action="store_true",
        help="refine the M128/N128/split-8 one-WG schedule",
    )
    parser.add_argument(
        "--small-cta-writeback-transfer-only", action="store_true",
        help="transfer refined one-WG writeback schedules across tail shapes",
    )
    parser.add_argument(
        "--architecture-gap-only", action="store_true",
        help="sweep deep pipelines, 64x80 tiles, and wide clustered tiles",
    )
    parser.add_argument(
        "--tail-end-only", action="store_true",
        help="target the remaining deep-K, extreme-tall, and extreme-wide tail",
    )
    parser.add_argument(
        "--recheck-best-only", action="store_true",
        help="remeasure each shape's current winner with the current compiler",
    )
    parser.add_argument(
        "--rhs-tma-slice-only", action="store_true",
        help="vary RHS TMA load slicing around each shape's current winner",
    )
    parser.add_argument(
        "--workgroup-parallel-only", action="store_true",
        help="partition each winner's CTA tile over more compute warpgroups",
    )
    parser.add_argument(
        "--smem-reuse-only", action="store_true",
        help="reuse operand/output shared memory around eligible winners",
    )
    parser.add_argument(
        "--wgmma-pipeline-only", action="store_true",
        help="pipeline WGMMA commit groups around each current winner",
    )
    parser.add_argument(
        "--continuous-wgmma-only", action="store_true",
        help="pipeline one WGMMA group safely across K-loop backedges",
    )
    parser.add_argument(
        "--wgmma-group-only", action="store_true",
        help="split K128 into four-instruction WGMMA commit groups",
    )
    parser.add_argument(
        "--wgmma-epilogue-only", action="store_true",
        help="pipeline only refill-free straight-line WGMMA epilogues",
    )
    parser.add_argument(
        "--continuous-cluster-only", action="store_true",
        help="combine continuous WGMMA overlap with multicast clusters",
    )
    parser.add_argument(
        "--continuous-architecture-only", action="store_true",
        help="combine continuous overlap with deep/parallel geometries",
    )
    parser.add_argument(
        "--grid-swap-only", action="store_true",
        help="swap ordinary M/N block axes around each current winner",
    )
    parser.add_argument(
        "--grid-swap-refine-only", action="store_true",
        help="refine tile geometry only around swapped-grid winners",
    )
    parser.add_argument(
        "--grid-swap-ablation-only", action="store_true",
        help="time both ordinary block-axis orders against one baseline",
    )
    parser.add_argument(
        "--exact-axis-ablation-only", action="store_true",
        help="time both exact-grid traversal orders against one baseline",
    )
    parser.add_argument(
        "--lhs-promotion-ablation-only", action="store_true",
        help="time all LHS TMA L2 promotion sizes against one baseline",
    )
    parser.add_argument(
        "--rhs-promotion-ablation-only", action="store_true",
        help="time all RHS TMA L2 promotion sizes against one baseline",
    )
    parser.add_argument(
        "--cache-hint-ablation-only", action="store_true",
        help="time all L2 eviction-hint combinations against one baseline",
    )
    parser.add_argument(
        "--stage-ablation-only", action="store_true",
        help="time nearby operand-pipeline depths against one baseline",
    )
    parser.add_argument(
        "--ptx-optimization-ablation-only", action="store_true",
        help="time PTX optimization levels 0 through 3 against one baseline",
    )
    parser.add_argument(
        "--output-layout-ablation-only", action="store_true",
        help="time linear and block-major TMA output layouts together",
    )
    parser.add_argument(
        "--output-slicing-ablation-only", action="store_true",
        help="time safe TMA writeback slice and buffer counts together",
    )
    parser.add_argument(
        "--output-n-grouping-ablation-only", action="store_true",
        help="time native N-tile grouping with ordinary/cyclic K scheduling",
    )
    parser.add_argument(
        "--continuous-toggle-ablation-only", action="store_true",
        help="time continuous and ordinary K scheduling at fixed geometry",
    )
    parser.add_argument(
        "--wgmma-feature-ablation-only", action="store_true",
        help="time selected WGMMA pipelining against the same schedule off",
    )
    parser.add_argument(
        "--output-store-ablation-only", action="store_true",
        help="time scalar and STMatrix/TMA output writeback together",
    )
    parser.add_argument(
        "--local-ablation-only", action="store_true",
        help="time local refinements together with the exact current base",
    )
    parser.add_argument(
        "--documented-gap-ablation-only", action="store_true",
        help="turn on historically winning but unsearched incumbent features",
    )
    parser.add_argument(
        "--candidate-json", type=Path,
        help="time exact target/config pairs from an analysis JSON manifest",
    )
    parser.add_argument(
        "--maximum-current-pct", type=float,
        help="keep shapes whose current --ranking-json winner is at or below "
        "this percent of Torch",
    )
    parser.add_argument(
        "--worst-shapes", type=int,
        help="select this many shapes with the greatest frequency-weighted "
        "Ayla latency deficit in --ranking-json",
    )
    parser.add_argument(
        "--ranking-json", type=Path, default=RAW_JSON,
        help="existing canonical results used by --worst-shapes",
    )
    parser.add_argument(
        "--closest-below-torch", type=int,
        help="select this many shapes nearest to, but still below, Torch",
    )
    parser.add_argument(
        "--lowest-pct-shapes", type=int,
        help="select this many shapes with the lowest current percent of Torch",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--config-num-shards", type=int, default=1)
    parser.add_argument("--config-shard-index", type=int, default=0)
    args = parser.parse_args()

    validations = validate_shapes(load_shapes(args.corpus))
    validation_rows = [validation_row(item) for item in validations]
    if not args.skip_validation_write:
        write_rows_csv(validation_rows, args.validation_csv)
        write_json(validation_rows, args.validation_json)
    supported_signatures = [item.shape for item in validations if item.supported]
    supported = benchmark_shapes(validations)
    if args.workload != "all":
        supported = [shape for shape in supported if shape.op == args.workload]
    supported_shape_ids = {shape.shape_id for shape in supported}
    print(
        f"validation: {len(supported_signatures)}/{len(validations)} pooled "
        f"signatures supported, producing {len(supported)} canonical shapes; "
        f"validation reports "
        f"{'reused' if args.skip_validation_write else 'written'}"
    )
    if args.validate_only:
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (validation reports were still written)")

    if args.shape_filter:
        pattern = re.compile(args.shape_filter)
        supported = [shape for shape in supported if pattern.search(shape.shape_id)]
    ranking_rows: list[dict[str, Any]] = []
    torch_by_shape: dict[str, dict[str, Any]] = {}
    ayla_by_shape: dict[str, list[dict[str, Any]]] = {}
    ayla_by_problem: dict[
        tuple[int, int, int, str], list[dict[str, Any]]
    ] = {}
    if (
        args.worst_shapes is not None
        or args.closest_below_torch is not None
        or args.lowest_pct_shapes is not None
        or args.multicast_resweep_only
        or args.multicast_transfer_only
        or args.multicast_neighborhood_only
        or args.local_refine_only
        or args.transfer_only
        or args.recheck_best_only
        or args.rhs_tma_slice_only
        or args.workgroup_parallel_only
        or args.smem_reuse_only
        or args.wgmma_pipeline_only
        or args.continuous_wgmma_only
        or args.wgmma_group_only
        or args.wgmma_epilogue_only
        or args.continuous_cluster_only
        or args.continuous_architecture_only
        or args.grid_swap_only
        or args.grid_swap_refine_only
        or args.grid_swap_ablation_only
        or args.exact_axis_ablation_only
        or args.lhs_promotion_ablation_only
        or args.rhs_promotion_ablation_only
        or args.cache_hint_ablation_only
        or args.stage_ablation_only
        or args.ptx_optimization_ablation_only
        or args.output_layout_ablation_only
        or args.output_slicing_ablation_only
        or args.output_n_grouping_ablation_only
        or args.continuous_toggle_ablation_only
        or args.wgmma_feature_ablation_only
        or args.output_store_ablation_only
        or args.local_ablation_only
        or args.documented_gap_ablation_only
        or args.candidate_json is not None
        or args.maximum_current_pct is not None
    ):
        ranking_rows = _read_existing(args.ranking_json)
        torch_by_shape = {
            str(row["shape_id"]): row for row in ranking_rows
            if row.get("implementation") == "torch"
            and row.get("status") == "ok"
            and row.get("latency_ms") is not None
        }
        for row in ranking_rows:
            if (
                row.get("implementation") == "ayla"
                and row.get("status") == "ok"
                and row.get("latency_ms") is not None
                and not bool(
                    row.get("configuration", {}).get("unsafe_reuse_smem")
                )
            ):
                ayla_by_shape.setdefault(str(row["shape_id"]), []).append(row)
                problem = tuple(
                    int(row[field]) for field in ("m", "n", "k")
                ) + (str(row["dtype"]),)
                ayla_by_problem.setdefault(problem, []).append(row)

    def current_best(shape: Shape) -> dict[str, Any] | None:
        candidates = ayla_by_shape.get(shape.shape_id, [])
        if not candidates and shape.op == "addmm":
            candidates = ayla_by_problem.get(
                (shape.m, shape.n, shape.k, shape.dtype), [],
            )
        baseline = torch_by_shape.get(shape.shape_id)

        def candidate_pct(row: dict[str, Any]) -> float:
            value = row.get("torch_pct")
            if value is not None:
                return float(value)
            if baseline is None:
                return float("-inf")
            return (
                float(baseline["latency_ms"]) / float(row["latency_ms"]) * 100
            )

        return max(
            candidates, key=candidate_pct,
        ) if candidates else None

    explicit_candidates = (
        _load_explicit_candidates(args.candidate_json)
        if args.candidate_json is not None else {}
    )
    if args.candidate_json is not None:
        supported = [
            shape for shape in supported
            if shape.shape_id in explicit_candidates
        ]

    if args.maximum_current_pct is not None:
        def within_current_pct(shape: Shape) -> bool:
            best = current_best(shape)
            baseline = torch_by_shape.get(shape.shape_id)
            if best is None:
                return False
            value = best.get("torch_pct")
            if value is not None:
                return float(value) <= args.maximum_current_pct
            return (
                baseline is not None
                and float(baseline["latency_ms"])
                / float(best["latency_ms"]) * 100
                <= args.maximum_current_pct
            )
        supported = [shape for shape in supported if within_current_pct(shape)]

    if args.recheck_best_only:
        ranked_count = len(supported)
        supported = _shapes_with_current_best(supported, current_best)
        print(
            f"recheck ranking coverage: {len(supported)}/{ranked_count} shapes"
        )

    if args.worst_shapes is not None:
        if args.worst_shapes < 1:
            parser.error("--worst-shapes must be positive")

        def deficit(shape: Shape) -> float:
            baseline = torch_by_shape.get(shape.shape_id)
            best = current_best(shape)
            if baseline is None or best is None:
                return float("-inf")
            return (
                float(best["latency_ms"]) - float(baseline["latency_ms"])
            ) * shape.count

        supported.sort(key=deficit, reverse=True)
        supported = supported[:args.worst_shapes]
    if args.closest_below_torch is not None:
        if args.closest_below_torch < 1:
            parser.error("--closest-below-torch must be positive")
        if args.worst_shapes is not None:
            parser.error(
                "--closest-below-torch and --worst-shapes are mutually exclusive"
            )

        def current_pct(shape: Shape) -> float:
            baseline = torch_by_shape.get(shape.shape_id)
            best = current_best(shape)
            if baseline is None or best is None:
                return float("-inf")
            return (
                float(baseline["latency_ms"])
                / float(best["latency_ms"])
                * 100
            )

        supported = [shape for shape in supported if current_pct(shape) < 100]
        supported.sort(key=current_pct, reverse=True)
        supported = supported[:args.closest_below_torch]
    if args.lowest_pct_shapes is not None:
        if args.lowest_pct_shapes < 1:
            parser.error("--lowest-pct-shapes must be positive")
        if args.worst_shapes is not None or args.closest_below_torch is not None:
            parser.error(
                "--lowest-pct-shapes, --closest-below-torch, and "
                "--worst-shapes are mutually exclusive"
            )

        def lowest_current_pct(shape: Shape) -> float:
            best = current_best(shape)
            if best is None:
                return float("-inf")
            value = best.get("torch_pct")
            if value is not None:
                return float(value)
            baseline = torch_by_shape.get(shape.shape_id)
            if baseline is None:
                return float("-inf")
            return (
                float(baseline["latency_ms"])
                / float(best["latency_ms"])
                * 100
            )

        supported.sort(key=lowest_current_pct)
        supported = supported[:args.lowest_pct_shapes]
    if args.max_shapes is not None:
        if args.max_shapes < 1:
            parser.error("--max-shapes must be positive")
        supported = supported[:args.max_shapes]
    if args.config_timeout_seconds < 0:
        parser.error("--config-timeout-seconds must be nonnegative")
    if args.max_timing_rounds < 3:
        parser.error("--max-timing-rounds must be at least three")
    if args.cold_cache_mib < 0:
        parser.error("--cold-cache-mib must be nonnegative")
    if args.timing == "cold" and not args.cold_cache_mib:
        parser.error("--timing cold requires --cold-cache-mib")
    if args.timing == "warm" and args.cold_cache_mib:
        parser.error("--timing warm does not flush; drop --cold-cache-mib")
    timing = {
        "cold_cache_bytes": args.cold_cache_mib * 1024 * 1024,
        "inference": args.timing == "inference",
    }
    global _TIMING_PROTOCOL
    _TIMING_PROTOCOL = (
        f"cold{args.cold_cache_mib}"
        if args.timing == "cold"
        else "inference-v2"
        if args.timing == "inference"
        else args.timing
    )
    benchmark_common._MAX_ROUNDS = args.max_timing_rounds
    if args.compile_workers < 0:
        parser.error("--compile-workers must be nonnegative")
    global _COMPILE_WORKERS
    _COMPILE_WORKERS = args.compile_workers and max(
        1, args.compile_workers // (args.num_shards * args.config_num_shards),
    )
    # The isolated worker always times each candidate as an interleaved
    # [Torch, Ayla] pair, so pairing and the per-configuration watchdog are
    # compatible. ``--pair-each-config`` only selects the equivalent
    # in-process path when isolation is disabled.
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.config_num_shards < 1:
        parser.error("--config-num-shards must be positive")
    if not 0 <= args.config_shard_index < args.config_num_shards:
        parser.error(
            "--config-shard-index must be in [0, --config-num-shards)"
        )

    representatives: dict[str, str] = {}
    for shape in supported:
        representatives.setdefault(regime_name(shape), shape.shape_id)
    supported = [
        shape for index, shape in enumerate(supported)
        if index % args.num_shards == args.shard_index
    ]

    rows: list[dict[str, object]] = (
        [] if args.no_resume else _read_existing(args.output_json)
    )
    # Validation rules can become stricter between resumed runs. Do not retain
    # measurements for shapes that the current run now classifies as skipped.
    rows = [
        row for row in rows
        if str(row.get("shape_id", "")) in supported_shape_ids
    ]
    # Rows from before the field existed were timed hot or after a plain flush
    # that L2-pinned operands survive; neither is comparable to a new row.
    stale = {
        str(row.get("timing_protocol", "legacy")) for row in rows
    } - {_TIMING_PROTOCOL}
    if stale:
        parser.error(
            f"{args.output_json} holds rows timed as {sorted(stale)}, not "
            f"{_TIMING_PROTOCOL!r}; write to a new output or pass --no-resume"
        )
    config_pattern = re.compile(args.config_filter) if args.config_filter else None
    torch.backends.cuda.matmul.allow_tf32 = True

    for index, shape in enumerate(supported, 1):
        exclusive = sum((
            args.fallback_only, args.refine_only, args.tile_refine_only,
            args.tune_only, args.cublas_inspired_only,
            args.resource_tune_only, args.coarse_tile_only, args.wg2d_only,
            args.multicast_only,
            args.multicast_ablation_only,
            args.multicast_resweep_only,
            args.multicast_transfer_only,
            args.multicast_neighborhood_only,
            args.pipeline_tile_only,
            args.gap_tune_only,
            args.narrow_tile_only,
            args.bmm_tune_only,
            args.exact_cta_only,
            args.exact_cta_refine_only,
            args.local_refine_only,
            args.cluster_tune_only,
            args.narrow_wg2d_only,
            args.transfer_only,
            args.static_loop_only,
            args.small_cta_writeback_only,
            args.small_cta_writeback_refine_only,
            args.small_cta_writeback_transfer_only,
            args.architecture_gap_only,
            args.tail_end_only,
            args.recheck_best_only,
            args.rhs_tma_slice_only,
            args.workgroup_parallel_only,
            args.smem_reuse_only,
            args.wgmma_pipeline_only,
            args.continuous_wgmma_only,
            args.wgmma_group_only,
            args.wgmma_epilogue_only,
            args.continuous_cluster_only,
            args.continuous_architecture_only,
            args.grid_swap_only,
            args.grid_swap_refine_only,
            args.grid_swap_ablation_only,
            args.exact_axis_ablation_only,
            args.lhs_promotion_ablation_only,
            args.rhs_promotion_ablation_only,
            args.cache_hint_ablation_only,
            args.stage_ablation_only,
            args.ptx_optimization_ablation_only,
            args.output_layout_ablation_only,
            args.output_slicing_ablation_only,
            args.output_n_grouping_ablation_only,
            args.continuous_toggle_ablation_only,
            args.wgmma_feature_ablation_only,
            args.output_store_ablation_only,
            args.local_ablation_only,
            args.documented_gap_ablation_only,
            args.candidate_json is not None,
        ))
        if exclusive > 1:
            parser.error("fallback/refinement modes are mutually exclusive")
        schedules = (
            list(explicit_candidates.get(shape.shape_id, ()))
            if args.candidate_json is not None
            else [fallback_config()] if args.fallback_only
            else list(refinement_configs(shape)) if args.refine_only
            else list(tile_refinement_configs()) if args.tile_refine_only
            else list(targeted_tuning_configs(shape)) if args.tune_only
            else list(cublas_inspired_configs(shape))
            if args.cublas_inspired_only
            else list(resource_tuning_configs(shape))
            if args.resource_tune_only
            else list(coarse_tile_configs(shape))
            if args.coarse_tile_only
            else list(two_dimensional_wg_configs(shape))
            if args.wg2d_only
            else list(multicast_configs(shape))
            if args.multicast_only
            else list(multicast_ablation_configs(shape))
            if args.multicast_ablation_only
            else list(multicast_resweep_configs(
                shape, _schedule_from_row(best),
            ))
            if args.multicast_resweep_only and (best := current_best(shape)) is not None
            else list(multicast_winner_transfer_configs(shape))
            if args.multicast_transfer_only
            else list(multicast_incumbent_neighborhood_configs(
                shape, _schedule_from_row(best),
            ))
            if args.multicast_neighborhood_only and (best := current_best(shape)) is not None
            else list(pipeline_tile_configs(shape))
            if args.pipeline_tile_only
            else list(gap_tuning_configs(shape))
            if args.gap_tune_only
            else list(narrow_tile_configs(shape))
            if args.narrow_tile_only
            else list(bmm_tuning_configs(shape))
            if args.bmm_tune_only
            else list(exact_cta_configs(shape))
            if args.exact_cta_only
            else list(exact_cta_refinement_configs(shape))
            if args.exact_cta_refine_only
            else list(local_refinement_configs(shape, _schedule_from_row(best)))
            if args.local_refine_only and (best := current_best(shape)) is not None
            else list(cluster_tuning_configs(shape))
            if args.cluster_tune_only
            else list(narrow_wg2d_configs(shape))
            if args.narrow_wg2d_only
            else list(transfer_configs(shape, ranking_rows))
            if args.transfer_only
            else list(static_loop_configs(shape))
            if args.static_loop_only
            else list(small_cta_writeback_configs(shape))
            if args.small_cta_writeback_only
            else list(small_cta_writeback_refinement_configs(shape))
            if args.small_cta_writeback_refine_only
            else list(small_cta_writeback_transfer_configs(shape))
            if args.small_cta_writeback_transfer_only
            else list(architecture_gap_configs(shape))
            if args.architecture_gap_only
            else list(tail_end_configs(shape))
            if args.tail_end_only
            else [_schedule_from_row(best)]
            if args.recheck_best_only and (best := current_best(shape)) is not None
            else list(rhs_tma_slicing_configs(shape, _schedule_from_row(best)))
            if args.rhs_tma_slice_only and (best := current_best(shape)) is not None
            else list(workgroup_parallelism_configs(
                shape, _schedule_from_row(best),
            ))
            if args.workgroup_parallel_only and (best := current_best(shape)) is not None
            else list(smem_reuse_configs(shape, _schedule_from_row(best)))
            if args.smem_reuse_only and (best := current_best(shape)) is not None
            else list(wgmma_pipeline_configs(shape, _schedule_from_row(best)))
            if args.wgmma_pipeline_only and (best := current_best(shape)) is not None
            else list(continuous_wgmma_configs(
                shape, _schedule_from_row(best),
            ))
            if args.continuous_wgmma_only and (best := current_best(shape)) is not None
            else list(wgmma_group_configs(
                shape, _schedule_from_row(best),
            ))
            if args.wgmma_group_only and (best := current_best(shape)) is not None
            else list(wgmma_epilogue_configs(
                shape, _schedule_from_row(best),
            ))
            if args.wgmma_epilogue_only and (best := current_best(shape)) is not None
            else list(continuous_cluster_configs(shape))
            if args.continuous_cluster_only
            else list(continuous_architecture_gap_configs(shape))
            if args.continuous_architecture_only
            else list(grid_swap_configs(shape, _schedule_from_row(best)))
            if args.grid_swap_only and (best := current_best(shape)) is not None
            else list(grid_swap_refinement_configs(
                shape, _schedule_from_row(best),
            ))
            if args.grid_swap_refine_only and (best := current_best(shape)) is not None
            else list(grid_swap_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.grid_swap_ablation_only and (best := current_best(shape)) is not None
            else list(exact_axis_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.exact_axis_ablation_only and (best := current_best(shape)) is not None
            else list(lhs_promotion_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.lhs_promotion_ablation_only and (best := current_best(shape)) is not None
            else list(rhs_promotion_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.rhs_promotion_ablation_only and (best := current_best(shape)) is not None
            else list(cache_hint_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.cache_hint_ablation_only and (best := current_best(shape)) is not None
            else list(stage_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.stage_ablation_only and (best := current_best(shape)) is not None
            else list(ptx_optimization_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.ptx_optimization_ablation_only and (best := current_best(shape)) is not None
            else list(output_layout_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.output_layout_ablation_only and (best := current_best(shape)) is not None
            else list(output_slicing_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.output_slicing_ablation_only and (best := current_best(shape)) is not None
            else list(output_n_grouping_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.output_n_grouping_ablation_only and (best := current_best(shape)) is not None
            else list(continuous_toggle_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.continuous_toggle_ablation_only and (best := current_best(shape)) is not None
            else list(wgmma_feature_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.wgmma_feature_ablation_only and (best := current_best(shape)) is not None
            else list(output_store_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.output_store_ablation_only and (best := current_best(shape)) is not None
            else list(local_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.local_ablation_only and (best := current_best(shape)) is not None
            else list(documented_gap_ablation_configs(
                shape, _schedule_from_row(best),
            ))
            if args.documented_gap_ablation_only and (best := current_best(shape)) is not None
            else [
                config for config in seed_configs(shape)
                if config_pattern is None or config_pattern.search(config.name)
            ]
        )
        if args.jit_heuristic:
            jit_config = _jit_heuristic_config(shape)
            schedules = [] if jit_config is None else [jit_config]
        if config_pattern is not None:
            schedules = [
                config for config in schedules if config_pattern.search(config.name)
            ]
        schedules = [
            config for config in schedules
            if int.from_bytes(
                hashlib.sha256(config.name.encode()).digest()[:8], "big",
            ) % args.config_num_shards == args.config_shard_index
        ]
        if (
            ranking_rows
            and args.candidate_json is None
            and not args.recheck_best_only
            and not args.multicast_ablation_only
            and not args.multicast_resweep_only
            and not args.multicast_neighborhood_only
            and not args.grid_swap_ablation_only
            and not args.exact_axis_ablation_only
            and not args.lhs_promotion_ablation_only
            and not args.rhs_promotion_ablation_only
            and not args.cache_hint_ablation_only
            and not args.stage_ablation_only
            and not args.ptx_optimization_ablation_only
            and not args.output_layout_ablation_only
            and not args.output_slicing_ablation_only
            and not args.output_n_grouping_ablation_only
            and not args.continuous_toggle_ablation_only
            and not args.wgmma_feature_ablation_only
            and not args.output_store_ablation_only
            and not args.local_ablation_only
            and not args.documented_gap_ablation_only
        ):
            measured_keys = {
                _schedule_key(_schedule_from_row(row))
                for row in ayla_by_shape.get(shape.shape_id, [])
            }
            schedules = [
                config for config in schedules
                if _schedule_key(config) not in measured_keys
            ]
        seed_names = {config.name for config in schedules}
        if (
            not args.fallback_only
            and not args.refine_only
            and not args.tile_refine_only
            and args.explore_knobs
            and representatives[regime_name(shape)] == shape.shape_id
        ):
            schedules.extend(
                config for config in exploration_configs(shape)
                if config.name not in seed_names
                and (config_pattern is None or config_pattern.search(config.name))
            )
        if not schedules:
            print(f"[shape {index}/{len(supported)}] no matching seed configs")
            continue
        expected = {(shape.shape_id, f"torch.{shape.op}")} | {
            (shape.shape_id, schedule.name) for schedule in schedules
        }
        if expected <= {_row_key(row) for row in rows}:
            print(f"[shape {index}/{len(supported)}] resume {shape.shape_id}")
            continue
        # Replace this invocation's rows if a prior run stopped mid-shape.
        # Rows for configs excluded by --config-filter remain untouched.
        rows = [row for row in rows if _row_key(row) not in expected]
        print(
            f"[shape {index}/{len(supported)}] {shape.shape_id} "
            f"frequency={shape.count}: build {len(schedules)} seeds",
            flush=True,
        )
        ayla_configs: list[GemmBenchConfig] = []
        built: list[tuple[ScheduleConfig, float, float, str]] = []

        def scope(schedule: ScheduleConfig) -> str:
            if schedule.name == fallback_config().name:
                return "coverage_fallback"
            if schedule.name.startswith("refine_"):
                return "bounded_refinement"
            return (
                "full_corpus_seed" if schedule.name in seed_names
                else "regime_probe"
            )

        if args.config_timeout_seconds > 0:
            baseline_result, measurements = _benchmark_isolated(
                shape, schedules, args.config_timeout_seconds, timing,
            )
            if baseline_result is not None:
                rows.append(_result_row(
                    shape, "torch", None,
                    latency_ms=baseline_result["latency_ms"],
                    tflops=baseline_result["tflops"],
                    torch_pct=100.0,
                    search_scope="torch_baseline",
                ))
            for schedule, measurement, error in measurements:
                if measurement is None:
                    rows.append(_result_row(
                        shape, "ayla", schedule,
                        error=error,
                        search_scope=scope(schedule),
                    ))
                    continue
                rows.append(_result_row(
                    shape, "ayla", schedule,
                    latency_ms=measurement["latency_ms"],
                    tflops=measurement["tflops"],
                    torch_pct=measurement.get("torch_pct"),
                    max_abs_error=measurement["max_abs"],
                    relative_rms_error=measurement["relative_rms"],
                    search_scope=scope(schedule),
                ))
            _write_raw(rows, args.output_json, args.output_csv)
            continue

        for schedule, entry in zip(
            schedules, _assemble_schedules(shape, schedules, None), strict=True,
        ):
            error = (
                f"{entry[0]}: {entry[1]}"
                if entry is not None and isinstance(entry[1], str) else None
            )
            if error is None:
                try:
                    bench, max_abs, relative_rms = _make_ayla_config(
                        shape, schedule, entry,
                    )
                except Exception as exception:
                    error = f"{type(exception).__name__}: {exception}"
                else:
                    ayla_configs.append(bench)
                    built.append((
                        schedule, max_abs, relative_rms,
                        scope(schedule),
                    ))
                    continue
            rows.append(_result_row(
                shape, "ayla", schedule,
                error=error,
                search_scope=scope(schedule),
            ))
            print(f"  reject {schedule.name}: {error}")

        baseline = _baseline_config(shape)
        if args.pair_each_config:
            torch_results = []
            for bench, (
                schedule, max_abs, relative_rms, search_scope,
            ) in zip(ayla_configs, built, strict=True):
                torch_result, ayla_result = run([baseline, bench], **timing)
                torch_results.append(torch_result)
                rows.append(_result_row(
                    shape, "ayla", schedule,
                    latency_ms=ayla_result.median_ms,
                    tflops=ayla_result.tflops,
                    torch_pct=ayla_result.torch_mm_pct,
                    max_abs_error=max_abs,
                    relative_rms_error=relative_rms,
                    search_scope=search_scope,
                ))
            if torch_results:
                torch_latency = statistics.median(
                    result.median_ms for result in torch_results
                )
                rows.append(_result_row(
                    shape, "torch", None,
                    latency_ms=torch_latency,
                    tflops=(
                        2 * shape.m * shape.n * shape.k
                        / (torch_latency / 1e3) / 1e12
                    ),
                    torch_pct=100.0,
                    search_scope="torch_baseline",
                ))
            _write_raw(rows, args.output_json, args.output_csv)
            continue
        configs = [baseline, *ayla_configs]
        results = run(configs, **timing)
        print_results(results)
        torch_result, *ayla_results = results
        rows.append(_result_row(
            shape, "torch", None,
            latency_ms=torch_result.median_ms,
            tflops=torch_result.tflops,
            torch_pct=100.0,
            search_scope="torch_baseline",
        ))
        for (schedule, max_abs, relative_rms, search_scope), result in zip(
            built, ayla_results, strict=True,
        ):
            rows.append(_result_row(
                shape, "ayla", schedule,
                latency_ms=result.median_ms,
                tflops=result.tflops,
                torch_pct=result.torch_mm_pct,
                max_abs_error=max_abs,
                relative_rms_error=relative_rms,
                search_scope=search_scope,
            ))
        _write_raw(rows, args.output_json, args.output_csv)
        del configs, ayla_configs
        torch.cuda.empty_cache()

    print(f"raw JSON: {args.output_json.resolve()}")
    print(f"raw CSV:  {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
