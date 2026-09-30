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
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch

from benchmarks.common import (
    GemmBenchConfig,
    assert_against_torch,
    gemm_bench_config,
    out_parameter_op,
    print_results,
    run,
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
    benchmark_shapes,
    cublas_inspired_configs,
    cluster_tuning_configs,
    coarse_tile_configs,
    exploration_configs,
    exact_cta_configs,
    exact_cta_refinement_configs,
    fallback_config,
    gap_tuning_configs,
    load_shapes,
    local_refinement_configs,
    multicast_configs,
    narrow_tile_configs,
    narrow_wg2d_configs,
    pipeline_tile_configs,
    regime,
    regime_name,
    refinement_configs,
    resource_tuning_configs,
    rhs_tma_slicing_configs,
    smem_reuse_configs,
    seed_configs,
    static_loop_configs,
    targeted_tuning_configs,
    tile_refinement_configs,
    two_dimensional_wg_configs,
    workgroup_parallelism_configs,
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

_DEFAULT_CONFIG_TIMEOUT_S = 120.0


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
        {"name": "torch.mm"}
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
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _make_ayla_config(
    shape: Shape, schedule: ScheduleConfig,
) -> tuple[GemmBenchConfig, float, float]:
    # Keep Ayla's eagerly registered collection out of the CPU-only validation
    # path. Importing this submodule compiles the collection as a side effect.
    from ayla.gemm_collection.gemm import (
        auto_matmul_4096_wgmma256_wg_tma_store_module,
    )
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher

    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=shape.m,
        n=shape.n,
        k=shape.k,
        dtype=IR_DTYPES[shape.dtype],
        **schedule.builder_kwargs(),
    )
    compute = out_parameter_op(make_ptx_launcher(make_ptx_kernel(module)))
    bench = gemm_bench_config(
        name=f"Ayla {schedule.name}",
        m=shape.m, n=shape.n, k=shape.k,
        compute=compute, dtype=TORCH_DTYPES[shape.dtype],
    )
    max_abs, relative_rms = assert_against_torch(
        bench,
        seed=int(shape.shape_id[-8:], 16),
        max_abs_error=4,
        max_relative_rms_error=2e-3,
    )
    return bench, max_abs, relative_rms


def _isolated_benchmark_worker(
    shape: Shape,
    schedules: tuple[ScheduleConfig, ...],
    include_baseline: bool,
    sender: Connection,
) -> None:
    """Build and time configs in a process the parent can safely terminate."""
    try:
        if include_baseline:
            sender.send(("started_baseline", None))
            baseline = gemm_bench_config(
                name="Torch", m=shape.m, n=shape.n, k=shape.k,
                compute=torch_mm_nt, dtype=TORCH_DTYPES[shape.dtype],
            )
            result = run([baseline])[0]
            sender.send(("baseline", {
                "latency_ms": result.median_ms,
                "tflops": result.tflops,
            }))
            del baseline
            torch.cuda.empty_cache()

        for schedule in schedules:
            sender.send(("started", schedule.name))
            try:
                bench, max_abs, relative_rms = _make_ayla_config(shape, schedule)
                result = run([bench])[0]
            except Exception as error:
                sender.send(("error", schedule.name, (
                    type(error).__name__, str(error),
                )))
            else:
                sender.send(("result", schedule.name, {
                    "latency_ms": result.median_ms,
                    "tflops": result.tflops,
                    "max_abs": max_abs,
                    "relative_rms": relative_rms,
                }))
                del bench
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
) -> tuple[dict[str, float], list[tuple[ScheduleConfig, dict[str, float] | None, str | None]]]:
    """Benchmark schedules with a restartable per-configuration watchdog."""
    context = mp.get_context("spawn")
    baseline: dict[str, float] | None = None
    measurements: list[
        tuple[ScheduleConfig, dict[str, float] | None, str | None]
    ] = []
    position = 0

    while position < len(schedules):
        receiver, sender = context.Pipe(duplex=False)
        remaining = tuple(schedules[position:])
        process = context.Process(
            target=_isolated_benchmark_worker,
            args=(shape, remaining, baseline is None, sender),
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
                if kind == "started_baseline":
                    deadline = time.monotonic() + timeout_s
                elif kind == "baseline":
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

    if baseline is None:
        raise RuntimeError("benchmark worker returned no Torch baseline")
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
        best = min(candidates, key=lambda row: float(row["latency_ms"]))
        m, n, k = (int(best[field]) for field in ("m", "n", "k"))
        pct = float(baseline["latency_ms"]) / float(best["latency_ms"]) * 100
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
        help="sweep tile shapes and pipeline depth with SMEM reuse",
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
        "--architecture-gap-only", action="store_true",
        help="sweep deep pipelines, 64x80 tiles, and wide clustered tiles",
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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    validations = validate_shapes(load_shapes(args.corpus))
    validation_rows = [validation_row(item) for item in validations]
    if not args.skip_validation_write:
        write_rows_csv(validation_rows, args.validation_csv)
        write_json(validation_rows, args.validation_json)
    supported_signatures = [item.shape for item in validations if item.supported]
    supported = benchmark_shapes(validations)
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
    if (
        args.worst_shapes is not None
        or args.closest_below_torch is not None
        or args.local_refine_only
        or args.transfer_only
        or args.recheck_best_only
        or args.rhs_tma_slice_only
        or args.workgroup_parallel_only
        or args.smem_reuse_only
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
            ):
                ayla_by_shape.setdefault(str(row["shape_id"]), []).append(row)

    def current_best(shape: Shape) -> dict[str, Any] | None:
        candidates = ayla_by_shape.get(shape.shape_id, [])
        return min(
            candidates, key=lambda row: float(row["latency_ms"]),
        ) if candidates else None

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
    if args.max_shapes is not None:
        if args.max_shapes < 1:
            parser.error("--max-shapes must be positive")
        supported = supported[:args.max_shapes]
    if args.config_timeout_seconds < 0:
        parser.error("--config-timeout-seconds must be nonnegative")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

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
    config_pattern = re.compile(args.config_filter) if args.config_filter else None
    torch.backends.cuda.matmul.allow_tf32 = True

    for index, shape in enumerate(supported, 1):
        exclusive = sum((
            args.fallback_only, args.refine_only, args.tile_refine_only,
            args.tune_only, args.cublas_inspired_only,
            args.resource_tune_only, args.coarse_tile_only, args.wg2d_only,
            args.multicast_only,
            args.pipeline_tile_only,
            args.gap_tune_only,
            args.narrow_tile_only,
            args.exact_cta_only,
            args.exact_cta_refine_only,
            args.local_refine_only,
            args.cluster_tune_only,
            args.narrow_wg2d_only,
            args.transfer_only,
            args.static_loop_only,
            args.architecture_gap_only,
            args.recheck_best_only,
            args.rhs_tma_slice_only,
            args.workgroup_parallel_only,
            args.smem_reuse_only,
        ))
        if exclusive > 1:
            parser.error("fallback/refinement modes are mutually exclusive")
        schedules = (
            [fallback_config()] if args.fallback_only
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
            else list(pipeline_tile_configs(shape))
            if args.pipeline_tile_only
            else list(gap_tuning_configs(shape))
            if args.gap_tune_only
            else list(narrow_tile_configs(shape))
            if args.narrow_tile_only
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
            else list(architecture_gap_configs(shape))
            if args.architecture_gap_only
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
            else [
                config for config in seed_configs(shape)
                if config_pattern is None or config_pattern.search(config.name)
            ]
        )
        if config_pattern is not None:
            schedules = [
                config for config in schedules if config_pattern.search(config.name)
            ]
        if ranking_rows and not args.recheck_best_only:
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
        expected = {(shape.shape_id, "torch.mm")} | {
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
                shape, schedules, args.config_timeout_seconds,
            )
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
                    torch_pct=(
                        baseline_result["latency_ms"]
                        / measurement["latency_ms"] * 100
                    ),
                    max_abs_error=measurement["max_abs"],
                    relative_rms_error=measurement["relative_rms"],
                    search_scope=scope(schedule),
                ))
            _write_raw(rows, args.output_json, args.output_csv)
            continue

        for schedule in schedules:
            try:
                bench, max_abs, relative_rms = _make_ayla_config(shape, schedule)
            except Exception as error:
                rows.append(_result_row(
                    shape, "ayla", schedule,
                    error=f"{type(error).__name__}: {error}",
                    search_scope=scope(schedule),
                ))
                print(f"  reject {schedule.name}: {type(error).__name__}: {error}")
            else:
                ayla_configs.append(bench)
                built.append((
                    schedule, max_abs, relative_rms,
                    scope(schedule),
                ))

        baseline = gemm_bench_config(
            name="Torch", m=shape.m, n=shape.n, k=shape.k,
            compute=torch_mm_nt, dtype=TORCH_DTYPES[shape.dtype],
        )
        configs = [baseline, *ayla_configs]
        results = run(configs)
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
