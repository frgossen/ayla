"""Benchmark the library's JIT GEMMs against Torch on recorded model shapes.

Full schedule search is available on feature/consolidated.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any
import json

import torch

from ayla.gemm_jit import gemm_op, select_gemm_schedule
from benchmarks import common as benchmark_common
from benchmarks.common import (
    GemmBenchConfig,
    addmm_bench_config,
    bmm_bench_config,
    gemm_bench_config,
    assert_addmm_against_torch,
    assert_against_torch,
    assert_bmm_against_torch,
    out_parameter_addmm_op,
    out_parameter_bmm_op,
    out_parameter_op,
    torch_addmm_nt,
    torch_bmm_nt,
    torch_mm_nt,
    print_results,
    run,
)
from benchmarks.oss_gemm_corpus import (
    CORPUS,
    RAW_CSV,
    RAW_JSON,
    VALIDATION_CSV,
    VALIDATION_JSON,
    Shape,
    benchmark_shapes,
    load_shapes,
    validate_shapes,
    validation_row,
    write_json,
    write_rows_csv,
)

TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
MAX_ABS_ERROR = {"float16": 4, "bfloat16": 32}


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
        name="Torch",
        m=shape.m,
        n=shape.n,
        k=shape.k,
        compute=compute,
        dtype=TORCH_DTYPES[shape.dtype],
    )


def _bench_ayla_launcher(
    shape: Shape,
    launcher: Any,
) -> tuple[GemmBenchConfig, float, float]:
    """Wrap ``launcher`` as a benchmark config after checking it against Torch."""
    if shape.op == "bmm":
        assert shape.batch is not None
        bench = bmm_bench_config(
            name="Ayla JIT",
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
            if shape.op == "addmm"
            else out_parameter_op(launcher)
        )
        bench_builder = addmm_bench_config if shape.op == "addmm" else gemm_bench_config
        bench = bench_builder(
            name="Ayla JIT",
            m=shape.m,
            n=shape.n,
            k=shape.k,
            compute=compute,
            dtype=TORCH_DTYPES[shape.dtype],
        )
        checker = (
            assert_addmm_against_torch if shape.op == "addmm" else assert_against_torch
        )
        max_abs, relative_rms = checker(
            bench,
            seed=int(shape.shape_id[-8:], 16),
            max_abs_error=MAX_ABS_ERROR[shape.dtype],
            max_relative_rms_error=2e-3,
        )
    return bench, max_abs, relative_rms


def _measure(
    shape: Shape, rounds: int, timing: str, cache_mib: int
) -> list[dict[str, Any]]:
    benchmark_common._MAX_ROUNDS = rounds
    seed = select_gemm_schedule(
        shape.op, shape.dtype, shape.m, shape.n, shape.k, shape.batch or 1
    )
    if seed is None:
        raise ValueError("no supported JIT schedule for this shape")
    bench, max_abs, relative_rms = _bench_ayla_launcher(
        shape, gemm_op(shape.op, shape.dtype)
    )
    configs = [_baseline_config(shape), bench]
    results = run(
        configs, inference=timing == "inference", cold_cache_bytes=cache_mib * 2**20
    )
    print_results(results)
    rows = []
    for implementation, result in zip(("torch", "ayla"), results, strict=True):
        rows.append(
            {
                "shape_id": shape.shape_id,
                "op": shape.op,
                "dtype": shape.dtype,
                "batch": shape.batch,
                "m": shape.m,
                "n": shape.n,
                "k": shape.k,
                "count": shape.count,
                "suites": shape.suites,
                "implementation": implementation,
                "compiler_feature_space": "main",
                "configuration": {"name": f"torch.{shape.op}"}
                if implementation == "torch"
                else {"name": "jit_heuristic", **seed.captured_config},
                "schedule_source_shape": seed.source_shape
                if implementation == "ayla"
                else None,
                "schedule_adaptations": seed.adaptations
                if implementation == "ayla"
                else (),
                "latency_ms": result.median_ms,
                "tflops": result.tflops,
                "torch_pct": result.torch_mm_pct,
                "correctness_max_abs": max_abs if implementation == "ayla" else None,
                "correctness_relative_rms": relative_rms
                if implementation == "ayla"
                else None,
                "status": "ok",
                "error": None,
                "timing_protocol": f"{timing}:main-jit-v1:{cache_mib}",
                "gpu": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
    return rows


def _worker(shape, rounds, timing, cache_mib, sender):
    try:
        sender.send((_measure(shape, rounds, timing, cache_mib), None))
    except Exception as error:
        sender.send((None, f"{type(error).__name__}: {error}"))
    finally:
        sender.close()


def _measure_isolated(shape, args):
    if not args.config_timeout_seconds:
        return _measure(shape, args.max_timing_rounds, args.timing, args.cold_cache_mib)
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_worker,
        args=(
            shape,
            args.max_timing_rounds,
            args.timing,
            args.cold_cache_mib,
            sender,
        ),
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(args.config_timeout_seconds):
            raise TimeoutError(f"exceeded {args.config_timeout_seconds:g}s")
        rows, error = receiver.recv()
        if error is not None:
            raise RuntimeError(error)
        return rows
    finally:
        receiver.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument(
        "--workload", choices=("mm", "addmm", "bmm", "all"), default="mm"
    )
    parser.add_argument(
        "--jit-heuristic",
        action="store_true",
        help="JIT execution is the default; retained for existing commands",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-json", type=Path, default=VALIDATION_JSON)
    parser.add_argument("--validation-csv", type=Path, default=VALIDATION_CSV)
    parser.add_argument("--output-json", type=Path, default=RAW_JSON)
    parser.add_argument("--output-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--shape-filter")
    parser.add_argument("--max-shapes", type=int)
    parser.add_argument("--max-timing-rounds", type=int, default=5)
    parser.add_argument(
        "--timing", choices=("inference", "cold", "warm"), default="inference"
    )
    parser.add_argument("--cold-cache-mib", type=int, default=0)
    parser.add_argument("--config-timeout-seconds", type=float, default=120)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not args.validate_only:
        benchmark_common.require_consolidated_benchmarks()
    if args.max_timing_rounds < 3 or args.config_timeout_seconds < 0:
        parser.error("rounds must be at least three and timeout must be nonnegative")
    if args.max_shapes is not None and args.max_shapes < 1:
        parser.error("--max-shapes must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard selection")
    if args.cold_cache_mib < 0 or (args.timing == "cold" and not args.cold_cache_mib):
        parser.error("cold timing requires a positive --cold-cache-mib")
    if args.timing == "warm" and args.cold_cache_mib:
        parser.error("warm timing does not flush the cache")

    validations = validate_shapes(load_shapes(args.corpus))
    validation_rows = [validation_row(item) for item in validations]
    write_json(validation_rows, args.validation_json)
    write_rows_csv(validation_rows, args.validation_csv)
    shapes = benchmark_shapes(validations)
    shapes = [
        shape for shape in shapes if args.workload == "all" or shape.op == args.workload
    ]
    if args.shape_filter:
        pattern = re.compile(args.shape_filter)
        shapes = [shape for shape in shapes if pattern.search(shape.shape_id)]
    shapes = [
        shape
        for i, shape in enumerate(shapes)
        if i % args.num_shards == args.shard_index
    ]
    if args.max_shapes is not None:
        shapes = shapes[: args.max_shapes]
    print(
        f"validation: {sum(item.supported for item in validations)}/{len(validations)} supported signatures; {len(shapes)} benchmark shapes"
    )
    if args.validate_only:
        return 0
    if not torch.cuda.is_available():
        parser.error("CUDA is required for benchmarking")
    rows = (
        []
        if args.no_resume or not args.output_json.exists()
        else json.loads(args.output_json.read_text())
    )
    protocol = f"{args.timing}:main-jit-v1:{args.cold_cache_mib}"
    if any(
        row.get("compiler_feature_space") != "main"
        or row.get("timing_protocol") != protocol
        for row in rows
    ):
        parser.error(
            "output uses a different compiler/timing protocol; use a new path or --no-resume"
        )
    for index, shape in enumerate(shapes, 1):
        completed = {
            row["implementation"]
            for row in rows
            if row["shape_id"] == shape.shape_id and row.get("status") == "ok"
        }
        if completed == {"torch", "ayla"}:
            print(f"[shape {index}/{len(shapes)}] resume {shape.shape_id}")
            continue
        print(f"[shape {index}/{len(shapes)}] {shape.shape_id}", flush=True)
        rows = [row for row in rows if row["shape_id"] != shape.shape_id]
        try:
            rows.extend(_measure_isolated(shape, args))
        except Exception as error:
            rows.append(
                {
                    "shape_id": shape.shape_id,
                    "implementation": "ayla",
                    "status": "error",
                    "error": str(error),
                    "compiler_feature_space": "main",
                    "timing_protocol": protocol,
                }
            )
            print(f"skipped: {error}", flush=True)
        write_json(rows, args.output_json)
        write_rows_csv(rows, args.output_csv)
    return 1 if any(row.get("status") == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
