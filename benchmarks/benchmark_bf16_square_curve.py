#!/usr/bin/env python3
"""Benchmark retained Ayla square BF16 GEMMs against cuBLAS."""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

import ayla.ptx_runtime as runtime
import ayla.torch_custom_call as custom_call


_make_launcher = runtime.make_ptx_launcher
_make_custom_call = custom_call.make_torch_custom_call
runtime.make_ptx_launcher = lambda kernel, **kwargs: None
custom_call.make_torch_custom_call = lambda *args, **kwargs: None
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
runtime.make_ptx_launcher = _make_launcher
custom_call.make_torch_custom_call = _make_custom_call

from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402


def _square(size: int, **config):
    return lambda: auto_matmul_4096_wgmma256_wg_tma_store_module(
        problem_size=size, **config,
    )


CONFIGS = {
    512: (_square(512, operand_stages=2), "m128n256_k64_s2"),
    1024: (
        _square(1024, output_tiles_per_cta=2, operand_stages=4,
                output_slices=2),
        "m2x128n256_k64_s4_split2",
    ),
    2048: (
        _square(2048, output_tiles_per_cta=4, operand_stages=3,
                output_slices=2, output_store_buffers=2,
                operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}}),
        "m4x128n256_k64_s3_split2",
    ),
    4096: (
        _square(4096, output_tiles_per_cta=4, operand_stages=3,
                output_slices=2, output_store_buffers=2,
                operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}}),
        "four_stage_split_output",
    ),
    8192: (
        _square(8192, output_tiles_per_cta=4, operand_stages=4,
                output_slices=2),
        "m4x128n256_k64_s4_split2",
    ),
    16384: (
        _square(16384, output_tiles_per_cta=4, operand_stages=4,
                output_slices=2),
        "m4x128n256_k64_s4_split2",
    ),
}


def _capture_graph(
    run, count: int, stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        for _ in range(count):
            run()
    stream.synchronize()
    return graph


def _graph_batch(
    graph: torch.cuda.CUDAGraph, count: int, stream: torch.cuda.Stream,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / count


def _cache_flush_bytes(args: argparse.Namespace) -> int:
    if args.cache_policy != "cold":
        return 0
    if args.cache_flush_mb is not None:
        return round(args.cache_flush_mb * 1024**2)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(properties, "L2_cache_size", 64 * 1024**2))
    return round(args.cache_flush_multiplier * l2_bytes)


def _l2_bytes() -> int:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(getattr(properties, "L2_cache_size", 64 * 1024**2))


def _flush_l2(eviction_buffer: torch.Tensor) -> None:
    eviction_buffer.add_(1)


def _ci(values: list[float], repetitions: int = 5000) -> tuple[float, float]:
    rng = random.Random(0)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(repetitions)
    )
    return medians[int(.025 * repetitions)], medians[int(.975 * repetitions)]


def _write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_isolated(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="square-curve-") as temp_dir:
        for size in args.sizes:
            output = Path(temp_dir) / f"{size}.csv"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--sizes", str(size),
                "--rounds", str(args.rounds),
                "--target-batch-ms", str(args.target_batch_ms),
                "--cache-policy", args.cache_policy,
                "--cache-flush-multiplier", str(args.cache_flush_multiplier),
                "--correctness-seeds",
                *[str(seed) for seed in args.correctness_seeds],
                "--output", str(output),
            ]
            if args.cache_flush_mb is not None:
                command.extend(["--cache-flush-mb", str(args.cache_flush_mb)])
            if args.rotating_inputs is not None:
                command.extend(["--rotating-inputs", str(args.rotating_inputs)])
            subprocess.run(command, check=True)
            with output.open() as input_file:
                rows.extend(csv.DictReader(input_file))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=sorted(CONFIGS),
        choices=sorted(CONFIGS),
    )
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--target-batch-ms", type=float, default=10.0)
    parser.add_argument(
        "--cache-policy", choices=("hot", "weight-hot", "cold"),
        default="hot",
        help=(
            "hot replays fixed inputs; weight-hot keeps the weight fixed while "
            "rotating activation/output buffers; cold evicts L2 before every "
            "timed single-GEMM graph replay"
        ),
    )
    parser.add_argument(
        "--rotating-inputs", type=int,
        help=(
            "activation/output working sets in weight-hot mode; by default "
            "choose enough distinct storage to exceed 2x L2"
        ),
    )
    parser.add_argument(
        "--cache-flush-multiplier", type=float, default=4.0,
        help="cold-mode eviction-buffer size as a multiple of reported L2",
    )
    parser.add_argument(
        "--cache-flush-mb", type=float,
        help="override the cold-mode eviction-buffer size in MiB",
    )
    parser.add_argument("--correctness-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--isolate-shapes", action="store_true",
        help="run each shape in a fresh process",
    )
    args = parser.parse_args()
    if args.cache_policy == "cold" and args.cache_flush_multiplier <= 1:
        parser.error("--cache-flush-multiplier must be greater than 1")
    if args.rotating_inputs is not None and args.rotating_inputs <= 1:
        parser.error("--rotating-inputs must be greater than 1")
    if (
        args.cache_policy == "cold"
        and args.cache_flush_mb is not None
        and args.cache_flush_mb <= 0
    ):
        parser.error("--cache-flush-mb must be positive")

    if args.isolate_shapes and len(args.sizes) > 1:
        rows = _run_isolated(args)
        _write_rows(rows, args.output)
        print(f"wrote {args.output}", flush=True)
        return 0

    rows: list[dict[str, object]] = []
    benchmark_stream = torch.cuda.current_stream()
    for size in args.sizes:
        build, config = CONFIGS[size]
        print(f"size={size}: build {config}", flush=True)
        kernel = make_ptx_kernel(build())
        launcher = _make_launcher(kernel)

        torch.manual_seed(args.correctness_seeds[0])
        a = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")
        l2_bytes = _l2_bytes()
        matrix_bytes = size * size * torch.bfloat16.itemsize
        if args.cache_policy == "weight-hot":
            input_count = args.rotating_inputs or max(
                2, math.ceil(2 * l2_bytes / matrix_bytes),
            )
            activation_pool_bytes = input_count * matrix_bytes
            output_pool_bytes = input_count * matrix_bytes
            rotating_working_set_bytes = 2 * input_count * matrix_bytes
            if activation_pool_bytes < 2 * l2_bytes:
                raise ValueError(
                    f"{input_count} rotating inputs only provide "
                    f"{activation_pool_bytes / 1024**2:.1f} MiB of "
                    f"activation storage; at least 2x the "
                    f"{l2_bytes / 1024**2:.1f} MiB L2 is required"
                )
        else:
            input_count = 1
            activation_pool_bytes = 0
            output_pool_bytes = 0
            rotating_working_set_bytes = 0
        inputs = [a] + [torch.randn_like(a) for _ in range(input_count - 1)]
        weight = torch.randn_like(a)
        weight_t = weight.t()
        outputs = {
            name: [torch.empty_like(a) for _ in range(input_count)]
            for name in ("ayla", "torch")
        }

        def run_ayla(slot: int) -> None:
            launcher(inputs[slot], weight, outputs["ayla"][slot])

        def run_torch(slot: int) -> None:
            torch.mm(inputs[slot], weight_t, out=outputs["torch"][slot])

        slot_runs = {
            "ayla": run_ayla,
            "torch": run_torch,
        }
        positions = {name: 0 for name in slot_runs}

        def rotating_run(name: str):
            def run() -> None:
                slot = positions[name] % input_count
                positions[name] += 1
                slot_runs[name](slot)
            return run

        runs = {name: rotating_run(name) for name in slot_runs}
        for seed_index, seed in enumerate(args.correctness_seeds):
            if seed_index:
                torch.manual_seed(seed)
                a.normal_()
                weight.normal_()
            for run in slot_runs.values():
                run(0)
            torch.cuda.synchronize()
            if not torch.equal(outputs["ayla"][0], outputs["torch"][0]):
                mismatches = int(
                    (outputs["ayla"][0] != outputs["torch"][0]).sum()
                )
                raise RuntimeError(
                    f"size {size}, seed {seed}: Ayla has {mismatches} mismatches"
                )

        if len(args.correctness_seeds) > 1:
            torch.manual_seed(args.correctness_seeds[0])
            for input_tensor in inputs:
                input_tensor.normal_()
            weight.normal_()
        warmup_cycles = 3 if input_count == 1 else input_count * 3
        for _ in range(warmup_cycles):
            for run in runs.values():
                run()
        torch.cuda.synchronize()

        cache_flush_bytes = _cache_flush_bytes(args)
        eviction_buffer = None
        if cache_flush_bytes:
            eviction_buffer = torch.zeros(
                cache_flush_bytes, dtype=torch.uint8, device="cuda",
            )
            torch.cuda.synchronize()
            batch = 1
        else:
            pilot_count = 8
            positions = {name: 0 for name in runs}
            pilot_graphs = {
                name: _capture_graph(run, pilot_count, benchmark_stream)
                for name, run in runs.items()
            }
            pilot_times = {
                name: statistics.median(
                    _graph_batch(graph, pilot_count, benchmark_stream)
                    for _ in range(3)
                )
                for name, graph in pilot_graphs.items()
            }
            batch = max(
                1,
                min(
                    4000,
                    round(args.target_batch_ms * 1e3 / max(pilot_times.values())),
                ),
            )
            del pilot_graphs
        positions = {name: 0 for name in runs}
        graphs = {
            name: _capture_graph(run, batch, benchmark_stream)
            for name, run in runs.items()
        }
        names = ("ayla", "torch")
        for _ in range(10):
            for name in names:
                if eviction_buffer is not None:
                    _flush_l2(eviction_buffer)
                graphs[name].replay()
        torch.cuda.synchronize()

        samples = {name: [] for name in runs}
        for round_index in range(args.rounds):
            offset = round_index % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                if eviction_buffer is not None:
                    _flush_l2(eviction_buffer)
                samples[name].append(
                    _graph_batch(graphs[name], batch, benchmark_stream)
                )

        ayla_deltas = [
            100 * (ayla_us / torch_us - 1)
            for ayla_us, torch_us in zip(samples["ayla"], samples["torch"])
        ]
        ayla_low, ayla_high = _ci(ayla_deltas)
        medians = {
            name: statistics.median(values) for name, values in samples.items()
        }
        ayla_median_delta_pct = 100 * (
            medians["ayla"] / medians["torch"] - 1
        )
        flops = 2 * size**3
        row = {
            "size": size,
            "config": config,
            "protocol": (
                "single_cuda_graph_l2_evicted_round_robin"
                if eviction_buffer is not None else (
                    "batched_cuda_graph_weight_stationary_activation_rotation"
                    if args.cache_policy == "weight-hot"
                    else "batched_cuda_graph_round_robin"
                )
            ),
            "stream_policy": "shared_current_torch_cuda_stream",
            "input_policy": (
                "fixed_seed_cache_cold_l2_evicted"
                if eviction_buffer is not None
                else (
                    f"fixed_weight_rotating_{input_count}_activations_outputs"
                    if args.cache_policy == "weight-hot"
                    else "fixed_seed_cache_hot"
                )
            ),
            "rotating_inputs": input_count,
            "l2_bytes": l2_bytes,
            "activation_pool_bytes": activation_pool_bytes,
            "output_pool_bytes": output_pool_bytes,
            "rotating_working_set_bytes": rotating_working_set_bytes,
            "activation_pool_l2_ratio": (
                activation_pool_bytes / l2_bytes
            ),
            "cache_flush_bytes": cache_flush_bytes,
            "correctness_seeds": ";".join(map(str, args.correctness_seeds)),
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "grid_x": kernel.grid_dim[0],
            "grid_y": kernel.grid_dim[1],
            "grid_z": kernel.grid_dim[2],
            "block_x": kernel.block_dim[0],
            "shared_bytes": kernel.shared_mem_bytes,
            "batch": batch,
            "rounds": args.rounds,
            "ayla_us": medians["ayla"],
            "ayla_tflops": flops / medians["ayla"] / 1e6,
            "torch_us": medians["torch"],
            "torch_tflops": flops / medians["torch"] / 1e6,
            "ayla_delta_pct": statistics.median(ayla_deltas),
            "ayla_median_delta_pct": ayla_median_delta_pct,
            "ayla_ci_low_pct": ayla_low,
            "ayla_ci_high_pct": ayla_high,
            "ayla_wins": sum(delta < 0 for delta in ayla_deltas),
        }
        rows.append(row)
        print(row, flush=True)

        del a, inputs, weight, weight_t, outputs, launcher, kernel, graphs
        del eviction_buffer
        torch.cuda.empty_cache()

    _write_rows(rows, args.output)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
