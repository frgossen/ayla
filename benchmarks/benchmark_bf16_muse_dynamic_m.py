#!/usr/bin/env python3
"""Benchmark Muse projection GEMMs across dynamic token counts."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import ayla.ptx_runtime as runtime
import ayla.torch_custom_call as custom_call


# Import the builder without eagerly registering every 4K collection kernel.
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


M_VALUES = (128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class Workload:
    label: str
    n: int
    k: int


WORKLOADS = {
    "up": Workload("up", 19968, 6656),
    "intermediate": Workload("intermediate", 19968, 19968),
    "down": Workload("down", 6656, 19968),
}

CONFIGS = {
    "one_tile_k3": {
        "output_tiles_per_cta": 1,
        "operand_stages": 3,
    },
    "two_tile_k4_split2": {
        "output_tiles_per_cta": 2,
        "operand_stages": 4,
        "output_slices": 2,
    },
    "four_tile_k3_split2": {
        "output_tiles_per_cta": 4,
        "operand_stages": 3,
        "output_slices": 2,
        "output_store_buffers": 2,
        "operand_move_attrs": {"lhs": {"l2_cache_hint": "evict_first"}},
    },
}


def _default_config(m: int) -> str:
    if m < 256:
        return "one_tile_k3"
    if m < 512:
        return "two_tile_k4_split2"
    return "four_tile_k3_split2"


def _capture(
    run: Callable[[], None], count: int, stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        for _ in range(count):
            run()
    stream.synchronize()
    return graph


def _time(
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


def _confidence_interval(
    values: list[float], repetitions: int = 3000,
) -> tuple[float, float]:
    rng = random.Random(0)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(repetitions)
    )
    return medians[int(.025 * repetitions)], medians[int(.975 * repetitions)]


def _measure(
    workload: Workload,
    m: int,
    config_name: str,
    pairs: int,
    cache_policy: str,
    eviction_buffer: torch.Tensor | None,
    stream: torch.cuda.Stream,
) -> dict[str, object]:
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        problem_m=m,
        problem_n=workload.n,
        problem_k=workload.k,
        **CONFIGS[config_name],
    )
    kernel = make_ptx_kernel(module)
    launcher = _make_launcher(kernel)

    torch.manual_seed(m)
    a = torch.randn((m, workload.k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((workload.n, workload.k), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, workload.n), dtype=torch.bfloat16, device="cuda")
    reference = torch.empty_like(output)

    def run_ayla() -> None:
        launcher(a, b, output)

    def run_torch() -> None:
        torch.mm(a, b.t(), out=reference)

    run_ayla()
    run_torch()
    stream.synchronize()
    difference = output.float() - reference.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        reference.float().square().mean().sqrt()
    )
    if max_abs > 4.0 or relative_rms > 1e-3:
        raise RuntimeError(
            f"{workload.label} M={m}: max abs {max_abs}, relative RMS "
            f"{relative_rms}"
        )

    batch = 1 if cache_policy == "cold" else 4
    for _ in range(5):
        run_ayla()
        run_torch()
    stream.synchronize()
    graphs = {
        "ayla": _capture(run_ayla, batch, stream),
        "torch": _capture(run_torch, batch, stream),
    }
    samples = {"ayla": [], "torch": []}
    names = ("ayla", "torch")
    for pair in range(pairs):
        order = names if pair % 2 == 0 else names[::-1]
        for name in order:
            if eviction_buffer is not None:
                with torch.cuda.stream(stream):
                    eviction_buffer.add_(1)
            samples[name].append(_time(graphs[name], batch, stream))

    ayla_us = statistics.median(samples["ayla"])
    torch_us = statistics.median(samples["torch"])
    latency_deltas = [
        100 * (ayla / torch_time - 1)
        for ayla, torch_time in zip(
            samples["ayla"], samples["torch"], strict=True,
        )
    ]
    ci_low, ci_high = _confidence_interval(latency_deltas)
    flops = 2 * m * workload.n * workload.k
    return {
        "workload": workload.label,
        "m": m,
        "n": workload.n,
        "k": workload.k,
        "config": config_name,
        "protocol": "paired_round_robin_cuda_graph",
        "stream_policy": "shared_current_torch_cuda_stream",
        "cache_policy": cache_policy,
        "batch": batch,
        "pairs": pairs,
        "ayla_us": ayla_us,
        "ayla_useful_tflops": flops / ayla_us / 1e6,
        "torch_us": torch_us,
        "torch_tflops": flops / torch_us / 1e6,
        "ayla_latency_delta_pct": statistics.median(latency_deltas),
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "correctness_max_abs": max_abs,
        "correctness_relative_rms": relative_rms,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _plot(csv_paths: list[Path], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with csv_path.open() as input_file:
            rows.extend(csv.DictReader(input_file))
    workloads = [
        name for name in WORKLOADS
        if any(row["workload"] == name for row in rows)
    ]
    figure, axes = plt.subplots(
        1, len(workloads), figsize=(6.4 * len(workloads), 5.6),
        sharey=True, squeeze=False,
    )
    for axis, workload_name in zip(axes[0], workloads, strict=True):
        selected = sorted(
            (row for row in rows if row["workload"] == workload_name),
            key=lambda row: int(row["m"]),
        )
        ms = [int(row["m"]) for row in selected]
        ayla = [float(row["ayla_useful_tflops"]) for row in selected]
        torch_values = [float(row["torch_tflops"]) for row in selected]
        axis.plot(ms, ayla, marker="o", linewidth=2, label="Ayla")
        axis.plot(ms, torch_values, marker="o", linewidth=2,
                  label="cuBLAS (torch.mm)")
        for index, (m, ayla_value, torch_value) in enumerate(zip(
            ms, ayla, torch_values, strict=True,
        )):
            difference = ayla_value - torch_value
            advantage = 100 * difference / torch_value
            axis.annotate(
                f"{advantage:+.1f}%\n{difference:+.1f} TFLOP/s",
                (m, max(ayla_value, torch_value)),
                xytext=(0, 7 + 12 * (index % 2)),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.5,
                color="#147d3f" if advantage >= 0 else "#b3261e",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(ms, [f"{m:,}" for m in ms], rotation=45, ha="right")
        workload = WORKLOADS[workload_name]
        axis.set_title(
            f"{workload_name.capitalize()}: N={workload.n:,}, K={workload.k:,}"
        )
        axis.set_xlabel("M = flattened tokens (B×S)")
        axis.grid(True, alpha=.3)
    axes[0][0].set_ylabel("Useful TFLOP/s")
    axes[0][-1].legend()
    figure.suptitle("H100 BF16 Muse projections — Ayla advantage vs cuBLAS")
    figure.tight_layout(rect=(0, 0, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180 if output.suffix.lower() == ".png" else None)
    if output.suffix.lower() != ".svg":
        figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("benchmark", "plot"), default="benchmark",
    )
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS,
                        default=list(WORKLOADS))
    parser.add_argument("--ms", nargs="+", type=int, default=list(M_VALUES))
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--cache-policy", choices=("cold", "hot"), default="cold")
    parser.add_argument("--cache-flush-multiplier", type=float, default=4.0)
    parser.add_argument("--input-csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "plot":
        _plot(args.input_csv or [args.output], args.output)
        return 0
    if any(m not in M_VALUES for m in args.ms):
        parser.error(f"--ms must be selected from {M_VALUES}")

    eviction_buffer = None
    if args.cache_policy == "cold":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        l2_bytes = int(getattr(properties, "L2_cache_size", 64 * 1024**2))
        eviction_buffer = torch.zeros(
            round(args.cache_flush_multiplier * l2_bytes),
            dtype=torch.uint8,
            device="cuda",
        )

    stream = torch.cuda.current_stream()
    rows = [
        _measure(
            WORKLOADS[workload_name], m, _default_config(m), args.pairs,
            args.cache_policy, eviction_buffer, stream,
        )
        for workload_name in args.workloads
        for m in args.ms
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
