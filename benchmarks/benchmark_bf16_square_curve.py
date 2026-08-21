#!/usr/bin/env python3
"""Benchmark retained square BF16 GEMMs against torch/cuBLAS."""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

import ayla.ptx_runtime as runtime
import ayla.torch_custom_call as custom_call


_make_launcher = runtime.make_ptx_launcher
_make_custom_call = custom_call.make_torch_custom_call
runtime.make_ptx_launcher = lambda kernel, **kwargs: None
custom_call.make_torch_custom_call = lambda *args, **kwargs: None
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (  # noqa: E402
    BEST_AUTO_MATMUL_1024_PTX_JIT_OPTIONS,
    BEST_AUTO_MATMUL_16384_PTX_JIT_OPTIONS,
    BEST_AUTO_MATMUL_2048_PTX_JIT_OPTIONS,
    BEST_AUTO_MATMUL_4096_PTX_JIT_OPTIONS,
    BEST_AUTO_MATMUL_512_PTX_JIT_OPTIONS,
    BEST_AUTO_MATMUL_8192_PTX_JIT_OPTIONS,
    best_auto_matmul_1024_wgmma256_wg_tma_store_module,
    best_auto_matmul_16384_wgmma256_wg_tma_store_module,
    best_auto_matmul_2048_wgmma256_wg_tma_store_module,
    best_auto_matmul_4096_wgmma256_wg_tma_store_module,
    best_auto_matmul_512_wgmma256_wg_tma_store_module,
    best_auto_matmul_8192_wgmma256_wg_tma_store_module,
)
runtime.make_ptx_launcher = _make_launcher
custom_call.make_torch_custom_call = _make_custom_call

from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402


CONFIGS = {
    512: (
        best_auto_matmul_512_wgmma256_wg_tma_store_module,
        BEST_AUTO_MATMUL_512_PTX_JIT_OPTIONS,
        "m64n64_k256_s2",
    ),
    1024: (
        best_auto_matmul_1024_wgmma256_wg_tma_store_module,
        BEST_AUTO_MATMUL_1024_PTX_JIT_OPTIONS,
        "m128n64_k128_s4_group4_start0123",
    ),
    2048: (
        best_auto_matmul_2048_wgmma256_wg_tma_store_module,
        BEST_AUTO_MATMUL_2048_PTX_JIT_OPTIONS,
        "m128n128_k64_s2_split2_group8_3d",
    ),
    4096: (
        best_auto_matmul_4096_wgmma256_wg_tma_store_module,
        BEST_AUTO_MATMUL_4096_PTX_JIT_OPTIONS,
        "m4x128n256_k64_s3",
    ),
    8192: (
        best_auto_matmul_8192_wgmma256_wg_tma_store_module,
        BEST_AUTO_MATMUL_8192_PTX_JIT_OPTIONS,
        "m4x128n256_k64_s4",
    ),
    16384: (
        best_auto_matmul_16384_wgmma256_wg_tma_store_module,
        BEST_AUTO_MATMUL_16384_PTX_JIT_OPTIONS,
        "m256n192_k64_s3",
    ),
}


def _capture_graph(run, count: int) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            run()
    torch.cuda.synchronize()
    return graph


def _graph_batch(graph: torch.cuda.CUDAGraph, count: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / count


def _ci(values: list[float], repetitions: int = 5000) -> tuple[float, float]:
    rng = random.Random(0)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(repetitions)
    )
    return medians[int(.025 * repetitions)], medians[int(.975 * repetitions)]


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    sizes = [int(row["size"]) for row in rows]
    ayla = [float(row["ayla_tflops"]) for row in rows]
    torch = [float(row["torch_tflops"]) for row in rows]
    fig, throughput = plt.subplots(figsize=(6.5, 4.5))
    throughput.plot(sizes, ayla, marker="o", linewidth=2, label="Ayla")
    throughput.plot(sizes, torch, marker="o", linewidth=2, label="cuBLAS")
    for size, ayla_tflops, torch_tflops, row in zip(
        sizes, ayla, torch, rows,
    ):
        latency_delta = 100 * (
            float(row["ayla_us"]) / float(row["torch_us"]) - 1
        )
        throughput_delta = ayla_tflops - torch_tflops
        throughput.annotate(
            f"{latency_delta:+.2f}%\n{throughput_delta:+.1f} TFLOP/s",
            (size, ayla_tflops),
            xytext=(0, 8 if latency_delta < 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1f4e79" if latency_delta < 0 else "#a61c00",
        )
    throughput.set_xscale("log", base=2)
    throughput.set_xticks(sizes, [str(size) for size in sizes])
    throughput.set_xlabel("Square matrix size")
    throughput.grid(True, alpha=.3)
    throughput.legend()
    throughput.set_ylabel("TFLOP/s")
    throughput.set_title("H100 square BF16 GEMM — graph-batched throughput")
    throughput.margins(y=.16)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=sorted(CONFIGS),
        choices=sorted(CONFIGS),
    )
    parser.add_argument("--pairs", type=int, default=80)
    parser.add_argument("--target-batch-ms", type=float, default=10.0)
    parser.add_argument("--correctness-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for size_index, size in enumerate(args.sizes):
        build, jit_options, config = CONFIGS[size]
        print(f"size={size}: build {config}", flush=True)
        kernel = make_ptx_kernel(build())
        launcher = _make_launcher(kernel, jit_options=jit_options)

        torch.manual_seed(args.correctness_seeds[0])
        a = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")
        b = torch.randn_like(a)
        bt = b.t()
        ayla_output = torch.empty_like(a)
        torch_output = torch.empty_like(a)

        def run_ayla() -> None:
            launcher(a, b, ayla_output)

        def run_torch() -> None:
            torch.mm(a, bt, out=torch_output)

        for seed_index, seed in enumerate(args.correctness_seeds):
            if seed_index:
                torch.manual_seed(seed)
                a.normal_()
                b.normal_()
            run_ayla()
            run_torch()
            torch.cuda.synchronize()
            if not torch.equal(ayla_output, torch_output):
                raise RuntimeError(
                    f"size {size}, seed {seed}: "
                    f"{int((ayla_output != torch_output).sum())} mismatches"
                )

        # The first seed is the fixed-input hot-cache timing workload.
        if len(args.correctness_seeds) > 1:
            torch.manual_seed(args.correctness_seeds[0])
            a.normal_()
            b.normal_()
        for _ in range(3):
            run_ayla()
            run_torch()
        torch.cuda.synchronize()

        pilot_count = 8
        ayla_pilot_graph = _capture_graph(run_ayla, pilot_count)
        torch_pilot_graph = _capture_graph(run_torch, pilot_count)
        ayla_single = statistics.median(
            _graph_batch(ayla_pilot_graph, pilot_count) for _ in range(3)
        )
        torch_single = statistics.median(
            _graph_batch(torch_pilot_graph, pilot_count) for _ in range(3)
        )
        batch = max(
            1,
            min(4000, round(args.target_batch_ms * 1e3 / max(
                ayla_single, torch_single,
            ))),
        )
        del ayla_pilot_graph, torch_pilot_graph
        ayla_graph = _capture_graph(run_ayla, batch)
        torch_graph = _capture_graph(run_torch, batch)
        for _ in range(10):
            for graph in (ayla_graph, torch_graph):
                graph.replay()
            torch.cuda.synchronize()

        ayla_times: list[float] = []
        torch_times: list[float] = []
        for pair in range(args.pairs):
            ayla_first = bool((pair + size_index) % 2)
            order = (
                ((ayla_graph, ayla_times), (torch_graph, torch_times))
                if ayla_first else
                ((torch_graph, torch_times), (ayla_graph, ayla_times))
            )
            for graph, samples in order:
                samples.append(_graph_batch(graph, batch))

        deltas = [
            100 * (ayla_us / torch_us - 1)
            for ayla_us, torch_us in zip(ayla_times, torch_times)
        ]
        low, high = _ci(deltas)
        ayla_us = statistics.median(ayla_times)
        torch_us = statistics.median(torch_times)
        flops = 2 * size**3
        attrs = launcher.function_attributes
        row = {
            "size": size,
            "config": config,
            "protocol": "batched_cuda_graph",
            "input_policy": "fixed_seed_cache_hot",
            "correctness_seeds": ";".join(map(str, args.correctness_seeds)),
            "reference": "torch.mm",
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "jit_optimization": jit_options.optimization_level,
            "grid_x": kernel.grid_dim[0],
            "grid_y": kernel.grid_dim[1],
            "grid_z": kernel.grid_dim[2],
            "block_x": kernel.block_dim[0],
            "registers": attrs["num_regs"],
            "shared_bytes": attrs["shared_size_bytes"],
            "batch": batch,
            "pairs": args.pairs,
            "ayla_us": ayla_us,
            "ayla_tflops": flops / ayla_us / 1e6,
            "torch_us": torch_us,
            "torch_tflops": flops / torch_us / 1e6,
            "ayla_delta_pct": statistics.median(deltas),
            "ci_low_pct": low,
            "ci_high_pct": high,
            "ayla_wins": sum(delta < 0 for delta in deltas),
        }
        rows.append(row)
        print(row, flush=True)

        del a, b, bt, ayla_output, torch_output, launcher, kernel
        del ayla_graph, torch_graph
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot is not None:
        _plot(rows, args.plot)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
