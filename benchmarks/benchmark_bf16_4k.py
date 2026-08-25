#!/usr/bin/env python3
"""Capture a paired, stream-symmetric 4096-square BF16 GEMM benchmark."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

import ayla.gemm_collection  # noqa: F401 -- registers torch.ops.ayla.*


SIZE = 4096
FLOPS = 2 * SIZE**3


def _capture(
    run, launches: int, stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        for _ in range(launches):
            run()
    stream.synchronize()
    return graph


def _time(
    graph: torch.cuda.CUDAGraph,
    launches: int,
    stream: torch.cuda.Stream,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / launches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ayla-op", required=True)
    parser.add_argument("--revision-label", required=True)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--graph-launches", type=int, default=4)
    parser.add_argument("--cache-policy", choices=("hot", "cold"), default="hot")
    parser.add_argument("--cache-flush-multiplier", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.rounds < 3:
        parser.error("--rounds must be at least 3")
    if args.graph_launches < 1:
        parser.error("--graph-launches must be positive")
    if args.cache_policy == "cold" and args.graph_launches != 1:
        parser.error("cold-cache measurements require --graph-launches 1")

    stream = torch.cuda.current_stream()
    torch.manual_seed(0)
    a = torch.randn((SIZE, SIZE), dtype=torch.bfloat16, device="cuda")
    bt = torch.randn_like(a)
    torch_output = torch.empty_like(a)
    ayla_op = getattr(torch.ops.ayla, args.ayla_op)

    def run_ayla():
        return ayla_op(a, bt)

    def run_torch() -> None:
        torch.mm(a, bt.t(), out=torch_output)

    ayla_output = run_ayla()
    run_torch()
    stream.synchronize()
    if not torch.equal(ayla_output, torch_output):
        difference = (ayla_output.float() - torch_output.float()).abs().max()
        raise RuntimeError(f"Ayla result differs from torch.mm; max abs {difference}")

    for _ in range(10):
        run_ayla()
        run_torch()
    stream.synchronize()

    graphs = {
        "ayla": _capture(run_ayla, args.graph_launches, stream),
        "torch": _capture(run_torch, args.graph_launches, stream),
    }
    eviction_buffer = None
    cache_flush_bytes = 0
    if args.cache_policy == "cold":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        l2_bytes = int(getattr(properties, "L2_cache_size", 64 * 1024**2))
        cache_flush_bytes = round(args.cache_flush_multiplier * l2_bytes)
        eviction_buffer = torch.zeros(
            cache_flush_bytes, dtype=torch.uint8, device="cuda",
        )

    samples = {"ayla": [], "torch": []}
    names = ("ayla", "torch")
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else names[::-1]
        for name in order:
            if eviction_buffer is not None:
                with torch.cuda.stream(stream):
                    eviction_buffer.add_(1)
            samples[name].append(_time(graphs[name], args.graph_launches, stream))

    ayla_ms = statistics.median(samples["ayla"])
    torch_ms = statistics.median(samples["torch"])
    ayla_tflops = FLOPS / (ayla_ms / 1e3) / 1e12
    torch_tflops = FLOPS / (torch_ms / 1e3) / 1e12
    row = {
        "revision": args.revision_label,
        "ayla_op": args.ayla_op,
        "protocol": "paired_round_robin_cuda_graph",
        "stream_policy": "shared_current_torch_cuda_stream",
        "cache_policy": args.cache_policy,
        "cache_flush_bytes": cache_flush_bytes,
        "rounds": args.rounds,
        "graph_launches": args.graph_launches,
        "ayla_ms": ayla_ms,
        "ayla_tflops": ayla_tflops,
        "torch_ms": torch_ms,
        "torch_tflops": torch_tflops,
        "ayla_latency_delta_pct": 100 * (ayla_ms / torch_ms - 1),
        "ayla_throughput_delta_pct": 100 * (ayla_tflops / torch_tflops - 1),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
