#!/usr/bin/env python3
"""Exhaustively benchmark every divisor-derived store-pipeline grid.

The base grid is 32x16.  For every Rm|32 and Rn|16, the worker builds grid
(32/Rm)x(16/Rn), computes Rm*Rn output tiles per block, verifies the result,
and measures it with the same CUDA-event methodology as
``benchmarks/bench_gemm_collection.py``.  Each shape gets a fresh subprocess
so a JIT/resource failure cannot poison the remaining sweep.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_storepipe import (
    auto_matmul_4096_wgmma256_wg_storepipe_module,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

N = 4096
FLOPS = 2 * N**3
PEAK = 989.5
RMS = (1, 2, 4, 8, 16, 32)
RNS = (1, 2, 4, 8, 16)
WARMUP_S = 0.5
WARMUP_ITERS = 100
TARGET_ROUND_MS = 50
MIN_RUN_TIME_S = 0.5
COOLDOWN_S = 1.0


def bench(run) -> dict[str, float | int]:
    time.sleep(COOLDOWN_S)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); start.record(); run(); end.record(); torch.cuda.synchronize()
    one_ms = max(start.elapsed_time(end), 1e-3)
    iters = max(1, round(TARGET_ROUND_MS / one_ms))
    rounds = max(2, round(MIN_RUN_TIME_S * 1e3 / (iters * one_ms)))
    warmups = max(1, min(WARMUP_ITERS, round(WARMUP_S * 1e3 / one_ms)))
    for _ in range(warmups):
        run()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for s, e in pairs:
        s.record()
        for _ in range(iters):
            run()
        e.record()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) / iters for s, e in pairs]
    return {
        "rounds": rounds, "iters": iters,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples),
        "min_ms": min(samples), "max_ms": max(samples),
    }


def worker(rm: int, rn: int) -> dict:
    piped = rm * rn > 1
    module = auto_matmul_4096_wgmma256_wg_storepipe_module(
        reduction_m=rm, reduction_factor=rn,
        k_stages=2 if piped else 4, store_pipeline=piped,
        kernel_name=f"storepipe_rm{rm}_rn{rn}",
    )
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    torch.manual_seed(1234)
    a = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    launcher(a, b, c); torch.cuda.synchronize()
    ref = a @ b.T
    max_err = (c.float() - ref.float()).abs().max().item()
    if not torch.isfinite(c).all() or max_err > 2.0:
        raise RuntimeError(f"incorrect result: max_err={max_err}")
    timing = bench(lambda: launcher(a, b, c))
    min_ms = float(timing["min_ms"])
    tflops = FLOPS / (min_ms / 1e3) / 1e12
    return {
        "rm": rm, "rn": rn, "tiles_per_block": rm * rn,
        "grid_x": 32 // rm, "grid_y": 16 // rn,
        "blocks": (32 // rm) * (16 // rn),
        "k_stages": 2 if piped else 4, "pipelined": piped,
        "max_err": max_err, **timing,
        "tflops": tflops, "peak_pct": tflops / PEAK * 100,
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", nargs=2, type=int, metavar=("RM", "RN"))
    p.add_argument("--output", type=Path, default=Path("docs/profiling/store_pipeline_grid_sweep.json"))
    args = p.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return
    results = []
    for rm in RMS:
        for rn in RNS:
            cmd = [sys.executable, __file__, "--worker", str(rm), str(rn)]
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.returncode:
                row = {"rm": rm, "rn": rn, "status": "failed", "error": proc.stderr.strip()[-1000:]}
            else:
                row = json.loads(proc.stdout.strip().splitlines()[-1])
            results.append(row)
            if row["status"] == "ok":
                print(f"Rm={rm:2} Rn={rn:2} grid={row['grid_x']:2}x{row['grid_y']:2} "
                      f"median={row['median_ms']:.3f} min={row['min_ms']:.3f}ms "
                      f"peak={row['peak_pct']:.1f}% err={row['max_err']:.3f}", flush=True)
            else:
                print(f"Rm={rm:2} Rn={rn:2} FAILED", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    valid = [r for r in results if r["status"] == "ok"]
    print("\nRanked by median:")
    for r in sorted(valid, key=lambda x: x["median_ms"])[:10]:
        print(f"Rm={r['rm']:2} Rn={r['rn']:2} grid={r['grid_x']:2}x{r['grid_y']:2} "
              f"median={r['median_ms']:.3f}ms min={r['min_ms']:.3f}ms")


if __name__ == "__main__":
    main()
