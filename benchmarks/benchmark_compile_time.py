#!/usr/bin/env python3
"""Measure static Ayla GEMM compilation time by pipeline stage."""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch

from ayla.gemm_collection.gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


def _milliseconds(start: float) -> float:
    return (time.perf_counter() - start) * 1e3


def _compile(m: int, n: int, k: int) -> dict[str, int | float]:
    # TODO: Integrate stage timing into the standard benchmark so compilation
    # time can be measured for every registered kernel, not just this builder.
    start = time.perf_counter()
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m, n=n, k=k,
    )
    build_ms = _milliseconds(start)

    start = time.perf_counter()
    kernel = make_ptx_kernel(module)
    ptx_ms = _milliseconds(start)

    start = time.perf_counter()
    launcher = make_ptx_launcher(kernel)
    launcher.ensure_loaded()
    load_ms = _milliseconds(start)

    return {
        "m": m,
        "n": n,
        "k": k,
        "build_ir_ms": build_ms,
        "lower_ptx_ms": ptx_ms,
        "cuda_jit_load_ms": load_ms,
        "total_ms": build_ms + ptx_ms + load_ms,
    }


def main() -> int:
    # Compilation timing should not silently hit NVIDIA's on-disk PTX cache.
    os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", nargs=3, type=int, action="append", metavar=("M", "N", "K"),
        help="static GEMM shape; may be supplied more than once",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shapes = args.shape or [(4096, 4096, 4096)]
    if any(min(shape) <= 0 for shape in shapes):
        parser.error("shape dimensions must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.cuda.init()
    torch.cuda.synchronize()
    rows = [_compile(*shape) for shape in shapes]

    print("       M        N        K   build IR   lower PTX    CUDA JIT       total")
    for row in rows:
        print(
            f"{row['m']:8d} {row['n']:8d} {row['k']:8d} "
            f"{row['build_ir_ms']:9.2f}ms {row['lower_ptx_ms']:10.2f}ms "
            f"{row['cuda_jit_load_ms']:10.2f}ms {row['total_ms']:10.2f}ms"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRaw results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
