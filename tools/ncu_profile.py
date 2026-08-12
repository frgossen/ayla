#!/usr/bin/env python3
"""Single-launch driver for profiling one GEMM under Nsight Compute (ncu).

ncu replays each *kernel launch* it captures, so the Python side just needs to
warm up (build the JIT launcher, ramp clocks) and then fire the target op a
handful of times. Point ncu at this script and use --launch-skip/--launch-count
to profile only the post-warmup launches.

    op=wgmma256 -> torch.ops.ayla.auto_matmul_4096_wgmma256_wg (best ayla WGMMA)
    op=auto     -> torch.ops.ayla.auto_matmul_4096  (wgmma128 ayla kernel)
    op=torch    -> torch.mm(a, b.T)                 (matched cuBLAS baseline)
    op=tma-store-k3-4tiles-split2 -> current 4-tile Ayla kernel

Usage (run under ncu, see the commands printed by this repo's docs):
    python tools/ncu_profile.py --op auto --warmup 5 --profile 1
"""
from __future__ import annotations

import argparse
import os
import sys

# load_inline shells out to `ninja`, which lives next to this interpreter.
_BINDIR = os.path.dirname(sys.executable)
if _BINDIR and _BINDIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _BINDIR + os.pathsep + os.environ.get("PATH", "")

import torch

import ayla.gemm_collection  # noqa: F401 — registers torch.ops.ayla.*

M = N = K = 4096
DTYPE = torch.bfloat16


def get_op(name: str):
    if name == "wgmma256":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg(a, b)
    if name == "tma-store-k3-2tiles":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles(a, b)
    if name == "tma-store-k3-4tiles-split2":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2(a, b)
    if name == "tma-store-k4-split2-2tiles":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k4_2tiles_split2(a, b)
    if name == "tma-store-k3-cluster1x2":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_cluster1x2(a, b)
    if name == "auto":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096(a, b)
    if name == "torch":
        return lambda a, b: torch.mm(a, b.t())
    raise SystemExit(f"unknown op: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=[
        "wgmma256", "tma-store-k3-2tiles", "tma-store-k3-4tiles-split2",
        "tma-store-k4-split2-2tiles", "tma-store-k3-cluster1x2",
        "auto", "torch",
    ], required=True)
    p.add_argument("--warmup", type=int, default=5,
                   help="launches before the profiled region (ncu --launch-skip)")
    p.add_argument("--profile", type=int, default=1,
                   help="launches to profile (ncu --launch-count)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=DTYPE, device="cuda")
    # Every op consumes the same row-major [N, K] B and computes A @ B.T.
    b = torch.randn(N, K, dtype=DTYPE, device="cuda")

    op = get_op(args.op)

    # Warm up: builds the JIT launcher on first call and ramps clocks. ncu skips
    # these with --launch-skip so only steady-state launches are measured.
    for _ in range(args.warmup):
        op(a, b)
    torch.cuda.synchronize()

    for _ in range(args.profile):
        op(a, b)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
