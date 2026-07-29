#!/usr/bin/env python3
"""Single-launch driver for profiling one GEMM under Nsight Compute (ncu).

ncu replays each *kernel launch* it captures, so the Python side just needs to
warm up (build the JIT launcher, ramp clocks) and then fire the target op a
handful of times. Point ncu at this script and use --launch-skip/--launch-count
to profile only the post-warmup launches.

    op=wgmma256 -> torch.ops.ayla.auto_matmul_4096_wgmma256_wg (best ayla WGMMA)
    op=auto     -> torch.ops.ayla.auto_matmul_4096  (wgmma128 ayla kernel)
    op=torch    -> torch.mm                         (cuBLAS baseline)

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
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_storepipe import (
    auto_matmul_4096_wgmma256_wg_storepipe_module,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

M = N = K = 4096
DTYPE = torch.bfloat16


def get_op(name: str):
    if name == "wgmma256":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg(a, b)
    if name == "storepipe":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096_wgmma256_wg_storepipe(a, b)
    if name == "auto":
        return lambda a, b: torch.ops.ayla.auto_matmul_4096(a, b)
    if name == "torch":
        return torch.mm
    raise SystemExit(f"unknown op: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=["wgmma256", "storepipe", "auto", "torch", "storepipe-sweep"], required=True)
    p.add_argument("--warmup", type=int, default=5,
                   help="launches before the profiled region (ncu --launch-skip)")
    p.add_argument("--profile", type=int, default=1,
                   help="launches to profile (ncu --launch-count)")
    p.add_argument("--shapes", type=str, default="1x1,1x2,1x4,2x2",
                   help="storepipe-sweep: comma list of RmxRn (e.g. 1x1,1x2,2x2)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=DTYPE, device="cuda")
    b = torch.randn(K, N, dtype=DTYPE, device="cuda")  # torch.mm wants KxN

    if args.op == "storepipe-sweep":
        # Build one launcher per RmxRn shape, warm all, then fire each once.
        # Run ncu with -k regex:auto_matmul so only these launches count toward
        # -s/-c: -s (warmup * n_shapes) -c n_shapes profiles one steady launch
        # of each. Identify shapes in ncu output by grid dim [32/Rm, 16/Rn, 1].
        c = torch.empty(M, N, dtype=DTYPE, device="cuda")
        shapes = [
            tuple(int(x) for x in tok.lower().split("x"))
            for tok in args.shapes.split(",")
        ]
        launchers = [
            (rm, rn, make_ptx_launcher(make_ptx_kernel(
                auto_matmul_4096_wgmma256_wg_storepipe_module(
                    reduction_factor=rn, reduction_m=rm))))
            for rm, rn in shapes
        ]
        for _ in range(args.warmup):
            for _, _, L in launchers:
                L(a, b, c)
        torch.cuda.synchronize()
        for rm, rn, L in launchers:
            L(a, b, c)  # one profiled launch each; grid=[32/rm,16/rn,1]
        torch.cuda.synchronize()
        return

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
