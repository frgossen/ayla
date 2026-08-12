"""Measure the per-k-stage-boundary overhead at a locked clock.

The stage-boundary hypothesis says the residual gap to peak is a fixed bubble
paid once per k-tile boundary (drain -> barrier -> TMA issue -> next group),
not anything proportional to the tensor work.  If that is right, halving the
number of boundaries by doubling `operand_k_tile` should halve the gap.

Run at a locked clock so "% of peak" means something:

    sudo nvidia-smi -i 0 -lgc 1350
    CUDA_VISIBLE_DEVICES=0 .pixi/envs/default/bin/python \
        tools/ptx_experiments/boundary_budget.py
"""
import statistics
import subprocess

import torch

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as M,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

N = 4096
FLOP = 2 * N ** 3
SMS = 132
# BF16 dense tensor throughput per SM per clock on GH100, from the 989.4
# TFLOP/s @ 1755 MHz / 132 SM datasheet point.
FLOP_PER_SM_CLK = 4096

CONFIGS = {
    # name: (kwargs, output tiles per CTA, k elements per stage)
    "4tile k64  s3 (current best)": (
        dict(output_tiles_per_cta=4, operand_stages=3, output_slices=2,
             output_tma_inner_padding=16, output_store_buffers=2,
             operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}}),
        4, 64),
    "2tile k64  s3": (
        dict(output_tiles_per_cta=2, operand_stages=3, output_slices=2,
             output_tma_inner_padding=16, output_store_buffers=2),
        2, 64),
    "2tile k128 s2 (half the boundaries)": (
        dict(output_tiles_per_cta=2, operand_stages=2, operand_k_tile=128,
             output_slices=2),
        2, 128),
    "2tile k64  s4": (
        dict(output_tiles_per_cta=2, operand_stages=4, output_slices=2),
        2, 64),
}


def sm_clock_mhz():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.splitlines()[0]
    return float(out)


def time_us(fn, a, b, c, batches=30, iters=20):
    for _ in range(100):
        fn(a, b, c)
    torch.cuda.synchronize()
    out = []
    for _ in range(batches):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn(a, b, c)
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters * 1e3)
    return statistics.median(out)


def main():
    torch.manual_seed(0)
    a = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(N, N, dtype=torch.bfloat16, device="cuda")

    clk = sm_clock_mhz()
    peak = SMS * FLOP_PER_SM_CLK * clk * 1e6 / 1e12
    # Tensor-busy cycles the SM on the critical path must spend regardless of
    # scheduling: total MACs / per-SM rate, spread over the CTAs it runs.
    print(f"SM clock {clk:.0f} MHz -> clock-scaled BF16 peak {peak:.1f} TFLOP/s\n")
    print(f"{'config':38s} {'us':>8s} {'TFLOP/s':>8s} {'%peak':>6s} "
          f"{'bnd/SM':>7s} {'dead cyc/bnd':>12s}")

    rows = []
    for name, (cfg, tiles, ktile) in CONFIGS.items():
        launcher = make_ptx_launcher(make_ptx_kernel(M(**cfg)))
        us = time_us(launcher, a, b, c)
        tflops = FLOP / (us * 1e-6) / 1e12
        # Waves of CTAs, and k-boundaries on the critical path per SM.
        ctas = (N // 128 // tiles) * (N // 256)
        waves = -(-ctas // SMS)
        boundaries = waves * tiles * (N // ktile)
        ideal_cyc = waves * tiles * (128 * 256 * N) * 2 / FLOP_PER_SM_CLK
        actual_cyc = us * 1e-6 * clk * 1e6
        dead = (actual_cyc - ideal_cyc) / boundaries
        rows.append((name, us, tflops, 100 * tflops / peak, boundaries, dead))
        print(f"{name:38s} {us:8.2f} {tflops:8.1f} {100*tflops/peak:5.1f}% "
              f"{boundaries:7d} {dead:12.1f}")

    def torch_mm(a, b, c):
        torch.mm(a, b.t(), out=c)
    us = time_us(torch_mm, a, b, c)
    t = FLOP / (us * 1e-6) / 1e12
    print(f"{'cuBLAS torch.mm(A, B.T)':38s} {us:8.2f} {t:8.1f} "
          f"{100*t/peak:5.1f}%")


if __name__ == "__main__":
    main()
