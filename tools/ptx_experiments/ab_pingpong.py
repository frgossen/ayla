"""Correctness and paired benchmark for the asymmetric WG ping-pong prototype."""
from __future__ import annotations

import dataclasses
import os
import random
import statistics

import torch

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as M,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from pingpong_named_barrier import transform


BEST = dict(
    output_tiles_per_cta=4, operand_stages=3, output_slices=2,
    output_tma_inner_padding=16, output_store_buffers=2,
    operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
)
FLOP = 2 * 4096**3


def timed_batch(fn, a, b, c, launches=20):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(launches):
        fn(a, b, c)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / launches * 1e3


def bootstrap_ci(values, samples=4000):
    rng = random.Random(0)
    estimates = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return estimates[samples // 40], estimates[-samples // 40]


def main():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    cb = torch.empty_like(a)
    cv = torch.empty_like(a)

    kernel = make_ptx_kernel(M(**BEST))
    ptx, changed = transform(kernel.ptx, verbose=True)
    if changed != 25:
        raise RuntimeError(f"expected 25 operand release sites, changed {changed}")
    baseline = make_ptx_launcher(kernel)
    variant = make_ptx_launcher(dataclasses.replace(kernel, ptx=ptx))

    baseline(a, b, cb)
    variant(a, b, cv)
    torch.cuda.synchronize()
    mismatches = int((cb != cv).sum())
    print(f"correctness: {mismatches} mismatches vs baseline")
    if mismatches:
        raise RuntimeError("ping-pong prototype is not bitwise correct")

    ramp = int(os.environ.get("PINGPONG_RAMP", "800"))
    for i in range(ramp):
        (baseline if i % 2 == 0 else variant)(a, b, cb if i % 2 == 0 else cv)
    torch.cuda.synchronize()

    tb, tv = [], []
    batches = int(os.environ.get("PINGPONG_BATCHES", "80"))
    for i in range(batches):
        if i % 2 == 0:
            tb.append(timed_batch(baseline, a, b, cb))
            tv.append(timed_batch(variant, a, b, cv))
        else:
            tv.append(timed_batch(variant, a, b, cv))
            tb.append(timed_batch(baseline, a, b, cb))
    deltas = [100 * (v - base) / base for base, v in zip(tb, tv)]
    lo, hi = bootstrap_ci(deltas)
    base_us, variant_us = statistics.median(tb), statistics.median(tv)
    print(f"baseline:  {base_us:.3f} us  {FLOP / (base_us * 1e-6) / 1e12:.1f} TFLOP/s")
    print(f"ping-pong: {variant_us:.3f} us  {FLOP / (variant_us * 1e-6) / 1e12:.1f} TFLOP/s")
    print(f"paired delta: {statistics.median(deltas):+.3f}% "
          f"95% CI [{lo:+.3f}%, {hi:+.3f}%]")


if __name__ == "__main__":
    main()
