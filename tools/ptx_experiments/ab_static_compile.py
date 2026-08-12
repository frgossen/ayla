"""Compare otherwise-identical PTX variants with correctness and paired timing."""
from __future__ import annotations

import dataclasses
import statistics
import sys
from pathlib import Path

import torch

import ayla.ptx_runtime as runtime
import ayla.torch_custom_call as custom_call


# Import the module builder without eagerly constructing all registered launchers.
real_make_launcher = runtime.make_ptx_launcher
runtime.make_ptx_launcher = lambda kernel: None
custom_call.make_torch_custom_call = lambda *args, **kwargs: None
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
runtime.make_ptx_launcher = real_make_launcher

from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402


def launcher(ptx_path: str):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_tiles_per_cta=4,
            operand_stages=3,
            output_slices=2,
            output_tma_inner_padding=16,
            output_store_buffers=2,
            operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
        )
    )
    return real_make_launcher(
        dataclasses.replace(kernel, ptx=Path(ptx_path).read_text())
    )


def burst(run, iters=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compare(label_a, run_a, label_b, run_b):
    for _ in range(1500):
        run_a()
        run_b()
    torch.cuda.synchronize()
    times = {label_a: [], label_b: []}
    ratios = []
    for i in range(240):
        if i % 2 == 0:
            a_time = burst(run_a)
            b_time = burst(run_b)
        else:
            b_time = burst(run_b)
            a_time = burst(run_a)
        times[label_a].append(a_time)
        times[label_b].append(b_time)
        ratios.append(b_time / a_time)
    for label in (label_a, label_b):
        values = times[label]
        print(
            f"{label:20s} median={statistics.median(values) * 1000:8.3f} us "
            f"mean={statistics.mean(values) * 1000:8.3f} us "
            f"stdev={statistics.stdev(values) * 1000:7.3f} us n={len(values)}"
        )
    ratios.sort()
    median_ratio = statistics.median(ratios)
    lo = ratios[int(0.05 * len(ratios))]
    hi = ratios[int(0.95 * len(ratios)) - 1]
    print(
        f"{label_b} speedup over {label_a}: {(1 - median_ratio) * 100:+.3f}% "
        f"(central 90% paired range {(1 - hi) * 100:+.3f}%.."
        f"{(1 - lo) * 100:+.3f}%)"
    )


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: ab_static_compile.py BASE.ptx CURRENT.ptx ELECT.ptx")
    names = ("recompute", "reuse-offset", "elect-loads")
    launchers = dict(zip(names, map(launcher, sys.argv[1:])))

    torch.manual_seed(0)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    outputs = {
        name: torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda")
        for name in names
    }
    for name in names:
        launchers[name](a, b, outputs[name])
    reference = (a.float() @ b.float().t()).bfloat16()
    torch.cuda.synchronize()
    for name in names:
        max_error = (outputs[name].float() - reference.float()).abs().max().item()
        mismatches = (outputs[name] != outputs["recompute"]).sum().item()
        print(f"{name:20s} max_error={max_error:g} baseline_mismatches={mismatches}")

    runs = {
        name: (lambda n=name: launchers[n](a, b, outputs[n]))
        for name in names
    }
    compare("recompute", runs["recompute"], "reuse-offset", runs["reuse-offset"])
    compare("reuse-offset", runs["reuse-offset"], "elect-loads", runs["elect-loads"])


if __name__ == "__main__":
    main()
