#!/usr/bin/env python3
"""Store-pipeline measurement probes for the reduced-grid GEMM variant.

Probe 1 (store-cost upper bound): time the reduced-grid variant with the real
epilogue store vs a build that elides it. The runtime delta is the *exposed*
epilogue-store cost — the maximum a store pipeline could recover. If it is
~0, store pipelining cannot help and we stop early.

Phase B (``--phase-b``): literal ON/OFF timing at ``k_stages=2``. OFF uses the
original scalar register→global epilogue; ON uses compact smem staging plus TMA
stores whose waits are delayed across the following output tile's compute.

Timing reuses the CUDA-event, min-of-rounds method from
``benchmarks/bench_gemm_collection._bench`` (fastest round = steady boost
clock, least noise).

Usage:
    python tools/measure_store_pipeline.py                # R in {1,2,4,8}
    python tools/measure_store_pipeline.py --factors 1 2  # specific R
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import time

import torch

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_storepipe import (
    auto_matmul_4096_wgmma256_wg_storepipe_module,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

N = 4096
FLOPS = 2 * N * N * N
PEAK_BF16_TFLOPS = 989.5  # NVIDIA H100 dense BF16 tensor-core peak

_WARMUP_ITERS = 100
_WARMUP_S = 0.5
_TARGET_ROUND_MS = 50
_MIN_RUN_TIME_S = 0.5
_COOLDOWN_S = 1.0


def _bench_launcher(run) -> tuple[float, float, int, int]:
    """Return (min_ms, stdev_ms, iters, rounds) for a no-arg ``run`` closure."""
    time.sleep(_COOLDOWN_S)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    run()
    end.record()
    torch.cuda.synchronize()
    one_ms = max(start.elapsed_time(end), 1e-3)

    iters = max(1, round(_TARGET_ROUND_MS / one_ms))
    round_ms = iters * one_ms
    rounds = max(1, round(_MIN_RUN_TIME_S * 1e3 / round_ms))
    warmup_iters = max(1, min(_WARMUP_ITERS, round(_WARMUP_S * 1e3 / one_ms)))
    for _ in range(warmup_iters):
        run()
    torch.cuda.synchronize()

    pairs = [
        (torch.cuda.Event(enable_timing=True),
         torch.cuda.Event(enable_timing=True))
        for _ in range(rounds)
    ]
    torch.cuda.synchronize()
    for s, e in pairs:
        s.record()
        for _ in range(iters):
            run()
        e.record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) / iters for s, e in pairs]
    stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return min(times_ms), stdev, iters, len(times_ms)


def _build_launcher(
    reduction_factor: int, skip_store: bool, *, k_stages: int = 4,
    store_pipeline: bool = False,
):
    module = auto_matmul_4096_wgmma256_wg_storepipe_module(
        reduction_factor=reduction_factor, skip_store=skip_store,
        k_stages=k_stages, store_pipeline=store_pipeline,
    )
    return make_ptx_launcher(make_ptx_kernel(module))


def _reduce_epilogue_stores(ptx: str, keep: int = 1) -> str:
    """Keep only the first ``keep`` epilogue ``st.global.b32`` stores.

    All K-accumulation WGMMAs write the same accumulator register tile, so a
    single surviving store keeps the whole compute chain live through ptxas
    (SASS) DCE while cutting the epilogue store traffic ~64x. This isolates
    the store-*traffic* cost, unlike fully eliding the store (which lets ptxas
    delete the compute that only fed it).
    """
    out: list[str] = []
    seen = 0
    for line in ptx.splitlines(keepends=True):
        if "st.global.b32" in line:
            seen += 1
            if seen > keep:
                continue
        out.append(line)
    return "".join(out)


def _build_reduced_store_launcher(reduction_factor: int, keep: int = 1):
    """Real-compute build whose epilogue store traffic is cut to ``keep`` regs."""
    module = auto_matmul_4096_wgmma256_wg_storepipe_module(
        reduction_factor=reduction_factor, skip_store=False,
    )
    kernel = make_ptx_kernel(module)
    kernel = dataclasses.replace(
        kernel, ptx=_reduce_epilogue_stores(kernel.ptx, keep=keep),
    )
    return make_ptx_launcher(kernel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factors", type=int, nargs="+", default=[1, 2, 4, 8],
        help="grid reduction factors R to sweep",
    )
    parser.add_argument(
        "--phase-b", action="store_true",
        help="measure the literal store pipeline ON/OFF (requires R > 1)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(0)
    a = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(N, N, dtype=torch.bfloat16, device="cuda")
    expected = (a @ b.T).float()

    def peak_pct(ms: float) -> float:
        return (FLOPS / (ms / 1e3) / 1e12) / PEAK_BF16_TFLOPS * 100

    if args.phase_b:
        print(f"{'R':>3}  {'off_ms':>9}  {'on_ms':>9}  {'speedup':>8}  "
              f"{'off_%pk':>8}  {'on_%pk':>8}  {'max_err':>8}")
        for R in args.factors:
            if R <= 1:
                continue
            off = _build_launcher(R, False, k_stages=2)
            on = _build_launcher(R, False, k_stages=2, store_pipeline=True)
            c.zero_(); on(a, b, c); torch.cuda.synchronize()
            max_err = (c.float() - expected).abs().max().item()
            off_ms, _, _, _ = _bench_launcher(lambda: off(a, b, c))
            on_ms, _, _, _ = _bench_launcher(lambda: on(a, b, c))
            print(f"{R:>3}  {off_ms:>9.3f}  {on_ms:>9.3f}  "
                  f"{off_ms / on_ms:>7.3f}x  {peak_pct(off_ms):>7.1f}%  "
                  f"{peak_pct(on_ms):>7.1f}%  {max_err:>8.3f}")
        return

    header = (
        f"{'R':>3}  {'real_ms':>9}  {'redstore_ms':>11}  {'Δ_store_ms':>11}  "
        f"{'Δ_store_%':>9}  {'real_%pk':>8}  {'elide_ms':>9}  {'max_err':>8}"
    )
    print(header)
    print("-" * len(header))

    for R in args.factors:
        real = _build_launcher(R, skip_store=False)
        redstore = _build_reduced_store_launcher(R, keep=1)
        elide = _build_launcher(R, skip_store=True)

        # Correctness of the real-store build.
        c.zero_()
        real(a, b, c)
        torch.cuda.synchronize()
        max_err = (c.float() - expected).abs().max().item()

        real_ms, _, _, _ = _bench_launcher(lambda: real(a, b, c))
        red_ms, _, _, _ = _bench_launcher(lambda: redstore(a, b, c))
        elide_ms, _, _, _ = _bench_launcher(lambda: elide(a, b, c))

        delta = real_ms - red_ms
        delta_pct = delta / real_ms * 100 if real_ms else 0.0
        print(
            f"{R:>3}  {real_ms:>9.3f}  {red_ms:>11.3f}  {delta:>11.4f}  "
            f"{delta_pct:>8.2f}%  {peak_pct(real_ms):>7.1f}%  "
            f"{elide_ms:>9.3f}  {max_err:>8.3f}"
        )

    print(
        "\nΔ_store = real - redstore. NOTE: redstore is produced by hand-editing"
        " PTX (keeping 1 of 64 epilogue stores so ptxas keeps the WGMMA chain),"
        " which perturbs register allocation/scheduling. A store-count sweep"
        " (keep=1..64) is non-monotonic, so |Δ_store| here is within ptxas"
        " codegen jitter, i.e. the exposed store-traffic cost is ~0 / not"
        " measurable. The reliable signal is real_%pk across R: flat until"
        " occupancy collapses. elide removes the store entirely and lets ptxas"
        " DCE the compute, so it is only a loose floor, not a target."
    )


if __name__ == "__main__":
    main()
