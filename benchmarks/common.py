"""Shared timing harness for the GEMM benchmarks.

Both ``bench_gemm_collection.py`` (one hand-picked shape, every kernel) and
``bench_oss_pytorch_gemm_collection.py`` (the shapes real models run) build a
list of :class:`BenchGemmConfig` and hand it to :func:`run`, so they report
identical columns measured identically.

Configs describe the mathematical GEMM, ``(m, k) @ (k, n)``. The operands are
built in the layout ayla's kernels take: the right one transposed, as ``(n, k)``,
computing ``a @ bt.T``. The torch baseline is :func:`torch_mm_nt` for the same
reason -- timing an NN GEMM against ayla's NT one would not be a comparison.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch


# Peak dense tensor-core TFLOPS by (exact GPU name, dtype).
# GPU names must match torch.cuda.get_device_name() exactly.
# Source: https://www.nvidia.com/en-us/data-center/h100/
_PEAK_DENSE_TC_TFLOPS = {
    ("NVIDIA H100", torch.float64): 67.0,  # FP64 Tensor Core (dense)
    (
        "NVIDIA H100",
        torch.float32,
    ): 494.5,  # TF32 Tensor Core, spec says 989 w/ sparsity
    ("NVIDIA H100", torch.bfloat16): 989.5,  # spec says 1979 w/ sparsity
    ("NVIDIA H100", torch.float16): 989.5,  # spec says 1979 w/ sparsity
    ("NVIDIA H100", torch.float8_e4m3fn): 1979.0,  # spec says 3958 w/ sparsity
    ("NVIDIA H100", torch.float8_e5m2): 1979.0,  # spec says 3958 w/ sparsity
}


@dataclass
class BenchGemmConfig:
    label: str  # e.g. "torch.mm" or "ayla.naive_matmul_4096"
    m: int
    n: int
    k: int
    op: Callable  # the GEMM to time, called as op(a, bt) with bt of shape (n, k)
    dtype: torch.dtype
    allow_tf32: bool = True

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


@dataclass
class BenchResult:
    label: str
    rounds: int   # timed event-pairs (the sample count for median/stdev/min/max)
    iters: int    # launches averaged within each round
    median_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    tflops: float
    peak_pct: float | None = None  # None if peak unspecified
    torch_mm_pct: float | None = None


# CUDA-event timing config (GPU-only, cudaEvent-based like matmul.cu).
_WARMUP_S = 0.5        # clock-ramp warmup budget...
_WARMUP_ITERS = 100    # ...capped at this many launches (survive slow kernels)
_TARGET_ROUND_MS = 50  # aim for ~this much GPU work per timed round
_MIN_RUN_TIME_S = 0.5   # total timing budget across all rounds (>1 round so
                        # min/median/max/stdev are meaningful, not one sample)
# Idle pause before each config so every kernel starts from the same thermal
# state — removes cross-config ordering/throttling bias (matmul.cu sleeps 5s).
_COOLDOWN_S = 1.0


def torch_mm_nt(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
    """The baseline GEMM in the collection's operand layout: ``a @ bt.T``."""
    return torch.mm(a, bt.t())


def _bench(cfg: BenchGemmConfig) -> BenchResult:
    # Cool down first so every config starts from the same thermal state —
    # removes cross-config ordering/throttling bias (matmul.cu sleeps 5s).
    time.sleep(_COOLDOWN_S)
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    # ayla's kernels take the right operand transposed, as (n, k), and compute
    # a @ bt.T -- so does the torch baseline, or the two would not be timing the
    # same GEMM. This is also the layout real models produce: nn.Linear lowers
    # to addmm(bias, x, W.t()).
    a = torch.randn(cfg.m, cfg.k, dtype=cfg.dtype, device="cuda")
    bt = torch.randn(cfg.n, cfg.k, dtype=cfg.dtype, device="cuda")
    op = cfg.op

    def run():
        return op(a, bt)

    result = run()
    assert result.dtype == cfg.dtype

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Rough single-launch estimate (clocks not yet ramped, but enough to size
    # the loop counts — and to keep them bounded for a very slow kernel).
    torch.cuda.synchronize()
    start.record()
    run()
    end.record()
    torch.cuda.synchronize()
    one_ms = max(start.elapsed_time(end), 1e-3)

    # Inner launches per timed round (~_TARGET_ROUND_MS of GPU work) and rounds
    # to cover ~_MIN_RUN_TIME_S of timing. Both clamp to >=1 (and `rounds` keys
    # off the actual round duration), so a crazy-slow kernel does the minimum
    # rather than 500 warmups / hundreds of rounds.
    iters = max(1, round(_TARGET_ROUND_MS / one_ms))
    round_ms = iters * one_ms
    rounds = max(1, round(_MIN_RUN_TIME_S * 1e3 / round_ms))

    # Warmup to ramp the GPU clocks to a steady boost state (keeps round-to-
    # round variance low). Bounded by a time budget and an iteration cap, so a
    # very slow kernel warms up for a few launches rather than _WARMUP_ITERS.
    warmup_iters = max(1, min(_WARMUP_ITERS, round(_WARMUP_S * 1e3 / one_ms)))
    for _ in range(warmup_iters):
        run()
    torch.cuda.synchronize()

    # Measure GPU time per launch via CUDA events. All rounds are queued
    # back-to-back with no intervening synchronize, so the GPU stays
    # continuously saturated — no idle gaps where clocks dip between rounds.
    # That keeps variance low and reflects steady-state kernel throughput
    # (CPU launch / output-allocation overhead overlaps with GPU work), like
    # matmul.cu's cudaEvent timing rather than a wall-clock Timer.
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

    n = len(times_ms)
    min_ms = min(times_ms)
    # Throughput / peak% from the fastest round (best-case steady clock, least
    # contaminated by clock dips or scheduling noise), not the median.
    tflops = cfg.flops / (min_ms / 1e3) / 1e12
    return BenchResult(
        label=f"{cfg.label} ({cfg.m}x{cfg.k} @ {cfg.k}x{cfg.n})",
        rounds=n,
        iters=iters,
        median_ms=statistics.median(times_ms),
        mean_ms=statistics.mean(times_ms),
        stdev_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
        min_ms=min_ms,
        max_ms=max(times_ms),
        tflops=tflops,
    )


def _populate_percentages(
    configs: list[BenchGemmConfig], results: list[BenchResult]
) -> None:
    torch_mm_tflops = next(
        (
            result.tflops
            for cfg, result in zip(configs, results)
            if cfg.op is torch_mm_nt
        ),
        None,
    )
    device_name = torch.cuda.get_device_name(0)
    for cfg, result in zip(configs, results):
        peak = _PEAK_DENSE_TC_TFLOPS.get((device_name, cfg.dtype))
        result.peak_pct = result.tflops / peak * 100 if peak else None
        if torch_mm_tflops is not None:
            result.torch_mm_pct = result.tflops / torch_mm_tflops * 100


def _format_value(value, name: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if name.endswith("_ms"):
            return f"{value:.3f}ms"
        if name.endswith("_pct"):
            return f"{value:.1f}%"
        return f"{value:.3f}"
    return str(value)


def _is_str_field(f) -> bool:
    return f.type is str or f.type == "str"


def _width(f, label_width: int) -> int:
    return label_width if _is_str_field(f) else 12


def _align(f) -> str:
    return "<" if _is_str_field(f) else ">"


def _format_field(r: BenchResult, f, label_width: int) -> str:
    value = _format_value(getattr(r, f.name), f.name)
    return f"{value:{_align(f)}{_width(f, label_width)}s}"


def _print(r: BenchResult, label_width: int, with_header: bool = False) -> None:
    fields = dataclasses.fields(r)
    if with_header:
        print(
            "  ".join(
                f"{f.name:{_align(f)}{_width(f, label_width)}s}" for f in fields
            )
        )
    print(
        "  ".join(_format_field(r, f, label_width) for f in fields)
    )


def run(configs: list[BenchGemmConfig]) -> list[BenchResult]:
    """Time every config, reporting progress, and fill the comparison columns."""
    results = []
    for i, cfg in enumerate(configs, 1):
        results.append(_bench(cfg))
        print(f"[{i}/{len(configs)}] {cfg.label}", file=sys.stderr, flush=True)

    _populate_percentages(configs, results)
    return results


def print_results(results: list[BenchResult]) -> None:
    """Print the results as a table, sized to the widest label."""
    label_width = max((len(result.label) for result in results), default=len("label"))
    for i, result in enumerate(results):
        _print(result, label_width, with_header=(i == 0))
