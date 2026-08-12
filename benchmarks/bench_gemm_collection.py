"""Benchmark GEMMs implemented as torch ops."""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import ayla.gemm_collection  # noqa: F401,E402 — registers torch.ops.ayla.*
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
    op: Callable  # the GEMM to time, called as op(a, b)
    dtype: torch.dtype
    allow_tf32: bool = True
    is_reference: bool = False

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


@dataclass
class BenchResult:
    label: str
    rounds: int   # timed event-pairs (the sample count for median/mean/stdev)
    iters: int    # launches averaged within each round
    median_ms: float
    mean_ms: float
    stdev_ms: float
    tflops: float
    peak_pct: float | None = None  # None if peak unspecified
    cublas_median_ms: float | None = None
    cublas_pct: float | None = None


# CUDA-event timing config (GPU-only, cudaEvent-based like matmul.cu).
_WARMUP_S = 0.5        # clock-ramp warmup budget...
_WARMUP_ITERS = 100    # ...capped at this many launches (survive slow kernels)
_TARGET_ROUND_MS = 50  # aim for ~this much GPU work per timed round
_MIN_RUN_TIME_S = 0.5   # total timing budget across all rounds
# Idle pause before each config so every kernel starts from the same thermal
# state — removes cross-config ordering/throttling bias (matmul.cu sleeps 5s).
_COOLDOWN_S = 1.0


def _inputs(cfg: BenchGemmConfig) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn(cfg.m, cfg.k, dtype=cfg.dtype, device="cuda")
    # Ayla GEMMs consume B in row-major [N, K] form and compute A @ B.T.
    # Keep the reference operand in that same physical layout so both kernels
    # execute the identical mathematical operation and cuBLAS transpose path.
    b = torch.randn(cfg.n, cfg.k, dtype=cfg.dtype, device="cuda")
    return a, b


def _one_launch_ms(run: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    run()
    end.record()
    torch.cuda.synchronize()
    return max(start.elapsed_time(end), 1e-3)


def _result(
    cfg: BenchGemmConfig, times_ms: list[float], iters: int,
) -> BenchResult:
    n = len(times_ms)
    median_ms = statistics.median(times_ms)
    # Median is the comparison statistic: it represents a typical sustained
    # round while rejecting isolated clock/scheduling excursions.  A minimum
    # selects the luckiest transient; a mean is more easily skewed by them.
    tflops = cfg.flops / (median_ms / 1e3) / 1e12
    return BenchResult(
        label=f"{cfg.label} ({cfg.m}x{cfg.k} @ {cfg.n}x{cfg.k}.T)",
        rounds=n,
        iters=iters,
        median_ms=median_ms,
        mean_ms=statistics.mean(times_ms),
        stdev_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
        tflops=tflops,
    )


def _bench(cfg: BenchGemmConfig) -> BenchResult:
    """Standalone timing for a config with no comparison reference."""
    time.sleep(_COOLDOWN_S)
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    a, b = _inputs(cfg)

    def run():
        return cfg.op(a, b)

    result = run()
    assert result.dtype == cfg.dtype
    one_ms = _one_launch_ms(run)

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

    return _result(cfg, times_ms, iters)


def _bench_pair(
    reference: BenchGemmConfig, candidate: BenchGemmConfig,
) -> tuple[BenchResult, BenchResult]:
    """Time a candidate and cuBLAS in balanced alternating blocks.

    Both use the same input tensors.  Even rounds run reference then candidate;
    odd rounds reverse that order.  This balances the power/clock advantage of
    running first, which a sequential whole-kernel benchmark cannot remove by
    taking a median afterward.
    """
    assert (candidate.m, candidate.n, candidate.k, candidate.dtype) == (
        reference.m, reference.n, reference.k, reference.dtype
    )
    time.sleep(_COOLDOWN_S)
    torch.backends.cuda.matmul.allow_tf32 = candidate.allow_tf32
    a, b = _inputs(candidate)

    runs = {
        "reference": lambda: reference.op(a, b),
        "candidate": lambda: candidate.op(a, b),
    }
    for name, run in runs.items():
        result = run()
        assert result.dtype == candidate.dtype, name

    one_ms = max(_one_launch_ms(run) for run in runs.values())
    iters = max(1, round(_TARGET_ROUND_MS / one_ms))
    round_ms = iters * one_ms
    # At least two rounds are required so AB and BA both occur.
    rounds = max(2, round(_MIN_RUN_TIME_S * 1e3 / round_ms))
    if rounds % 2:
        rounds += 1
    warmup_iters = max(1, min(_WARMUP_ITERS, round(_WARMUP_S * 1e3 / one_ms)))
    for i in range(warmup_iters):
        order = ("reference", "candidate") if i % 2 == 0 else (
            "candidate", "reference"
        )
        for name in order:
            runs[name]()
    torch.cuda.synchronize()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in runs
    }
    for i in range(rounds):
        order = ("reference", "candidate") if i % 2 == 0 else (
            "candidate", "reference"
        )
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                runs[name]()
            end.record()
            events[name].append((start, end))
    torch.cuda.synchronize()

    samples = {
        name: [start.elapsed_time(end) / iters for start, end in pairs]
        for name, pairs in events.items()
    }
    reference_result = _result(reference, samples["reference"], iters)
    candidate_result = _result(candidate, samples["candidate"], iters)
    candidate_result.cublas_median_ms = reference_result.median_ms
    candidate_result.cublas_pct = (
        candidate_result.tflops / reference_result.tflops * 100
    )
    return reference_result, candidate_result


def _populate_peak_percentages(
    configs: list[BenchGemmConfig], results: list[BenchResult]
) -> None:
    device_name = torch.cuda.get_device_name(0)
    for cfg, result in zip(configs, results):
        peak = _PEAK_DENSE_TC_TFLOPS.get((device_name, cfg.dtype))
        result.peak_pct = result.tflops / peak * 100 if peak else None


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


def main() -> None:
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Benchmark GEMM kernels")
    parser.add_argument("-k", "--filter", type=str, default=None,
                        help="Regex to filter benchmarks by label")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Run the benchmarks in reverse order")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    def ayla_bf16(label: str, op: str) -> BenchGemmConfig:
        return BenchGemmConfig(
            label, 4096, 4096, 4096, getattr(torch.ops.ayla, op), torch.bfloat16,
        )

    all_configs = [
        BenchGemmConfig(
            label="cuBLAS torch.mm(A, B.T).bf16",
            m=4096,
            n=4096,
            k=4096,
            op=lambda a, b: torch.mm(a, b.t()),
            dtype=torch.bfloat16,
            is_reference=True,
        ),
        BenchGemmConfig(
            label="ayla.pranjalssh",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.pranjalssh_4096,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.pranjalssh_pipelined (prev best)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.pranjalssh_4096_pipelined,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma128)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.auto_matmul_4096,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma256 wg)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.auto_matmul_4096_wgmma256_wg,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma256 wg, TMA store K=3)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma256 wg, TMA store K=3, 2 tiles/CTA, pre-STSM stores)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_pre_stsm,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma256 wg, TMA store K=3, 2 tiles/CTA)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles,
            dtype=torch.bfloat16,
        ),
        ayla_bf16(
            "ayla.auto_matmul (wgmma256 wg, TMA store K=3, split output, 2 tiles/CTA)",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2",
        ),
        ayla_bf16(
            "ayla.auto_matmul (wgmma256 wg, TMA store K=3, split output, 4 tiles/CTA, A+C evict-first)",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2",
        ),
        ayla_bf16(
            "ayla.auto_matmul (wgmma256 wg, 3D input TMA K-tile=128, split output, 2 tiles/CTA)",
            "auto_matmul_4096_wgmma256_wg_tma_store_k2_ktile128_2tiles_split2",
        ),
        ayla_bf16(
            "ayla.auto_matmul (wgmma256 wg, TMA store K=4, split output, 2 tiles/CTA)",
            "auto_matmul_4096_wgmma256_wg_tma_store_k4_2tiles_split2",
        ),
        ayla_bf16(
            "ayla.auto_matmul (128x384 CTA, three N=128 warpgroups, TMA store K=2)",
            "auto_matmul_4096_wgmma256_wg_tma_store_k2_n384_split3",
        ),
        ayla_bf16(
            "ayla.auto_matmul (192x256 CTA, three M=64 warpgroups, TMA store K=3)",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_m192_split2",
        ),
        BenchGemmConfig(
            label="ayla.fastcu_5_1 (fast.cu static smem)",
            m=4096,
            n=4096,
            k=4096,
            op=torch.ops.ayla.fastcu_5_1,
            dtype=torch.bfloat16,
        ),
    ]
    configs = all_configs
    if args.filter:
        pattern = re.compile(args.filter)
        configs = [c for c in configs if pattern.search(c.label)]
    if args.reverse:
        configs = configs[::-1]

    reference = next((c for c in all_configs if c.is_reference), None)
    candidates = [c for c in configs if not c.is_reference]
    results = []
    result_configs = []
    if reference is not None and candidates:
        for i, cfg in enumerate(candidates, 1):
            _, result = _bench_pair(reference, cfg)
            results.append(result)
            result_configs.append(cfg)
            print(
                f"[{i}/{len(candidates)}] {cfg.label} vs matched cuBLAS",
                file=sys.stderr, flush=True,
            )
    else:
        for i, cfg in enumerate(configs, 1):
            results.append(_bench(cfg))
            result_configs.append(cfg)
            print(f"[{i}/{len(configs)}] {cfg.label}", file=sys.stderr, flush=True)

    _populate_peak_percentages(result_configs, results)

    label_width = max(
        (len(result.label) for result in results), default=len("label")
    )
    for i, result in enumerate(results):
        _print(result, label_width, with_header=(i == 0))


if __name__ == "__main__":
    main()
