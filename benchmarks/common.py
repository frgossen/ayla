"""Shared timing harness for the GEMM benchmarks.

The benchmark entry points build lists of :class:`BenchGemmConfig` and hand
them to :func:`run`, so they report identical columns measured identically.

Configs describe the mathematical GEMM, ``(m, k) @ (k, n)``. The operands are
built in the layout ayla's kernels take: the right one transposed, as ``(n, k)``,
computing ``a @ bt.T``. The torch baseline is :func:`torch_mm_nt` for the same
reason -- timing an NN GEMM against ayla's NT one would not be a comparison.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


# Peak dense tensor-core TFLOPS by (exact GPU name, dtype).
# GPU names must match torch.cuda.get_device_name() exactly.
# Sources:
# - H100: https://www.nvidia.com/en-us/data-center/h100/
# - B200: https://www.nvidia.com/en-us/data-center/b200/
#   NVIDIA specifies 2.25 PFLOP/s dense BF16 Tensor Core throughput (and
#   4.5 PFLOP/s with structured sparsity); benchmark percentages use dense.
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
    ("NVIDIA B200", torch.bfloat16): 2250.0,
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


# CUDA-event timing config. Both implementations are captured and replayed on
# the same PyTorch CUDA stream, so neither Python nor dispatcher launch overhead
# is attributed to one implementation but not the other.
_WARMUP_S = 0.5        # clock-ramp warmup budget...
_WARMUP_ITERS = 100    # ...capped at this many launches (survive slow kernels)
_GRAPH_LAUNCHES = 4
_MIN_RUN_TIME_S = 0.5
_MAX_ROUNDS = 200


def torch_mm_nt(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
    """The baseline GEMM in the collection's operand layout: ``a @ bt.T``."""
    return torch.mm(a, bt.t())


def out_parameter_op(launcher: Callable) -> Callable:
    """Adapt a launcher that writes its third argument to a benchmark op."""
    output = None

    def op(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
        nonlocal output
        if output is None:
            output = torch.empty(
                (a.shape[0], bt.shape[0]), dtype=a.dtype, device=a.device,
            )
        launcher(a, bt, output)
        return output

    return op


def assert_against_torch(
    cfg: BenchGemmConfig,
    *,
    seed: int = 0,
    max_abs_error: float = 0,
    max_relative_rms_error: float = 0,
) -> tuple[float, float]:
    """Assert a GEMM config agrees with torch and return its error metrics."""
    torch.manual_seed(seed)
    a = torch.randn((cfg.m, cfg.k), dtype=cfg.dtype, device="cuda")
    bt = torch.randn((cfg.n, cfg.k), dtype=cfg.dtype, device="cuda")
    actual = cfg.op(a, bt)
    expected = torch_mm_nt(a, bt)
    torch.cuda.synchronize()
    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    if max_abs > max_abs_error or relative_rms > max_relative_rms_error:
        raise RuntimeError(
            f"{cfg.label}: max abs {max_abs}, relative RMS {relative_rms}"
        )
    return max_abs, relative_rms


def _capture_graph(
    run: Callable, stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        for _ in range(_GRAPH_LAUNCHES):
            run()
    stream.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / _GRAPH_LAUNCHES


def _prepare(
    cfg: BenchGemmConfig,
    a: torch.Tensor,
    bt: torch.Tensor,
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, float]:
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    op = cfg.op

    def invoke():
        return op(a, bt)

    result = invoke()
    assert result.dtype == cfg.dtype

    # Use one explicit stream for the warmup, graph capture, Ayla launch, torch
    # launch, and timing events. PtxLauncher selects PyTorch's current stream,
    # while torch.mm naturally enqueues onto that same current stream.
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        probe_start.record(stream)
        invoke()
        probe_end.record(stream)
    probe_end.synchronize()
    one_ms = max(probe_start.elapsed_time(probe_end), 1e-3)

    warmup_iters = max(1, min(_WARMUP_ITERS, round(_WARMUP_S * 1e3 / one_ms)))
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            invoke()
    stream.synchronize()

    graph = _capture_graph(invoke, stream)
    return graph, max(_time_graph(graph, stream), 1e-3)


def _result(cfg: BenchGemmConfig, times_ms: list[float]) -> BenchResult:
    n = len(times_ms)
    min_ms = min(times_ms)
    median_ms = statistics.median(times_ms)
    # Median throughput is more stable than selecting the luckiest round.
    tflops = cfg.flops / (median_ms / 1e3) / 1e12
    return BenchResult(
        label=f"{cfg.label} ({cfg.m}x{cfg.k} @ {cfg.k}x{cfg.n})",
        rounds=n,
        iters=_GRAPH_LAUNCHES,
        median_ms=median_ms,
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
    """Time configs in rotating order on one stream and shared shape inputs."""
    if not configs:
        return []
    stream = torch.cuda.current_stream()
    inputs: dict[
        tuple[int, int, int, torch.dtype],
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    prepared: list[tuple[BenchGemmConfig, torch.cuda.CUDAGraph, float]] = []
    for i, cfg in enumerate(configs, 1):
        key = cfg.m, cfg.n, cfg.k, cfg.dtype
        if key not in inputs:
            inputs[key] = (
                torch.randn(cfg.m, cfg.k, dtype=cfg.dtype, device="cuda"),
                torch.randn(cfg.n, cfg.k, dtype=cfg.dtype, device="cuda"),
            )
        graph, graph_ms = _prepare(cfg, *inputs[key], stream)
        prepared.append((cfg, graph, graph_ms))
        print(
            f"[prepare {i}/{len(configs)}] {cfg.label}",
            file=sys.stderr,
            flush=True,
        )

    slowest_graph_ms = max(ms for _, _, ms in prepared) * _GRAPH_LAUNCHES
    rounds = max(
        3,
        min(_MAX_ROUNDS, round(_MIN_RUN_TIME_S * 1e3 / slowest_graph_ms)),
    )
    samples = [[] for _ in prepared]
    for round_index in range(rounds):
        offset = round_index % len(prepared)
        order = range(offset, offset + len(prepared))
        for raw_index in order:
            index = raw_index % len(prepared)
            samples[index].append(_time_graph(prepared[index][1], stream))

    results = [
        _result(cfg, times_ms)
        for (cfg, _, _), times_ms in zip(prepared, samples, strict=True)
    ]
    _populate_percentages(configs, results)
    return results


def print_results(results: list[BenchResult]) -> None:
    """Print the results as a table, sized to the widest label."""
    label_width = max((len(result.label) for result in results), default=len("label"))
    for i, result in enumerate(results):
        _print(result, label_width, with_header=(i == 0))


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    """Write benchmark rows, creating the output directory if necessary."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_tflops_curve(
    axis,
    x_values: list[int],
    ayla_tflops: list[float],
    torch_tflops: list[float],
    *,
    annotation_prefix: str = "",
    annotation_fontsize: float = 7,
) -> None:
    """Plot and annotate an Ayla/cuBLAS throughput comparison curve."""
    axis.plot(x_values, ayla_tflops, marker="o", linewidth=2, label="Ayla")
    axis.plot(
        x_values, torch_tflops, marker="o", linewidth=2,
        label="cuBLAS (torch.mm)",
    )
    for index, (x, ayla_value, torch_value) in enumerate(zip(
        x_values, ayla_tflops, torch_tflops, strict=True,
    )):
        difference = ayla_value - torch_value
        advantage = 100 * difference / torch_value
        axis.annotate(
            f"{annotation_prefix}{advantage:+.1f}%\n{difference:+.1f} TFLOP/s",
            (x, max(ayla_value, torch_value)),
            xytext=(0, 7 + 12 * (index % 2)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annotation_fontsize,
            color="#147d3f" if advantage >= 0 else "#b3261e",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        x_values, [f"{value:,}" for value in x_values], rotation=45, ha="right",
    )
    log_min = math.log2(min(x_values))
    log_max = math.log2(max(x_values))
    padding = max(.4, .05 * (log_max - log_min))
    axis.set_xlim(2 ** (log_min - padding), 2 ** (log_max + padding))
    axis.grid(True, alpha=.3)
    axis.margins(y=.25)
