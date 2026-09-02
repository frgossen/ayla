"""Shared timing harness for the GEMM benchmarks.

The benchmark entry points build lists of :class:`GemmBenchConfig` and hand
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


@dataclass(frozen=True)
class BenchConfig:
    """A computation to measure: what to feed it, and what to run."""

    name: str
    make_inputs: Callable[[], list[torch.Tensor]]
    compute: Callable[..., torch.Tensor]


@dataclass(frozen=True)
class BenchResult:
    """One row's measurement. Times are milliseconds."""

    name: str
    rounds: int   # timed event-pairs (the sample count for median/stdev/min/max)
    iters: int    # launches averaged within each round
    median_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float


@dataclass
class GemmBenchConfig:
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


@dataclass(frozen=True)
class GemmBenchResult(BenchResult):
    """A GEMM's measurement: the timing, plus what it means as throughput."""

    tflops: float
    peak_pct: float | None = None  # None if peak unspecified
    torch_mm_pct: float | None = None


# The reporting below renders either, and nothing else about them.
AnyResult = BenchResult | GemmBenchResult


# CUDA-event timing config. Both implementations are captured and replayed on
# the same PyTorch CUDA stream, so neither Python nor dispatcher launch overhead
# is attributed to one implementation but not the other.
_WARMUP_S = 0.5        # clock-ramp warmup budget...
_WARMUP_ITERS = 100    # ...capped at this many launches (survive slow kernels)
_GRAPH_WARMUP = 3      # launches on the capture stream before recording
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
    cfg: GemmBenchConfig,
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
    """Record `run` into a graph that can be replayed.

    The launches before recording are not about clock ramp -- that is the
    caller's business, and :func:`_prepare` budgets for it. They are for
    anything a call does only once, allocating a workspace or initialising a
    handle: that has to happen before recording starts, or it lands inside the
    graph and is replayed forever after.

    `_prepare` already warms enough to cover that. Doing it here as well costs
    three launches and makes the function correct to call on its own, which is
    what a caller wants if it ramps the clocks by replaying the graph rather
    than by calling through Python.
    """
    with torch.cuda.stream(stream):
        for _ in range(_GRAPH_WARMUP):
            run()
    stream.synchronize()

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
    cfg: GemmBenchConfig,
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


def _result(cfg: GemmBenchConfig, times_ms: list[float]) -> GemmBenchResult:
    n = len(times_ms)
    min_ms = min(times_ms)
    median_ms = statistics.median(times_ms)
    # Median throughput is more stable than selecting the luckiest round.
    tflops = cfg.flops / (median_ms / 1e3) / 1e12
    return GemmBenchResult(
        name=f"{cfg.label} ({cfg.m}x{cfg.k} @ {cfg.k}x{cfg.n})",
        rounds=n,
        iters=_GRAPH_LAUNCHES,
        median_ms=median_ms,
        mean_ms=statistics.mean(times_ms),
        stdev_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
        min_ms=min_ms,
        max_ms=max(times_ms),
        tflops=tflops,
    )


def _with_percentages(
    configs: list[GemmBenchConfig], results: list[GemmBenchResult]
) -> list[GemmBenchResult]:
    """The same rows, with the two relative columns filled in."""
    torch_mm_tflops = next(
        (
            result.tflops
            for cfg, result in zip(configs, results)
            if cfg.op is torch_mm_nt
        ),
        None,
    )
    device_name = torch.cuda.get_device_name(0)
    filled = []
    for cfg, result in zip(configs, results, strict=True):
        peak = _PEAK_DENSE_TC_TFLOPS.get((device_name, cfg.dtype))
        filled.append(dataclasses.replace(
            result,
            peak_pct=result.tflops / peak * 100 if peak else None,
            torch_mm_pct=(None if torch_mm_tflops is None
                          else result.tflops / torch_mm_tflops * 100),
        ))
    return filled


def _format_value(value, name: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if name.endswith("_ms"):
            # Four decimals, not three: these benchmarks run at fractions of
            # a millisecond, and three rounds every row's stdev to the same
            # number, which is exactly the column you need to read a spread.
            return f"{value:.4f}ms"
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


def _format_field(r: AnyResult, f, label_width: int) -> str:
    value = _format_value(getattr(r, f.name), f.name)
    return f"{value:{_align(f)}{_width(f, label_width)}s}"


def run(configs: list[GemmBenchConfig]) -> list[GemmBenchResult]:
    """Time configs in rotating order on one stream and shared shape inputs."""
    if not configs:
        return []
    stream = torch.cuda.current_stream()
    inputs: dict[
        tuple[int, int, int, torch.dtype],
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    prepared: list[tuple[GemmBenchConfig, torch.cuda.CUDAGraph, float]] = []
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
    return _with_percentages(configs, results)


def print_results(
    results: list[AnyResult],
    label_width: int | None = None,
    with_header: bool = True,
) -> None:
    """Print rows as a table, whatever dataclass they are.

    Nothing is assumed about the row type beyond its being a dataclass. Its
    fields are the columns, in declaration order; a `str` field is
    left-aligned and sized to the widest value in it, everything else is
    right-aligned in a fixed column; and the suffix on a field name picks the
    units, `_ms` or `_pct`. So :class:`GemmBenchResult` and
    :class:`BenchResult` both render without this knowing either of them.

    `label_width` is measured from the rows unless given. Pass it, with
    `with_header=False`, to append to a table already printed -- a caller
    reporting a sweep row by row as it goes, rather than all at the end.
    """
    if not results:
        return
    fields = dataclasses.fields(results[0])
    if label_width is None:
        label_width = max(
            [len(f.name) for f in fields if _is_str_field(f)]
            + [len(str(getattr(r, f.name)))
               for r in results for f in fields if _is_str_field(f)],
            default=len("label"),
        )
    if with_header:
        print("  ".join(
            f"{f.name:{_align(f)}{_width(f, label_width)}s}" for f in fields
        ))
    for result in results:
        print("  ".join(
            _format_field(result, f, label_width) for f in fields
        ))


# ---------------------------------------------------------------------------
# A general harness, for anything that is not a GEMM
# ---------------------------------------------------------------------------

def _capture(config: BenchConfig, stream) -> torch.cuda.CUDAGraph:
    """Capture one config's compute into a graph that can be replayed."""
    inputs = config.make_inputs()
    return _capture_graph(lambda: config.compute(*inputs), stream)


def run_bench(configs: list[BenchConfig]) -> list[BenchResult]:
    """Time every row once per round, on one stream.

    Everything after capture is a graph replay, warmup included. Interleaving
    is the point: warmed and timed row by row instead, the first would run on
    a cold card and the last would find the clocks already up, and whatever
    drifts over the run would land on one row rather than all of them.

    The rounds are not rotated, which is the one place this departs from
    :func:`run` above. That shifts the starting config each round to keep any
    one of them out of a fixed position; measured on three same-sized configs
    it moved the result by less than the round-to-round spread, so it is left
    out here. Worth revisiting for configs of very different sizes, where a
    big one's cache footprint would reliably land on its neighbour.
    """
    stream = torch.cuda.current_stream()
    graphs = [_capture(config, stream) for config in configs]

    def replay_all() -> list[float]:
        """Replay and time every row once, in row order."""
        return [_time_graph(graph, stream) for graph in graphs]

    # Warm to a time budget rather than a fixed count, the way :func:`_prepare`
    # does, so a slow config is not warmed a hundred times over. One round
    # replays every config, so the budget is spent against their sum -- unlike
    # `rounds` below, which sizes off the slowest, because there the goal is
    # per-config measured time rather than elapsed ramp. The cap is
    # `_WARMUP_ITERS` launches per config, which is what binds for anything
    # fast.
    per_round_ms = max(sum(replay_all()), 1e-3) * _GRAPH_LAUNCHES
    for _ in range(max(1, min(_WARMUP_ITERS // _GRAPH_LAUNCHES,
                              round(_WARMUP_S * 1e3 / per_round_ms)))):
        replay_all()

    # Size the run off the slowest config, priced warm.
    slowest = max(max(replay_all()), 1e-3) * _GRAPH_LAUNCHES
    rounds = max(3, min(_MAX_ROUNDS,
                        round(_MIN_RUN_TIME_S * 1e3 / slowest)))

    by_round = [replay_all() for _ in range(rounds)]
    # A round per entry, transposed to a row per entry. Both zips are strict:
    # every round has to have timed every row, and every row has to have a
    # config. Neither can fail today, and a lenient zip would answer a broken
    # one by quietly dropping rows off the end of the table.
    by_row = zip(*by_round, strict=True)

    return [
        BenchResult(
            name=config.name,
            rounds=rounds,
            iters=_GRAPH_LAUNCHES,
            median_ms=statistics.median(times),
            mean_ms=statistics.mean(times),
            stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_ms=min(times),
            max_ms=max(times),
        )
        for config, times in zip(configs, by_row, strict=True)
    ]


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
