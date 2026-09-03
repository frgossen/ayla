"""Shared timing harness for the benchmarks.

Configs describe the mathematical GEMM, ``(m, k) @ (k, n)``, but the operands
are built in the layout ayla's kernels take -- the right one transposed, as
``(n, k)``, computing ``a @ bt.T``. Timing an NN GEMM against ayla's NT one
would not be a comparison.
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


@dataclass(frozen=True, kw_only=True)
class BenchConfig:
    """A computation to measure: what to feed it, and what to run.

    `verify` sees the output and inputs once before timing, and should raise
    if the computation is not doing what it claims. `make_inputs` is optional
    only so a subclass can derive it -- required when constructing directly.
    """

    name: str
    compute: Callable[..., torch.Tensor]
    make_inputs: Callable[[], list[torch.Tensor]] | None = None
    verify: Callable[..., None] | None = None


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


@dataclass(frozen=True, kw_only=True)
class GemmBenchConfig(BenchConfig):
    """A GEMM, as a benchmark: ``compute(a, bt)`` computing ``a @ bt.T``."""

    m: int
    n: int
    k: int
    dtype: torch.dtype

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


def gemm_bench_config(
    *, name: str, m: int, n: int, k: int, compute: Callable,
    dtype: torch.dtype,
) -> GemmBenchConfig:
    """A :class:`GemmBenchConfig` with its operands and check derived."""
    def make_inputs() -> list[torch.Tensor]:
        return [
            torch.randn(m, k, dtype=dtype, device="cuda"),
            torch.randn(n, k, dtype=dtype, device="cuda"),
        ]

    def verify(output: torch.Tensor, _inputs) -> None:
        assert output.dtype == dtype, (
            f"{name}: produced {output.dtype}, config says {dtype}"
        )

    return GemmBenchConfig(
        name=name, compute=compute, make_inputs=make_inputs, verify=verify,
        m=m, n=n, k=k, dtype=dtype,
    )


@dataclass(frozen=True)
class GemmBenchResult(BenchResult):
    """A GEMM's measurement: the timing, plus what it means as throughput."""

    tflops: float
    peak_pct: float | None = None  # None if peak unspecified
    torch_mm_pct: float | None = None


# Everything is captured and replayed on one stream, so launch overhead is not
# attributed to one implementation but not another.
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


def error_metrics(
    actual: torch.Tensor, expected: torch.Tensor,
) -> tuple[float, float]:
    """Return maximum absolute error and relative root-mean-square error."""
    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    return max_abs, relative_rms


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
    actual = cfg.compute(a, bt)
    expected = torch_mm_nt(a, bt)
    torch.cuda.synchronize()
    max_abs, relative_rms = error_metrics(actual, expected)
    if max_abs > max_abs_error or relative_rms > max_relative_rms_error:
        raise RuntimeError(
            f"{cfg.name}: max abs {max_abs}, relative RMS {relative_rms}"
        )
    return max_abs, relative_rms


def _time_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / _GRAPH_LAUNCHES


def _with_percentages(
    configs: list[GemmBenchConfig], results: list[GemmBenchResult]
) -> list[GemmBenchResult]:
    """The same rows, with the two relative columns filled in."""
    torch_mm_tflops = next(
        (
            result.tflops
            for cfg, result in zip(configs, results)
            if cfg.compute is torch_mm_nt
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
            # Four decimals: three collapses every row's stdev to one number.
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


def _format_field(r: BenchResult, f, label_width: int) -> str:
    value = _format_value(getattr(r, f.name), f.name)
    return f"{value:{_align(f)}{_width(f, label_width)}s}"


def run(configs: list[GemmBenchConfig]) -> list[GemmBenchResult]:
    """Time GEMM configs, and say what the timing means as throughput.

    :func:`run_bench` does the timing; this adds only the GEMM arithmetic.
    """
    if not configs:
        return []
    timed = run_bench(configs)
    carried = [field.name for field in dataclasses.fields(BenchResult)]
    results = [
        GemmBenchResult(
            **{name: getattr(row, name) for name in carried}
            | {"name": f"{cfg.name} ({cfg.m}x{cfg.k} @ {cfg.k}x{cfg.n})"},
            tflops=cfg.flops / (row.median_ms / 1e3) / 1e12,
        )
        for cfg, row in zip(configs, timed, strict=True)
    ]
    return _with_percentages(configs, results)


def print_results(
    results: list[BenchResult],
    label_width: int | None = None,
    with_header: bool = True,
) -> None:
    """Print rows as a table, whatever dataclass they are.

    Fields are the columns in declaration order; `str` fields are left-aligned
    and sized to their widest value, the rest right-aligned, and a `_ms` or
    `_pct` suffix picks the units. Pass `label_width` with `with_header=False`
    to append to a table already printed.
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


def _capture(
    config: BenchConfig, stream,
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]:
    """Capture one config's compute into a graph that can be replayed.

    The launches before recording flush anything a call does only once -- a
    workspace, a handle -- which would otherwise land inside the graph.
    """
    inputs = config.make_inputs()

    def invoke():
        return config.compute(*inputs)

    if config.verify is not None:
        config.verify(invoke(), inputs)

    with torch.cuda.stream(stream):
        for _ in range(_GRAPH_WARMUP):
            invoke()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        for _ in range(_GRAPH_LAUNCHES):
            invoke()
    stream.synchronize()
    # A CUDA graph records input addresses, not the Python objects that own
    # them.  Keep the tensors alive alongside the graph so the allocator
    # cannot reuse their storage while the benchmark suite is being captured.
    return graph, inputs


def run_bench(configs: list[BenchConfig]) -> list[BenchResult]:
    """Time every row once per round, on one stream.

    Interleaving is the point: warmed and timed row by row instead, the first
    would run on a cold card and the last would find the clocks already up.
    Rounds rotate their starting row for the same reason.
    """
    stream = torch.cuda.current_stream()
    graphs = []
    graph_inputs = []
    for i, config in enumerate(configs, 1):
        graph, inputs = _capture(config, stream)
        graphs.append(graph)
        graph_inputs.append(inputs)
        # Capturing a long suite takes minutes; say where it is up to.
        print(f"[capture {i}/{len(configs)}] {config.name}",
              file=sys.stderr, flush=True)

    def replay_all(offset: int = 0) -> list[float]:
        """Replay and time every row once, starting at `offset`."""
        times = [0.0] * len(graphs)
        for i in range(len(graphs)):
            index = (offset + i) % len(graphs)
            times[index] = _time_graph(graphs[index], stream)
        return times

    # A budget, not a fixed count, so a slow config is not warmed a hundred
    # times over. One round replays every config, so it is spent against their
    # sum -- unlike `rounds` below, which sizes off the slowest.
    per_round_ms = max(sum(replay_all()), 1e-3) * _GRAPH_LAUNCHES
    for _ in range(max(1, min(_WARMUP_ITERS // _GRAPH_LAUNCHES,
                              round(_WARMUP_S * 1e3 / per_round_ms)))):
        replay_all()

    # Size the run off the slowest config, priced warm.
    slowest = max(max(replay_all()), 1e-3) * _GRAPH_LAUNCHES
    rounds = max(3, min(_MAX_ROUNDS,
                        round(_MIN_RUN_TIME_S * 1e3 / slowest)))

    # Rotate the starting row each round, so no row sits permanently first
    # (cold) or permanently behind the same neighbour.
    by_round = [replay_all(r) for r in range(rounds)]
    # A round per entry, transposed to a row per entry. Strict: a lenient zip
    # would answer a mismatch by dropping rows off the end of the table.
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
