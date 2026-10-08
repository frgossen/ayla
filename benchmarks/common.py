"""Shared timing harness for the benchmarks.

Configs describe the mathematical GEMM, ``(m, k) @ (k, n)``, but the operands
are built in the layout ayla's kernels take -- the right one transposed, as
``(n, k)``, computing ``a @ bt.T``. Timing an NN GEMM against ayla's NT one
would not be a comparison.
"""

from __future__ import annotations

import csv
import ctypes
import dataclasses
import functools
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

    `weight_inputs` indexes the inputs that are model parameters. Inference
    timing gives every layer its own cold copy of those; the rest are
    activations, rewritten by the preceding op and so resident in L2.
    """

    name: str
    compute: Callable[..., torch.Tensor]
    make_inputs: Callable[[], list[torch.Tensor]] | None = None
    verify: Callable[..., None] | None = None
    weight_inputs: tuple[int, ...] = ()


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
        weight_inputs=(1,), m=m, n=n, k=k, dtype=dtype,
    )


def addmm_bench_config(
    *, name: str, m: int, n: int, k: int, compute: Callable,
    dtype: torch.dtype,
) -> GemmBenchConfig:
    """A GEMM benchmark with a vector-N bias epilogue."""
    def make_inputs() -> list[torch.Tensor]:
        return [
            torch.randn(n, dtype=dtype, device="cuda"),
            torch.randn(m, k, dtype=dtype, device="cuda"),
            torch.randn(n, k, dtype=dtype, device="cuda"),
        ]

    def verify(output: torch.Tensor, _inputs) -> None:
        assert output.dtype == dtype, (
            f"{name}: produced {output.dtype}, config says {dtype}"
        )

    return GemmBenchConfig(
        name=name, compute=compute, make_inputs=make_inputs, verify=verify,
        weight_inputs=(0, 2), m=m, n=n, k=k, dtype=dtype,
    )


def bmm_bench_config(
    *, name: str, batch: int, m: int, n: int, k: int, compute: Callable,
    dtype: torch.dtype,
) -> GemmBenchConfig:
    """A batched GEMM benchmark using contiguous ``A`` and ``BT``.

    Both operands are activations (attention-style), so neither is a weight.
    """
    def make_inputs() -> list[torch.Tensor]:
        return [
            torch.randn(batch, m, k, dtype=dtype, device="cuda"),
            torch.randn(batch, n, k, dtype=dtype, device="cuda"),
        ]

    def verify(output: torch.Tensor, _inputs) -> None:
        assert output.shape == (batch, m, n)
        assert output.dtype == dtype, (
            f"{name}: produced {output.dtype}, config says {dtype}"
        )

    return GemmBenchConfig(
        name=name, compute=compute, make_inputs=make_inputs, verify=verify,
        m=batch * m, n=n, k=k, dtype=dtype,
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
# Inference timing gives each layer its own weights, as many layers as fit this
# budget within the bounds below; a layer's weights are used once per sample.
_INFERENCE_WEIGHT_BYTES = 256 * 1024 * 1024
_INFERENCE_MIN_LAYERS = 4
_INFERENCE_MAX_LAYERS = 32


def torch_mm_nt(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
    """The baseline GEMM in the collection's operand layout: ``a @ bt.T``."""
    return torch.mm(a, bt.t())


def torch_addmm_nt(
    bias: torch.Tensor, a: torch.Tensor, bt: torch.Tensor,
) -> torch.Tensor:
    """The addmm baseline in Ayla's operand layout."""
    return torch.addmm(bias, a, bt.t())


def torch_bmm_nt(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
    """The batched baseline in Ayla's operand layout."""
    return torch.bmm(a, bt.transpose(1, 2))


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


def out_parameter_addmm_op(launcher: Callable) -> Callable:
    """Adapt a bias/A/BT/output launcher to the benchmark interface."""
    output = None

    def op(
        bias: torch.Tensor, a: torch.Tensor, bt: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal output
        if output is None:
            output = torch.empty(
                (a.shape[0], bt.shape[0]), dtype=a.dtype, device=a.device,
            )
        launcher(bias, a, bt, output)
        return output

    return op


def out_parameter_bmm_op(launcher: Callable) -> Callable:
    """Adapt a rank-3 A/BT/output launcher to the benchmark interface."""
    output = None

    def op(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
        nonlocal output
        if output is None:
            output = torch.empty(
                (a.shape[0], a.shape[1], bt.shape[1]),
                dtype=a.dtype,
                device=a.device,
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
    actual = cfg.compute(a, bt)
    expected = torch_mm_nt(a, bt)
    torch.cuda.synchronize()
    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    if max_abs > max_abs_error or relative_rms > max_relative_rms_error:
        raise RuntimeError(
            f"{cfg.name}: max abs {max_abs}, relative RMS {relative_rms}"
        )
    return max_abs, relative_rms


def assert_addmm_against_torch(
    cfg: GemmBenchConfig,
    *,
    seed: int = 0,
    max_abs_error: float = 0,
    max_relative_rms_error: float = 0,
) -> tuple[float, float]:
    """Assert a vector-bias addmm config agrees with Torch."""
    torch.manual_seed(seed)
    bias = torch.randn(cfg.n, dtype=cfg.dtype, device="cuda")
    a = torch.randn((cfg.m, cfg.k), dtype=cfg.dtype, device="cuda")
    bt = torch.randn((cfg.n, cfg.k), dtype=cfg.dtype, device="cuda")
    actual = cfg.compute(bias, a, bt)
    expected = torch_addmm_nt(bias, a, bt)
    torch.cuda.synchronize()
    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    if max_abs > max_abs_error or relative_rms > max_relative_rms_error:
        raise RuntimeError(
            f"{cfg.name}: max abs {max_abs}, relative RMS {relative_rms}"
        )
    return max_abs, relative_rms


def assert_bmm_against_torch(
    cfg: GemmBenchConfig,
    *,
    batch: int,
    m: int,
    seed: int = 0,
    max_abs_error: float = 0,
    max_relative_rms_error: float = 0,
) -> tuple[float, float]:
    """Assert a batched GEMM config agrees with Torch."""
    torch.manual_seed(seed)
    a = torch.randn((batch, m, cfg.k), dtype=cfg.dtype, device="cuda")
    bt = torch.randn((batch, cfg.n, cfg.k), dtype=cfg.dtype, device="cuda")
    actual = cfg.compute(a, bt)
    expected = torch_bmm_nt(a, bt)
    torch.cuda.synchronize()
    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    if max_abs > max_abs_error or relative_rms > max_relative_rms_error:
        raise RuntimeError(
            f"{cfg.name}: max abs {max_abs}, relative RMS {relative_rms}"
        )
    return max_abs, relative_rms


@functools.cache
def _libcuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _reset_l2(cache_flush: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Leave nothing a previous sample touched in L2.

    Lines loaded with an ``evict_last`` policy survive any amount of
    normal-priority traffic: without the reset, a kernel that pins its operands
    finds them resident through the flush, while a kernel that does not reads
    them from DRAM. The reset applies immediately rather than in stream order,
    so it must only be called while the device is idle.
    """
    status = _libcuda().cuCtxResetPersistingL2Cache()
    if status != 0:
        raise RuntimeError(f"cuCtxResetPersistingL2Cache failed: {status}")
    with torch.cuda.stream(stream):
        cache_flush.add_(1)


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    *,
    graph_launches: int,
    cache_flush: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Time one replay: milliseconds per launch, and for the whole replay."""
    if cache_flush is not None:
        _reset_l2(cache_flush, stream)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    elapsed = start.elapsed_time(end)
    return elapsed / graph_launches, elapsed


def _time_inference(
    chain: torch.cuda.CUDAGraph,
    events: list[list[torch.cuda.Event]],
    stream: torch.cuda.Stream,
    *,
    cache_flush: torch.Tensor,
) -> tuple[float, float]:
    """Time one layer chain: the cost per layer, and the whole chain.

    A layer costs its compute, plus how much longer the consumer takes right
    after it than straight after another consumer -- that is, once the output
    is as resident as it can be.
    """
    _, wall = _time_graph(chain, stream, graph_launches=1, cache_flush=cache_flush)
    cost = sum(
        compute.elapsed_time(consume)
        + consume.elapsed_time(reconsume)
        - reconsume.elapsed_time(end)
        for compute, consume, reconsume, end in events
    )
    return cost / len(events), wall


def _consume_output_read_only(result: torch.Tensor) -> torch.Tensor:
    """Model a following op without creating another dirty output pass."""
    return result.sum()


def _with_percentages(
    configs: list[GemmBenchConfig], results: list[GemmBenchResult]
) -> list[GemmBenchResult]:
    """The same rows, with the two relative columns filled in."""
    torch_mm_tflops = next(
        (
            result.tflops
            for cfg, result in zip(configs, results)
            if cfg.compute in (torch_mm_nt, torch_addmm_nt, torch_bmm_nt)
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


def run(
    configs: list[GemmBenchConfig],
    *,
    cold_cache_bytes: int = 0,
    inference: bool = False,
) -> list[GemmBenchResult]:
    """Time GEMM configs, and say what the timing means as throughput.

    :func:`run_bench` does the timing; this adds only the GEMM arithmetic.
    """
    if not configs:
        return []
    timed = run_bench(
        configs, cold_cache_bytes=cold_cache_bytes, inference=inference,
    )
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
    config: BenchConfig, stream, graph_launches: int,
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
        for _ in range(graph_launches):
            invoke()
    stream.synchronize()
    # A CUDA graph records input addresses, not the Python objects that own
    # them.  Keep the tensors alive alongside the graph so the allocator
    # cannot reuse their storage while the benchmark suite is being captured.
    return graph, inputs


def _capture_inference(
    config: BenchConfig, stream,
) -> tuple[torch.cuda.CUDAGraph, list[list[torch.cuda.Event]], list[object]]:
    """Capture a chain of model layers, with events around what is timed.

    Each layer is a producer that rewrites the activations in place, the
    compute on those activations and the layer's own copy of the weights, and
    two read-only consumers of the output. Activations are thus
    in L2 as they would be after the op that made them, and weights are not, as
    they would not be after the rest of a model ran. The producer is not timed:
    what it costs depends on dirty L2 lines the previous layer happens to
    leave, which says more about this synthetic chain than about the compute.
    The consumers are timed, so that a compute which keeps its output out of L2
    is charged for the next op's reads -- see :func:`_time_inference`.
    """
    inputs = config.make_inputs()
    output = config.compute(*inputs)
    if config.verify is not None:
        config.verify(output, inputs)
    weight_bytes = sum(inputs[i].nbytes for i in config.weight_inputs)
    layers = _INFERENCE_MAX_LAYERS if not weight_bytes else max(
        _INFERENCE_MIN_LAYERS,
        min(_INFERENCE_MAX_LAYERS,
            math.ceil(_INFERENCE_WEIGHT_BYTES / weight_bytes)),
    )
    layer_inputs = [inputs] + [
        [
            x.clone() if i in config.weight_inputs else x
            for i, x in enumerate(inputs)
        ]
        for _ in range(layers - 1)
    ]
    activations = [
        x for i, x in enumerate(inputs) if i not in config.weight_inputs
    ]
    events = [
        [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
        for _ in layer_inputs
    ]

    def chain(timed: bool) -> None:
        for layer, layer_events in zip(layer_inputs, events, strict=True):
            for x in activations:
                x.mul_(1)
            compute, consume, reconsume, end = layer_events
            if timed:
                compute.record()
            result = config.compute(*layer)
            if timed:
                consume.record()
            _consume_output_read_only(result)
            if timed:
                reconsume.record()
            _consume_output_read_only(result)
            if timed:
                end.record()

    with torch.cuda.stream(stream):
        chain(timed=False)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        chain(timed=True)
    stream.synchronize()
    return graph, events, layer_inputs


def run_bench(
    configs: list[BenchConfig],
    *,
    cold_cache_bytes: int = 0,
    inference: bool = False,
) -> list[BenchResult]:
    """Time every row once per round, on one stream.

    Interleaving is the point: warmed and timed row by row instead, the first
    would run on a cold card and the last would find the clocks already up.
    Rounds rotate their starting row for the same reason.

    By default a row is timed hot: the same inputs, launch after launch. With
    ``cold_cache_bytes``, one launch after L2 is reset. With ``inference``, as
    one layer of a model: see :func:`_capture_inference`. A sample is then one
    chain, and its time the mean cost of a layer. Inference resets L2
    before each chain, flushing ``cold_cache_bytes`` or twice the L2 size,
    whichever is larger.
    """
    if cold_cache_bytes < 0:
        raise ValueError("cold_cache_bytes must be nonnegative")
    stream = torch.cuda.current_stream()
    if inference:
        cold_cache_bytes = max(
            cold_cache_bytes,
            2 * torch.cuda.get_device_properties(stream.device).L2_cache_size,
        )
    graph_launches = 1 if cold_cache_bytes else _GRAPH_LAUNCHES
    cache_flush = (
        torch.zeros(
            max(1, cold_cache_bytes // 4), dtype=torch.float32, device="cuda",
        )
        if cold_cache_bytes else None
    )
    samplers: list[Callable[[], tuple[float, float]]] = []
    iters: list[int] = []
    # A CUDA graph records input addresses, not the Python objects that own
    # them.  Keep the tensors alive alongside the graph so the allocator
    # cannot reuse their storage while the benchmark suite is being captured.
    keep_alive = []
    for i, config in enumerate(configs, 1):
        if inference:
            assert cache_flush is not None
            chain, events, tensors = _capture_inference(config, stream)
            samplers.append(functools.partial(
                _time_inference, chain, events, stream,
                cache_flush=cache_flush,
            ))
            iters.append(len(events))
            keep_alive.append((chain, events, tensors))
        else:
            graph, inputs = _capture(config, stream, graph_launches)
            samplers.append(functools.partial(
                _time_graph, graph, stream,
                graph_launches=graph_launches, cache_flush=cache_flush,
            ))
            iters.append(graph_launches)
            keep_alive.append((graph, inputs))
        # Capturing a long suite takes minutes; say where it is up to.
        print(f"[capture {i}/{len(configs)}] {config.name}",
              file=sys.stderr, flush=True)

    def replay_all(offset: int = 0) -> tuple[list[float], list[float]]:
        """Time every row once, starting at `offset`: per launch, and wall."""
        times = [0.0] * len(samplers)
        walls = [0.0] * len(samplers)
        for i in range(len(samplers)):
            index = (offset + i) % len(samplers)
            times[index], walls[index] = samplers[index]()
        return times, walls

    # A budget, not a fixed count, so a slow config is not warmed a hundred
    # times over. One round replays every config, so it is spent against their
    # sum -- unlike `rounds` below, which sizes off the slowest.
    per_round_ms = max(sum(replay_all()[1]), 1e-3)
    for _ in range(max(1, min(_WARMUP_ITERS // max(iters),
                              round(_WARMUP_S * 1e3 / per_round_ms)))):
        replay_all()

    # Size the run off the slowest config, priced warm.
    slowest = max(max(replay_all()[1]), 1e-3)
    rounds = max(3, min(_MAX_ROUNDS,
                        round(_MIN_RUN_TIME_S * 1e3 / slowest)))

    # Rotate the starting row each round, so no row sits permanently first
    # (cold) or permanently behind the same neighbour.
    by_round = [replay_all(r)[0] for r in range(rounds)]
    # A round per entry, transposed to a row per entry. Strict: a lenient zip
    # would answer a mismatch by dropping rows off the end of the table.
    by_row = zip(*by_round, strict=True)

    return [
        BenchResult(
            name=config.name,
            rounds=rounds,
            iters=row_iters,
            median_ms=statistics.median(times),
            mean_ms=statistics.mean(times),
            stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_ms=min(times),
            max_ms=max(times),
        )
        for config, row_iters, times in zip(configs, iters, by_row, strict=True)
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
