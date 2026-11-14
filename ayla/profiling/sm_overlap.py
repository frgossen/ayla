"""Capture and plot resident/static/reference schedules for a supplied workload.

    data = profile_workload(compute, inputs, output=".benchmarks/my_workload")
    plot_capture(data, ".benchmarks/my_workload")

Matplotlib is imported only by plot_capture. SM bars are observed execution
regions; local pipeline waits stay inside work bars. PyTorch Profiler exposes
reference kernel spans, not per-SM residency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import ctypes
import json
from pathlib import Path
import statistics
from typing import cast

import torch

from ayla.base.ir import clone_op
from ayla.gpu_dialect import KernelOp, ModuleOp
from ayla.jit_compiler import ScratchBuffer
from ayla.ptx_codegen import PtxLoweringPipeline, make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher, memset_zero
from ayla.tile_scheduler import compile_workload, _workload_device
from ayla.torch_custom_call import to_torch_dtype
from ._sm_trace import (
    CAPACITY, COLORS, _bench_batch, _capture_batch, _decode, _groups,
    _instrument, _launch_activity_summary, _phase_kind, _profile,
    _tag_resident_tasks, _tag_roots, _work_segments,
)


def _resources(launcher):
    launcher.ensure_loaded()
    query = ctypes.CDLL("libcuda.so.1").cuFuncGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    query.restype = ctypes.c_int
    result = {}
    for name, attribute in (("shared_bytes", 1), ("local_bytes", 3), ("registers", 4)):
        value = ctypes.c_int()
        status = query(ctypes.byref(value), attribute, ctypes.c_void_p(launcher._function))
        if status:
            raise RuntimeError(f"cuFuncGetAttribute({attribute}) failed: {status}")
        result[name] = value.value
    return result


@dataclass
class _Captured:
    data: dict
    graph: object
    batch: object
    # Retain launchers, allocations and graph owners until all captures finish.
    owners: object


def _capture_ayla(compute, inputs, expected, *, scheduler, name, out,
                  samples, replays, capacity, rtol, atol):
    sources = []
    lowered = []

    def emit(module):
        _tag_roots(module)
        _tag_resident_tasks(module)
        sources.append(clone_op(module, {}))
        result = PtxLoweringPipeline().run(module)
        lowered.append(clone_op(result, {}))
        return make_ptx_kernel(result, lowered=True)

    workload = compile_workload(compute, inputs, scheduler=scheduler,
                                name=f"ayla_{name}", emit_kernel=emit)
    actual = workload(*inputs)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    saved_outputs = [t.clone() for t in workload.outputs]
    [route] = workload.plan.routes
    source = cast(ModuleOp, sources[0])
    cached = cast(ModuleOp, lowered[0])
    if make_ptx_kernel(cast(ModuleOp, clone_op(cached, {})), lowered=True).ptx != route.kernel.ptx:
        raise AssertionError("cached lowering changed the measured kernel")
    observer = make_ptx_launcher(_instrument(source, cached, capacity=capacity))
    scratch = {
        b: torch.zeros(b.tensor_type.static_shape, device=workload.device,
                       dtype=to_torch_dtype(b.tensor_type.dtype))
        for b in route.bindings if isinstance(b, ScratchBuffer)
    }
    tensors = workload.arguments(inputs)
    arguments = [tensors[b] if isinstance(b, int) else scratch[cast(ScratchBuffer, b)] for b in route.bindings]
    ticks = torch.zeros((route.kernel.grid_dim[0], capacity + 1, 2),
                        dtype=torch.int64, device=workload.device)

    def observe(frame=ticks):
        for buffer, tensor in scratch.items():
            if buffer.reset_each_launch:
                memset_zero(tensor)
        observer(*arguments, frame)

    observe()
    torch.cuda.synchronize()
    if any(not torch.equal(a, b) for a, b in zip(workload.outputs, saved_outputs, strict=True)):
        raise AssertionError(f"{name}: instrumentation changed outputs")
    def plain():
        return workload(*inputs)
    plain_graph = _capture_batch(plain, replays=1)
    batch = _capture_batch(plain, replays=replays)
    observer_batch = _capture_batch(observe, replays=replays)
    frames = [torch.empty_like(ticks) for _ in range(samples)]
    observation = torch.cuda.CUDAGraph()
    with torch.cuda.graph(observation):
        for frame in frames:
            observe(frame)
    observation.replay()
    torch.cuda.synchronize()
    decoded = [_decode(frame) for frame in frames]
    intervals, lifetimes = sorted(decoded, key=lambda item: max(r["end_ms"] for r in item[1]))[samples // 2]
    roots = _tag_roots(source) or _tag_resident_tasks(source) or [
        dict(root=i, kind=_phase_kind(group))
        for i, group in enumerate(_groups(cast(KernelOp, source.body.ops[0]).body))
    ]
    root_types = {row["root"]: row["kind"] for row in roots}
    for region in intervals:
        if region["kind"] == "other work" and region.get("root") in root_types:
            region["kind"] = "other GEMM" if root_types[region["root"]] == "gemm" else "pointwise / norm"
    metadata = cast(KernelOp, source.body.ops[0]).attrs
    report = dict(resources=_resources(make_ptx_launcher(route.kernel)),
                  observer_resources=_resources(observer),
                  observer_outputs_bitwise_equal=True, roots=roots,
                  **{k: v for k, v in metadata.items() if k.startswith("tile_")})
    trace_path = out / f"{name}_torch_trace.json"
    trace = _profile(plain_graph.replay, trace_path)
    report["device_operations_per_invocation"] = _launch_activity_summary(trace_path)
    return _Captured(
        dict(report=report, profiler_kernels=trace, intervals=intervals, lifetimes=lifetimes),
        plain_graph, batch,
        (workload, observer, scratch, frames, observation, observer_batch),
    )


def profile_workload(
    compute: Callable,
    inputs: Sequence[torch.Tensor | int],
    *,
    output: str | Path,
    reference: Callable | None = None,
    name: str = "workload",
    samples: int = 3,
    rounds: int = 5,
    replays: int = 100,
    capacity: int = CAPACITY,
    rtol: float = 0.015,
    atol: float = 0.015,
) -> dict:
    """Compare the same workload with resident and static Ayla and a reference.

    ``compute`` supplies both the FX graph and correctness baseline. Optional
    ``reference`` is the timed implementation (e.g. torch.compile(compute)).
    Inputs must satisfy compile_workload's static inference requirements.
    Saves raw profiler traces and one JSON capture; call plot_capture to render.
    """
    for label, value in (("samples", samples), ("rounds", rounds),
                         ("replays", replays), ("capacity", capacity)):
        if type(value) is not int or value < (3 if label == "capacity" else 1):
            raise ValueError(f"{label} must be a positive integer" if label != "capacity" else "capacity must be at least 3")
    inputs = tuple(inputs)
    device = _workload_device(inputs)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with torch.cuda.device(device), torch.no_grad():
        expected = compute(*inputs)
        variants = {}
        for variant, scheduler in (("resident", None), ("static", "static")):
            variants[variant] = _capture_ayla(
                compute, inputs, expected, scheduler=scheduler, name=variant, out=out,
                samples=samples, replays=replays, capacity=capacity, rtol=rtol, atol=atol,
            )
        reference = reference or compute
        torch.testing.assert_close(reference(*inputs), expected, rtol=rtol, atol=atol)
        def run_reference():
            return reference(*inputs)
        ref_graph = _capture_batch(run_reference, replays=1)
        ref_batch = _capture_batch(run_reference, replays=replays)
        path = out / "reference_torch_trace.json"
        variants["reference"] = _Captured(
            dict(report={},
                 profiler_kernels=_profile(ref_graph.replay, path), intervals=[], lifetimes=[]),
            ref_graph, ref_batch, (reference, inputs),
        )
        variants["reference"].data["report"]["device_operations_per_invocation"] = _launch_activity_summary(path)
        # Rotate timing order to avoid comparing different thermal/cache intervals.
        timing = {variant: [] for variant in variants}
        observer_timing = {variant: [] for variant in ("resident", "static")}
        for trial in range(rounds):
            order = list(variants)
            order = order[trial % len(order):] + order[:trial % len(order)]
            for variant in order:
                timing[variant].append(_bench_batch(variants[variant].batch, replays))
                if variant != "reference":
                    observer_timing[variant].append(_bench_batch(variants[variant].owners[-1], replays))
        for variant, captured in variants.items():
            report = captured.data["report"]
            report.update(plain_ms=statistics.median(timing[variant]),
                          plain_samples_ms=timing[variant], replays=replays,
                          timing_method="captured_batch")
            if variant != "reference":
                observed = statistics.median(observer_timing[variant])
                overhead = 100 * (observed / report["plain_ms"] - 1)
                report.update(observer_ms=observed, observer_samples_ms=observer_timing[variant],
                              observer_overhead_pct=overhead, timing_qualified=abs(overhead) < 2)
        data = dict(name=name, gpu=torch.cuda.get_device_name(),
                    reference_sm_activity_available=False,
                    variants={key: value.data for key, value in variants.items()})
        serialized = json.dumps(data, indent=2) + "\n"
        (out / "sm_overlap.json").write_text(serialized)
        return json.loads(serialized)


def plot_capture(data: dict, output: str | Path) -> None:
    """Render saved capture data without requiring a GPU or re-running kernels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    variants = data["variants"]
    profiler_end = max(event["start_ms"] + event["duration_ms"]
                       for variant in variants.values() for event in variant["profiler_kernels"])
    observed_end = max((region["end_ms"] for variant in variants.values()
                        for region in variant["lifetimes"]), default=profiler_end)
    figure, axes = plt.subplots(2, len(variants), figsize=(6 * len(variants), 8),
                                squeeze=False, constrained_layout=True,
                                gridspec_kw={"height_ratios": [1, 4]})
    for column, (name, variant) in enumerate(variants.items()):
        top, bottom = axes[:, column]
        trace = variant["profiler_kernels"]
        for event in trace:
            top.broken_barh([(event["start_ms"], event["duration_ms"])], (-.35, .7),
                            facecolors="#4b596b")
        report = variant["report"]
        top.set(title=f"{name} · {report['plain_ms']:.3f} ms · {len(trace)} compute launches",
                yticks=[], ylim=(-.6, .6), xlim=(0, profiler_end * 1.02),
                xlabel="PyTorch Profiler kernel spans (ms)")
        if variant["lifetimes"]:
            regions = _work_segments(variant["intervals"])
            for region in regions:
                bottom.broken_barh([(region["start_ms"], region["end_ms"] - region["start_ms"])],
                                   (region["sm"] - .42, .84),
                                   facecolors=COLORS.get(region["kind"], "#65a69a"), linewidth=0)
            smids = sorted({region["sm"] for region in variant["lifetimes"]})
            bottom.set(ylim=(min(smids)-1, max(smids)+1), ylabel="Physical SM ID",
                       xlim=(0, observed_end * 1.02),
                       xlabel="Observed schedule time (ms)", yticks=smids[::8],
                       title=f"%smid / %globaltimer · {len(smids)} SMs\n"
                       f"Probe timing change {report.get('observer_overhead_pct', 0):+.1f}%"
                       + (" · timing unqualified" if report.get("timing_qualified") is False else ""))
        else:
            for row, event in enumerate(trace):
                lower = event["name"].lower()
                kind = "other GEMM" if any(s in lower for s in ("gemm", "xmma", "cublas", "cutlass", "nvjet_")) else "pointwise / norm"
                bottom.broken_barh([(event["start_ms"], event["duration_ms"])],
                                   (row-.35, .7), facecolors=COLORS[kind])
            bottom.set(ylim=(len(trace)-.3, -.7), yticks=range(len(trace)),
                       yticklabels=range(1, len(trace)+1), xlim=(0, profiler_end * 1.02),
                       ylabel="CUDA kernel sequence", xlabel="PyTorch Profiler time (ms)",
                       title="Reference kernel sequence\nPer-SM residency unavailable")
        for ax in (top, bottom):
            ax.set_xlim(left=0)
            ax.grid(axis="x", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{data.get('name', 'workload')} · {data['gpu']}\n"
                   "SM bars are execution regions; local pipeline waits stay inside work bars. Profiler spans and probes use separate clocks.")
    used = {r["kind"] for v in variants.values() for r in v["intervals"]}
    figure.legend(handles=[Patch(color=color, label=kind) for kind, color in COLORS.items() if kind in used],
                  loc="outside lower center", ncols=4, frameon=False)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out / f"sm_overlap.{suffix}", dpi=160)
    plt.close(figure)
