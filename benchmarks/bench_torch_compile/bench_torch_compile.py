"""Compare stock ``torch.compile`` with the same graph using Ayla kernels.

The benchmark exports the concrete callable and inputs with Torch Dynamo to
obtain an Aten FX graph. Ayla's Inductor lowering writes the originating FX
node name beside each emitted custom-op call. Those annotations drive both the
``>>>`` marks in ``--show-graph`` output and the mapping from captured regions
to generated kernels.

End-to-end latency uses the CUDA graph and CUDA-event method in
``benchmarks.common``. Per-region latency is not measured by extracting and
rerunning a subgraph: both complete compiled graphs run uninterrupted under
``torch.profiler`` and the benchmark attributes each launch's CUDA device time
back to its source FX nodes using the generated Inductor wrapper metadata.
Torch profiler uses Kineto/CUPTI for this trace; Nsight Compute (NCU) is not
required. Profiling samples are collected separately from the end-to-end
samples so profiler overhead does not affect the headline comparison.

What gets measured lives in ``corpus/``, one module per graph.

    python -m benchmarks.bench_torch_compile.bench_torch_compile --show-graph
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code
from torch.fx import GraphModule
from torch.profiler import ProfilerActivity, profile

from ayla import torch_compile
from benchmarks.common import BenchConfig, BenchResult, print_results, run_bench

# TODO: Imports every corpus module, each building its configs at import
# time. Select and load on demand once that gets slow.
from .corpus import build_configs

_AYLA_KERNEL_PATTERN = re.compile(
    r"^\s*# ayla nodes: ([A-Za-z0-9_,]+)\n"
    r"\s*torch\.ops\.ayla\.([A-Za-z0-9_]+)",
    re.MULTILINE,
)
_SOURCE_NODES_PATTERN = re.compile(
    r"# (?:Topologically Sorted|Unsorted) Source Nodes: \[([^]]*)\]"
)
_AYLA_NODES_PATTERN = re.compile(r"# ayla nodes: ([A-Za-z0-9_,]+)")
_TRITON_CALL_PATTERN = re.compile(r"\b(triton_[A-Za-z0-9_]+)\.run\(")
_EXTERN_CALL_PATTERN = re.compile(r"\bextern_kernels\.([A-Za-z0-9_]+)\(")
_AYLA_CALL_PATTERN = re.compile(
    r"\btorch\.ops\.ayla\.([A-Za-z0-9_]+)\.default\("
)
_PROFILE_ROUNDS = 20


@dataclass(frozen=True)
class _ProfileRegion:
    requested_nodes: tuple[str, ...]
    compared_nodes: tuple[str, ...]
    launch_indices: tuple[tuple[int, ...], ...]


def capture_aten_graph(fn, inputs) -> GraphModule:
    """Export the exact callable and inputs that will be benchmarked."""
    return torch._dynamo.export(fn, aten_graph=True)(*inputs).graph_module


def selected_ayla_kernels(
    codes: Sequence[str],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Read source FX nodes and their selected kernels from generated code."""
    return tuple(
        (tuple(match.group(1).split(",")), match.group(2))
        for code in codes
        for match in _AYLA_KERNEL_PATTERN.finditer(code)
    )


def format_captured_graph(
    graph_module: GraphModule, captured_node_names: set[str],
) -> str:
    """Mark nodes selected for Ayla directly in the captured FX graph."""
    lines = []
    for line in str(graph_module.graph).splitlines():
        match = re.match(r"^\s+%([A-Za-z0-9_]+)\s", line)
        lines.append(
            f">>> {line.lstrip()}"
            if match is not None and match.group(1) in captured_node_names
            else line
        )
    return "\n".join(lines)


def _compile(fn, inputs, *, use_ayla: bool):
    """Compile inside the install window and retain generated source."""
    compiled = torch.compile(fn, dynamic=False)
    if use_ayla:
        torch_compile.install()
    try:
        output, codes = run_and_get_code(compiled, *inputs)
    finally:
        torch_compile.uninstall()
    return compiled, output, codes


def profiled_launches(
    codes: Sequence[str],
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Map generated launches to their profiler names and direct FX nodes.

    Inductor propagates a producer's source-node metadata into downstream
    launches. When an earlier launch's complete source set is contained in a
    later launch's set, those earlier nodes are provenance rather than work
    performed by the later launch and are removed.
    """
    launches = []
    for code in codes:
        source_nodes: frozenset[str] = frozenset()
        for line in code.splitlines():
            if match := _SOURCE_NODES_PATTERN.search(line):
                source_nodes = frozenset(
                    name.strip() for name in match.group(1).split(",")
                    if name.strip()
                )
                continue
            if match := _AYLA_NODES_PATTERN.search(line):
                source_nodes = frozenset(match.group(1).split(","))
                continue
            event_name = None
            if match := _TRITON_CALL_PATTERN.search(line):
                event_name = match.group(1)
            elif match := _EXTERN_CALL_PATTERN.search(line):
                event_name = f"aten::{match.group(1)}"
            elif match := _AYLA_CALL_PATTERN.search(line):
                event_name = f"ayla::{match.group(1)}"
            if event_name is not None:
                launches.append((event_name, source_nodes))
                source_nodes = frozenset()
    direct_launches = []
    previous_source_sets: list[frozenset[str]] = []
    for event_name, source_nodes in launches:
        inherited = frozenset().union(*(
            previous
            for previous in previous_source_sets
            if previous < source_nodes
        ))
        direct_launches.append((event_name, source_nodes - inherited))
        previous_source_sets.append(source_nodes)
    return tuple(direct_launches)


def _comparison_regions(
    launches_by_variant: tuple[
        tuple[tuple[str, frozenset[str]], ...], ...
    ],
    regions: tuple[tuple[str, ...], ...],
    node_order: tuple[str, ...],
) -> tuple[_ProfileRegion, ...]:
    """Close requested regions over complete launches in every variant."""
    result = []
    seen_regions: set[frozenset[str]] = set()
    for requested in regions:
        compared = set(requested)
        changed = True
        while changed:
            changed = False
            for launches in launches_by_variant:
                for _event, sources in launches:
                    if compared & sources and not sources <= compared:
                        compared.update(sources)
                        changed = True

        indices_by_variant = []
        for launches in launches_by_variant:
            owners = {
                node: tuple(
                    index
                    for index, (_event, sources) in enumerate(launches)
                    if node in sources
                )
                for node in compared
            }
            missing = sorted(node for node in requested if not owners[node])
            ambiguous = sorted(
                node for node, indices in owners.items() if len(indices) > 1
            )
            if missing or ambiguous:
                raise AssertionError(
                    f"profiler could not map region {requested}: "
                    f"missing={missing}, ambiguous={ambiguous}"
                )
            indices_by_variant.append(tuple(
                index
                for index, (_event, sources) in enumerate(launches)
                if compared & sources
            ))

        key = frozenset(compared)
        if key in seen_regions:
            continue
        seen_regions.add(key)
        ordered = tuple(name for name in node_order if name in compared)
        ordered += tuple(sorted(compared - set(ordered)))
        result.append(_ProfileRegion(
            requested_nodes=requested,
            compared_nodes=ordered,
            launch_indices=tuple(indices_by_variant),
        ))
    return tuple(result)


def _profiled_result(name: str, samples_us: list[float]) -> BenchResult:
    samples_ms = [sample / 1e3 for sample in samples_us]
    return BenchResult(
        name=name,
        rounds=len(samples_ms),
        iters=1,
        median_ms=statistics.median(samples_ms),
        mean_ms=statistics.fmean(samples_ms),
        stdev_ms=statistics.stdev(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def _profile_regions(
    config: BenchConfig,
    inputs: Sequence[object],
    compiled,
    compiled_codes: Sequence[str],
    ayla,
    ayla_codes: Sequence[str],
    selected_kernels: tuple[tuple[tuple[str, ...], str], ...],
    node_order: tuple[str, ...],
) -> list[BenchResult]:
    """Attribute CUPTI device time while repeatedly running each complete graph."""
    regions = tuple(nodes for nodes, _kernel in selected_kernels)
    variants = (
        ("torch.compile", compiled, profiled_launches(compiled_codes)),
        ("Ayla", ayla, profiled_launches(ayla_codes)),
    )
    profile_regions = _comparison_regions(
        tuple(launches for _name, _fn, launches in variants),
        regions,
        node_order,
    )

    for _ in range(5):
        compiled(*inputs)
        ayla(*inputs)
    torch.cuda.synchronize()

    call_order = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for round_index in range(_PROFILE_ROUNDS):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for variant_index in order:
                variants[variant_index][1](*inputs)
                call_order.append(variant_index)
        torch.cuda.synchronize()

    expected_names = [
        event
        for variant_index in call_order
        for event, _sources in variants[variant_index][2]
    ]
    expected_set = set(expected_names)
    events = [
        event for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CPU
        and event.name in expected_set
    ]
    actual_names = [event.name for event in events]
    if actual_names != expected_names:
        raise AssertionError(
            "profiler launch sequence does not match generated wrappers: "
            f"expected {expected_names}, got {actual_names}"
        )

    samples = [
        [[] for _region in profile_regions]
        for _variant in variants
    ]
    cursor = 0
    for variant_index in call_order:
        launches = variants[variant_index][2]
        call_events = events[cursor:cursor + len(launches)]
        cursor += len(launches)
        for region_index, region in enumerate(profile_regions):
            launch_indices = region.launch_indices[variant_index]
            samples[variant_index][region_index].append(sum(
                call_events[index].device_time_total
                for index in launch_indices
            ))

    results = []
    for region_index, region in enumerate(profile_regions):
        region_name = ", ".join(
            f"%{name}" for name in region.requested_nodes
        )
        if set(region.compared_nodes) != set(region.requested_nodes):
            compared = ", ".join(f"%{name}" for name in region.compared_nodes)
            region_name += f" (expanded to {compared})"
        for variant_index, (variant_name, _fn, _launches) in enumerate(variants):
            results.append(_profiled_result(
                f"{config.name} in-context region {region_name} {variant_name}",
                samples[variant_index][region_index],
            ))
    return results


def benchmark(config: BenchConfig, *, show_graph: bool = False) -> list[BenchResult]:
    """Compile and time one complete graph and each selected region."""
    assert config.make_inputs is not None
    inputs = config.make_inputs()
    graph_module = capture_aten_graph(config.compute, inputs)

    # Two lambdas, not one: dynamo caches on the code object.
    compiled, expected, compiled_codes = _compile(
        lambda *args: config.compute(*args), inputs, use_ayla=False)
    ayla, actual, codes = _compile(
        lambda *args: config.compute(*args), inputs, use_ayla=True)

    selected_kernels = selected_ayla_kernels(codes)
    captured_node_names = {
        name for names, _kernel in selected_kernels for name in names
    }
    if show_graph:
        print(format_captured_graph(graph_module, captured_node_names))
    print("Ayla kernels: {")
    for names, kernel in selected_kernels:
        print(f"  {', '.join(f'%{name}' for name in names)}: {kernel},")
    print("}")
    if selected_kernels:
        torch.cuda.synchronize()
        if config.compare_outputs is None:
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        else:
            config.compare_outputs(actual, expected)

    end_to_end = [
        BenchConfig(name=f"{config.name} end-to-end torch.compile",
                    make_inputs=config.make_inputs, compute=compiled),
    ]
    if selected_kernels:
        end_to_end.append(BenchConfig(
            name=f"{config.name} end-to-end torch.compile + Ayla",
            make_inputs=config.make_inputs,
            compute=ayla,
        ))
    results = run_bench(end_to_end)
    if selected_kernels:
        results.extend(_profile_regions(
            config, inputs, compiled, compiled_codes, ayla, codes,
            selected_kernels,
            tuple(node.name for node in graph_module.graph.nodes),
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-k", "--filter", default=None,
        help="only benchmark configs whose names contain this text",
    )
    parser.add_argument(
        "--show-graph", action="store_true",
        help="print the complete captured Aten FX graph",
    )
    parser.add_argument(
        "--muse-num-tokens", type=int,
        help="benchmark one Muse block with this concrete input size",
    )
    args = parser.parse_args()

    configs = [
        config
        for config in build_configs(muse_num_tokens=args.muse_num_tokens)
        if args.filter is None or args.filter in config.name
    ]
    if not configs:
        parser.error(f"no benchmark config matches {args.filter!r}")
    # Both span the run: the backend is outside the cache key, and dynamo
    # guards on grad mode.
    with (
        dynamo_config.patch(
            recompile_limit=max(dynamo_config.recompile_limit, len(configs)),
        ),
        inductor_config.patch(force_disable_caches=True),
        torch.no_grad(),
    ):
        results = [
            result
            for config in configs
            for result in benchmark(config, show_graph=args.show_graph)
        ]
        print_results(results)


if __name__ == "__main__":
    main()
