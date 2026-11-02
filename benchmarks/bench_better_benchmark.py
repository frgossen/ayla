"""Compare native Ayla and stock Inductor on Better-Benchmark saved graphs.

The primary score sums graph latencies within each model/mode workload before
computing the geometric mean. Graph geomean is reported separately. Workers
use fixed GPU assignments, fresh per-arm caches and alternating paired order.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "third_party" / "better-benchmark"


def stock():
    """Give both benchmark arms the same reproducible saved-graph inputs."""
    import torch

    main = sys.modules["__main__"]
    original = main.load_full_graph

    def load(definition, *args, **kwargs):
        torch.manual_seed(int(hashlib.sha256(str(definition).encode()).hexdigest()[:8], 16))
        return original(definition, *args, **kwargs)

    main.load_full_graph = load


def install():
    """Install the native JSON router in each Better-Benchmark worker."""
    stock()
    from ayla.torch_compile import install as native_install

    native_install()


def _sweep(bench, graphs, gpu, output, cache, arm):
    command = [
        sys.executable, str(bench / "scripts/bench_parallel.py"), *graphs,
        "--full-graphs", "--gpus", str(gpu), "--workers-per-gpu", "1",
        "--no-cd", "--no-lock-clocks", "--strict-gpu-lock",
        "--worker-init", f"benchmarks.bench_better_benchmark:{'stock' if arm == 'stock' else 'install'}",
        "--output", str(output),
    ]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(bench)]),
           "TORCHINDUCTOR_CACHE_DIR": str(cache), "TORCH_NO_COMPILER_WRAPPER": "1",
           "OMP_NUM_THREADS": "1"}
    env.pop("AYLA_E2E_CONFIG", None)
    output.with_suffix(".command.json").write_text(json.dumps({
        "command": command, "arm": arm, "native_json_router": arm == "ayla",
        "cache": str(cache), "configuration_override": None,
    }, indent=2) + "\n")
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def _load_times(directory, bench):
    times = {}
    for path in sorted(directory.glob("gpu*.json")):
        if path.name.endswith(".command.json"):
            continue
        for name, row in json.loads(path.read_text()).items():
            if not name.endswith(".py"):
                continue
            graph = Path(name)
            graph = str((graph if graph.is_absolute() else bench / graph).resolve())
            value = row.get("default", {}).get("compiled_us")
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                if graph in times:
                    raise ValueError(f"Duplicate saved graph: {graph}")
                times[graph] = value
    return times


def _geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def _report(out, bench, repetitions, inputs, expected):
    raw = [{arm: _load_times(out / f"rep{rep}" / arm, bench)
            for arm in ("stock", "ayla")} for rep in range(repetitions)]
    cohort = set(expected) if expected is not None else set(raw[0]["stock"])
    if not cohort:
        raise RuntimeError("No stock graphs timed successfully")
    pairs = []
    for rep, arms in enumerate(raw):
        missing = {arm: sorted(cohort - set(times)) for arm, times in arms.items()}
        complete = not any(missing.values())
        graphs = []
        groups = defaultdict(lambda: {"stock_us": 0, "ayla_us": 0, "graphs": []})
        for graph in sorted(cohort):
            if any(graph not in times for times in arms.values()):
                continue
            a, b = arms["stock"][graph], arms["ayla"][graph]
            graphs.append(dict(graph=graph, stock_us=a, ayla_us=b, speedup=a / b))
            group = groups[str(Path(graph).parent)]
            group["stock_us"] += a
            group["ayla_us"] += b
            group["graphs"].append(graph)
        workloads = [dict(workload=work, **values, speedup=values["stock_us"] / values["ayla_us"])
                     for work, values in sorted(groups.items())]
        pairs.append(dict(rep=rep, status="complete" if complete else "incomplete",
                          missing=missing, rows=graphs, workloads=workloads,
                          workload_speedup=_geomean([r["speedup"] for r in workloads]) if complete else None,
                          graph_speedup=_geomean([r["speedup"] for r in graphs]) if complete else None))
    complete = all(p["status"] == "complete" for p in pairs)
    def aggregate(field):
        ratio = _geomean([p[field] for p in pairs]) if complete else None
        return dict(speedup=ratio, gain_percent=(ratio - 1) * 100 if ratio else None)
    result = dict(status="complete" if complete else "incomplete", repetitions=repetitions,
                  corpus_graphs=len(inputs), timed_graphs=len(cohort),
                  workloads=len({str(Path(g).parent) for g in cohort}),
                  preflight_exclusions=len(set(inputs) - cohort),
                  definition="Sum saved-graph latencies within each workload; geometric mean of stock/Ayla workload ratios, then geometric mean across paired repetitions.",
                  native_json_router=True, configuration_overrides=False,
                  all_workloads=aggregate("workload_speedup"), all_graphs=aggregate("graph_speedup"),
                  pairs=pairs)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2), flush=True)
    if not complete:
        raise RuntimeError("Fixed cohort incomplete; missing graphs retained in comparison.json")
    return result


def _run(args, out):
    bench = args.bench_root.resolve()
    if args.graphs_manifest:
        graphs = json.loads(args.graphs_manifest.read_text())["graphs"]
    else:
        search = Path(args.models_root)
        search = search if search.is_absolute() else bench / search
        graphs = [str(p.resolve()) for p in sorted(search.rglob("full_graph_*.py"))]
    graphs = [str(Path(p).resolve()) for p in graphs]
    if not graphs or len(graphs) != len(set(graphs)):
        raise ValueError("Expected a nonempty, unique saved-graph input list")
    expected = json.loads(args.timed_graphs_manifest.read_text())["graphs"] if args.timed_graphs_manifest else None
    if expected is not None and not set(expected) <= set(graphs):
        raise ValueError("Timed cohort must be a subset of the input graph list")
    gpus = args.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(gpus) != len(set(gpus)):
        raise ValueError("Expected unique comma-separated physical GPU indices")
    out.mkdir(parents=True, exist_ok=True)
    source_files = [*sorted((ROOT / "ayla").rglob("*.py")),
                    *sorted((ROOT / "ayla").glob("*.json")), Path(__file__),
                    bench / "scripts/bench_parallel.py"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest = dict(source_hashes=hashes,
                    graph_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in graphs},
                    gpu_assignment={p: gpus[i % len(gpus)] for i, p in enumerate(graphs)},
                    timed_cohort=expected, repetitions=args.repetitions, native_json_router=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Measurement inputs changed; use a fresh output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def worker(index, gpu):
        subset = graphs[index::len(gpus)]
        if not subset:
            return
        for rep in range(args.repetitions):
            arms = ("stock", "ayla") if (index + rep) % 2 == 0 else ("ayla", "stock")
            for arm in arms:
                directory = out / f"rep{rep}" / arm
                directory.mkdir(parents=True, exist_ok=True)
                done = directory / f"gpu{gpu}.done"
                if done.exists():
                    continue
                print(f"Starting pair {rep + 1} {arm} GPU {gpu}", flush=True)
                _sweep(bench, subset, gpu, directory / f"gpu{gpu}.json",
                       directory / f"cache_gpu{gpu}", arm)
                if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
                    raise RuntimeError("Source or router changed during measurement")
                done.write_text("complete\n")
                print(f"Completed pair {rep + 1} {arm} GPU {gpu}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    result = _report(out, bench, args.repetitions, graphs, expected)
    (out / "complete").write_text("complete\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default="repros/models")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--bench-root", type=Path, default=BENCH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--graphs-manifest", type=Path, help="Frozen input list with a graphs array")
    parser.add_argument("--timed-graphs-manifest", type=Path, help="Fixed runnable cohort; missing results make the score incomplete")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.output_dir:
        _run(args, args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory() as temporary:
            _run(args, Path(temporary))


if __name__ == "__main__":
    main()
