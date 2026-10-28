"""Run better-benchmark's full-graph suite with and without the ayla backend.

    python benchmarks/bench_better_benchmark.py --models-root repros/models/hf/infer

Graphs are compared like ``bench_parallel.py --compare``: each graph's
coordinate-descent time (else its default compiled time), summarized as the
geomean of per-graph speedups.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import print_results, require_consolidated_benchmarks

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "third_party" / "better-benchmark"


@dataclass(frozen=True)
class Row:
    model: str
    stock_us: float
    ayla_us: float
    delta_pct: float


def _sweep(bench: Path, models_root: str, gpus: str, output: Path, cache: Path,
           worker_init: list[str]) -> dict[str, float]:
    cmd = [
        sys.executable, "scripts/bench_parallel.py", "--full-graphs",
        "--models-root", models_root, "--gpus", gpus, "--output", str(output),
    ]
    for spec in worker_init:
        cmd += ["--worker-init", spec]
    # A private cache per arm: the backend is not part of the Inductor cache
    # key, so a shared one would serve this arm the other arm's graphs.
    # Workers must import this checkout's ayla, not an installed one.
    path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    subprocess.run(cmd, cwd=bench, check=True, env={
        **os.environ, "TORCHINDUCTOR_CACHE_DIR": str(cache), "PYTHONPATH": path,
    })
    return {
        name: result["default"].get("coord_descent_us")
        or result["default"]["compiled_us"]
        for name, result in json.loads(output.read_text()).items()
        if name.startswith("repros/") and "default" in result
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default="repros/models")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--bench-root", type=Path, default=BENCH)
    parser.add_argument("--output-dir", type=Path,
                        help="keep each arm's bench_parallel JSON here")
    args = parser.parse_args()
    require_consolidated_benchmarks()
    args.bench_root = args.bench_root.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = args.output_dir or tmp
        out.mkdir(parents=True, exist_ok=True)
        stock = _sweep(args.bench_root, args.models_root, args.gpus,
                       out / "stock.json", tmp / "cache_stock", [])
        ayla = _sweep(args.bench_root, args.models_root, args.gpus,
                      out / "ayla.json", tmp / "cache_ayla",
                      ["ayla.torch_compile:install"])

    print_results([
        Row(model=name.split("models/")[-1],
            stock_us=stock[name],
            ayla_us=ayla[name],
            delta_pct=(ayla[name] - stock[name]) / stock[name] * 100)
        for name in sorted(stock.keys() & ayla.keys())
    ])
    common = stock.keys() & ayla.keys()
    if not common:
        print(
            "No graphs have timings in both arms; "
            f"missing with ayla: {sorted(stock.keys() - ayla.keys())}"
        )
        return
    speedups = [stock[name] / ayla[name] for name in common]
    print(
        f"geomean speedup over {len(common)} graphs: "
        f"{math.exp(sum(map(math.log, speedups)) / len(speedups)):.4f}x; "
        f"missing with ayla: {sorted(stock.keys() - ayla.keys())}"
    )


if __name__ == "__main__":
    main()
