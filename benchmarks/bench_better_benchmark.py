"""Run better-benchmark's full-graph suite with and without the ayla backend.

    python benchmarks/bench_better_benchmark.py --models-root repros/models/hf/infer
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import print_results

BENCH = Path(__file__).resolve().parents[1] / "third_party" / "better-benchmark"


@dataclass(frozen=True)
class Row:
    model: str
    stock_us: float
    ayla_us: float
    delta_pct: float


def _sweep(models_root: str, gpus: str, output: Path, cache: Path,
           worker_init: list[str]) -> dict[str, float]:
    cmd = [
        sys.executable, "scripts/bench_parallel.py", "--full-graphs",
        "--models-root", models_root, "--gpus", gpus, "--output", str(output),
    ]
    for spec in worker_init:
        cmd += ["--worker-init", spec]
    # A private cache per arm: the backend is not part of the Inductor cache
    # key, so a shared one would serve this arm the other arm's graphs.
    subprocess.run(cmd, cwd=BENCH, check=True,
                   env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(cache)})
    return {
        name: result["default"]["compiled_us"]
        for name, result in json.loads(output.read_text()).items()
        if name.startswith("repros/")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default="repros/models")
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stock = _sweep(args.models_root, args.gpus, tmp / "stock.json",
                       tmp / "cache_stock", [])
        ayla = _sweep(args.models_root, args.gpus, tmp / "ayla.json",
                      tmp / "cache_ayla", ["ayla.torch_compile:install"])

    print_results([
        Row(model=name.split("models/")[-1],
            stock_us=stock[name],
            ayla_us=ayla[name],
            delta_pct=(ayla[name] - stock[name]) / stock[name] * 100)
        for name in sorted(stock.keys() & ayla.keys())
    ])


if __name__ == "__main__":
    main()
