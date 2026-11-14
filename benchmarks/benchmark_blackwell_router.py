"""Geomean of the shipped Blackwell router against cuBLAS (inference-v2).

Builds every route of ayla/better_benchmark_gemm_router_blackwell.json, checks
it against an fp32 reference on NaN-poisoned output, and times it against
torch.mm(a, b.T) with the inference-v2 protocol (256 MiB cold reset, three
batches of 15 rounds in alternating order). ``transpose_output`` routes build
C^T = B A^T and launch on the column-major view of C.

    python benchmarks/benchmark_blackwell_router.py --shard 0/6 --out shard0.json
    python benchmarks/benchmark_blackwell_router.py --summarize shard*.json
"""

import argparse
import json
import math
import statistics
from pathlib import Path

ROUTER = Path(__file__).resolve().parents[1] / "ayla/better_benchmark_gemm_router_blackwell.json"


def measure(route):
    import torch
    from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher
    from benchmarks import common

    m, n, k = route["m"], route["n"], route["k"]
    config = dict(route["configuration"])
    transpose = config.pop("transpose_output", False)
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=n if transpose else m, n=m if transpose else n, k=k, dtype="f16", **config))
    raw = make_ptx_launcher(kernel)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    launch = (lambda x, y: raw(y, x, c.T)) if transpose else (lambda x, y: raw(x, y, c))
    c.fill_(float("nan"))
    launch(a, b)
    reference = a.float() @ b.float().T
    error = ((c.float() - reference).square().mean() / reference.square().mean()).sqrt().item()
    if not (torch.isfinite(c).all().item() and error < 2e-3):
        raise AssertionError(f"{m}x{n}x{k}: relative RMS error {error}")
    del reference
    benches = [
        common.GemmBenchConfig(name=name, compute=compute, make_inputs=lambda: [a, b],
                               weight_inputs=(1,), m=m, n=n, k=k, dtype=torch.float16)
        for name, compute in (("ayla", lambda x, y: (launch(x, y), c)[1]),
                              ("cuBLAS", lambda x, y: torch.mm(x, y.T, out=c)))
    ]
    common._MAX_ROUNDS = 15
    samples = {"ayla": [], "cuBLAS": []}
    for repeat in range(3):
        for result in common.run_bench(benches if repeat % 2 == 0 else benches[::-1],
                                       inference=True, cold_cache_bytes=256 * 1024**2):
            samples[result.name].append(result.median_ms)
    ayla, cublas = statistics.median(samples["ayla"]), statistics.median(samples["cuBLAS"])
    return dict(ayla_ms=ayla, cublas_ms=cublas, speedup=cublas / ayla)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1", help="i/n: measure routes with index %% n == i")
    parser.add_argument("--out", type=Path, default=Path("blackwell_router.json"))
    parser.add_argument("--summarize", nargs="*", type=Path)
    args = parser.parse_args()
    if args.summarize is not None:
        rows = {}
        for path in args.summarize:
            rows.update(json.loads(path.read_text()))
        speedups = [row["speedup"] for row in rows.values()]
        geomean = math.exp(sum(map(math.log, speedups)) / len(speedups))
        print(f"{len(speedups)} routes: geomean {100 * geomean:.2f}% of cuBLAS")
        return
    index, count = map(int, args.shard.split("/"))
    routes = json.loads(ROUTER.read_text())["routes"]
    rows = json.loads(args.out.read_text()) if args.out.exists() else {}
    for i, route in enumerate(routes):
        key = f"{route['m']}x{route['n']}x{route['k']}"
        if i % count != index or key in rows:
            continue
        rows[key] = measure(route)
        args.out.write_text(json.dumps(rows, indent=1))
        print(key, f"{100 * rows[key]['speedup']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
