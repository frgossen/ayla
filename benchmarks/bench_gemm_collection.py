"""Benchmark every GEMM kernel in the collection on one hand-picked shape."""

from __future__ import annotations

import argparse
import re

import ayla.gemm_collection
import torch

from common import BenchGemmConfig, print_results, run, torch_mm_nt


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GEMM kernels")
    parser.add_argument("-k", "--filter", type=str, default=None,
                        help="Regex to filter benchmarks by label")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Run the benchmarks in reverse order")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    configs = [
        BenchGemmConfig(
            label="torch.mm.bf16",
            m=4096,
            n=4096,
            k=4096,
            op=torch_mm_nt,
            dtype=torch.bfloat16,
        ),
    ]
    capability = torch.cuda.get_device_capability()
    candidates = ayla.gemm_collection.query(
        m=4096, k=4096, n=4096, dtype=torch.bfloat16,
    )
    compatible = [
        kernel for kernel in candidates
        if kernel.target.supports_compute_capability(capability)
    ]
    incompatible = [kernel for kernel in candidates if kernel not in compatible]
    for kernel in incompatible:
        print(
            f"[skip] {kernel.label}: target {kernel.target.architecture} "
            f"is incompatible with compute capability {capability[0]}.{capability[1]}"
        )
    configs.extend(
        BenchGemmConfig(
            label=kernel.label,
            m=kernel.m,
            n=kernel.n,
            k=kernel.k,
            op=kernel.op,
            dtype=kernel.dtype,
        )
        for kernel in compatible
    )
    if args.filter:
        pattern = re.compile(args.filter)
        configs = [c for c in configs if pattern.search(c.label)]
    if args.reverse:
        configs = configs[::-1]
    print_results(run(configs))


if __name__ == "__main__":
    main()
