"""Benchmark every GEMM kernel in the collection on one hand-picked shape."""

from __future__ import annotations

import argparse
import re

import ayla.gemm_collection  # noqa: F401 -- registers torch.ops.ayla.*
import torch

from common import gemm_bench_config, print_results, run, torch_mm_nt


def main() -> None:
    # Every config in this file wants tf32 matmuls; it used to be a
    # per-config field that nothing ever set to False.
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = argparse.ArgumentParser(description="Benchmark GEMM kernels")
    parser.add_argument("-k", "--filter", type=str, default=None,
                        help="Regex to filter benchmarks by label")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Run the benchmarks in reverse order")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    configs = [
        gemm_bench_config(
            name="torch.mm.bf16",
            m=4096,
            n=4096,
            k=4096,
            compute=torch_mm_nt,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.pranjalssh",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla.pranjalssh_4096,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.pranjalssh_pipelined (prev best)",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla.pranjalssh_4096_pipelined,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.auto_matmul (wgmma128)",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla.auto_matmul_4096,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.auto_matmul (wgmma256 wg)",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla.auto_matmul_4096_wgmma256_wg,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.auto_matmul (wgmma256 wg, TMA store K=3)",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.auto_matmul (wgmma256 wg, TMA store K=3, 2 tiles/CTA)",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla
            .auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles,
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name=(
                "ayla.auto_matmul (wgmma256 wg, TMA store K=3, split output, "
                "2 tiles/CTA)"
            ),
            m=4096,
            n=4096,
            k=4096,
            compute=(
                torch.ops.ayla
                .auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2
            ),
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name=(
                "ayla.auto_matmul (wgmma256 wg, PTX JIT O1, packed shared "
                "allocations, A promote 256B, split output, 4 tiles/CTA, "
                "A+C evict-first)"
            ),
            m=4096,
            n=4096,
            k=4096,
            compute=(
                torch.ops.ayla
                .auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2
            ),
            dtype=torch.bfloat16,
        ),
        gemm_bench_config(
            name="ayla.fastcu_5_1 (fast.cu static smem)",
            m=4096,
            n=4096,
            k=4096,
            compute=torch.ops.ayla.fastcu_5_1,
            dtype=torch.bfloat16,
        ),
    ]
    if args.filter:
        pattern = re.compile(args.filter)
        configs = [c for c in configs if pattern.search(c.name)]
    if args.reverse:
        configs = configs[::-1]
    print_results(run(configs))


if __name__ == "__main__":
    main()
