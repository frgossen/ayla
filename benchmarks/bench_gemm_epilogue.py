"""Benchmark a GEMM with a fused epilogue against doing it in two passes.

The question a fused epilogue answers is not "how fast is this GEMM" -- it is
"what does the epilogue cost when it happens in registers instead of as a
second pass over the output". So the comparison is against torch doing the
same work eagerly: a GEMM, then a sigmoid reading 4096x4096 back out of gmem
and writing it again.

Kept apart from bench_gemm_collection for the same reason. That harness ranks
GEMMs by TFLOPS over 2*m*n*k, and a kernel doing extra work would report as a
slower GEMM rather than as a different workload. The flops here are the GEMM's
too -- what makes the number meaningful is that every row does the same GEMM,
so the gap between them is the epilogue.

    python benchmarks/bench_gemm_epilogue.py
"""

from __future__ import annotations

import sys

import torch

import ayla.gemm_collection  # noqa: F401 — registers torch.ops.ayla.*
from bench_gemm_collection import (  # noqa: E402
    BenchGemmConfig,
    _bench,
    _populate_percentages,
    _print,
)


def _eager_gemm_sigmoid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.mm(a, b.t()))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    configs = [
        # The floor: the same ayla kernel without the epilogue. The gap to the
        # row below is what the fused sigmoid costs.
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma256 wg), no epilogue",
            m=4096, n=4096, k=4096,
            op=torch.ops.ayla.auto_matmul_4096_wgmma256_wg,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="ayla.auto_matmul (wgmma256 wg), fused sigmoid",
            m=4096, n=4096, k=4096,
            op=torch.ops.ayla.auto_matmul_4096_wgmma256_wg_sigmoid,
            dtype=torch.bfloat16,
        ),
        # torch.mm alone, for the same floor on the eager side.
        BenchGemmConfig(
            label="torch.mm, no epilogue",
            m=4096, n=4096, k=4096,
            op=torch.mm,
            dtype=torch.bfloat16,
        ),
        BenchGemmConfig(
            label="torch.mm + torch.sigmoid (two passes)",
            m=4096, n=4096, k=4096,
            op=_eager_gemm_sigmoid,
            dtype=torch.bfloat16,
        ),
    ]

    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.label}", file=sys.stderr, flush=True)
        results.append(_bench(cfg))
    _populate_percentages(configs, results)

    label_width = max(len(r.label) for r in results)
    for i, r in enumerate(results):
        _print(r, label_width, with_header=(i == 0))


if __name__ == "__main__":
    main()
