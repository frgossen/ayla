"""A GEMM with a sigmoid on its output -- the epilogue-fusion case."""
from __future__ import annotations

import torch

from benchmarks.common import BenchConfig


def build_gemm_sigmoid_config(
    *, m: int, k: int, n: int, dtype: torch.dtype,
) -> BenchConfig:
    """A GEMM with a sigmoid on its output -- the epilogue-fusion case.

    The weight is [N, K], the layout ayla's ops take and the one a weight is
    normally stored in, so the transpose costs nothing.
    """
    def make_inputs() -> list[torch.Tensor]:
        torch.manual_seed(0)
        return [
            torch.randn(m, k, dtype=dtype, device="cuda"),
            torch.randn(n, k, dtype=dtype, device="cuda"),
        ]

    def compute(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(a @ bt.t())

    return BenchConfig(
        name=f"gemm_sigmoid {m}x{k}x{n}x{dtype}".replace("torch.", ""),
        make_inputs=make_inputs,
        compute=compute,
    )


def build_configs() -> list[BenchConfig]:
    """What this module contributes to the corpus."""
    return [
        # 4096 cubed bf16, a shape the collection covers, so ayla has
        # something to claim and the bottom row means something.
        build_gemm_sigmoid_config(m=4096, k=4096, n=4096, dtype=torch.bfloat16),
    ]
