"""GEMMs with elementwise output epilogues."""
from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarks.common import BenchConfig


def _build_gemm_epilogue_config(
    *, name: str, m: int, k: int, n: int, dtype: torch.dtype,
    epilogue: Callable[[torch.Tensor], torch.Tensor] | None,
) -> BenchConfig:
    def make_inputs() -> list[torch.Tensor]:
        torch.manual_seed(0)
        return [
            torch.randn(m, k, dtype=dtype, device="cuda"),
            torch.randn(n, k, dtype=dtype, device="cuda"),
        ]

    def compute(a: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
        result = a @ bt.t()
        return result if epilogue is None else epilogue(result)

    return BenchConfig(
        name=f"gemm_{name} {m}x{k}x{n}x{dtype}".replace("torch.", ""),
        make_inputs=make_inputs,
        compute=compute,
    )


def build_gemm_config(
    *, m: int, k: int, n: int, dtype: torch.dtype,
) -> BenchConfig:
    """A GEMM without an epilogue, exposing each implementation's floor."""
    return _build_gemm_epilogue_config(
        name="only", m=m, k=k, n=n, dtype=dtype, epilogue=None,
    )


def build_gemm_sigmoid_config(
    *, m: int, k: int, n: int, dtype: torch.dtype,
) -> BenchConfig:
    """A GEMM followed by sigmoid."""
    return _build_gemm_epilogue_config(
        name="sigmoid", m=m, k=k, n=n, dtype=dtype,
        epilogue=torch.sigmoid,
    )


def build_gemm_silu_config(
    *, m: int, k: int, n: int, dtype: torch.dtype,
) -> BenchConfig:
    """A GEMM followed by SiLU, kept explicit as sigmoid plus multiply."""
    def silu(value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)

    return _build_gemm_epilogue_config(
        name="silu", m=m, k=k, n=n, dtype=dtype, epilogue=silu,
    )


def build_configs() -> list[BenchConfig]:
    """What this module contributes to the corpus."""
    return [
        # The common floor makes the incremental cost of each fused epilogue
        # visible alongside the end-to-end comparison with torch.compile.
        build_gemm_config(m=4096, k=4096, n=4096, dtype=torch.bfloat16),
        build_gemm_sigmoid_config(m=4096, k=4096, n=4096, dtype=torch.bfloat16),
        build_gemm_silu_config(m=4096, k=4096, n=4096, dtype=torch.bfloat16),
    ]
