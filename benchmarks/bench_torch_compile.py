"""Compare eager, ``torch.compile``, and ``torch.compile`` over ayla.

The row that matters is `ayla_compiled` against `compiled`: the same graph
lowered the same way everywhere except the matmul.

    python benchmarks/bench_torch_compile.py
"""
from __future__ import annotations

import torch
import torch._inductor.config as inductor_config

from ayla import torch_compile
from common import BenchConfig, print_results, run_bench


def build_gemm_sigmoid_benchmark(
    *, m: int, k: int, n: int, dtype: torch.dtype,
) -> BenchConfig:
    """A GEMM with a sigmoid on its output -- the epilogue-fusion case."""
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


CONFIGS = [
    build_gemm_sigmoid_benchmark(m=4096, k=4096, n=4096, dtype=torch.bfloat16),
]


def _compile(fn, inputs, *, use_ayla: bool):
    """Compile inside the install window -- the first call is what compiles."""
    compiled = torch.compile(fn, dynamic=False)
    if use_ayla:
        torch_compile.install()
    try:
        compiled(*inputs)
    finally:
        torch_compile.uninstall()
    return compiled


def variants(config: BenchConfig) -> list[BenchConfig]:
    """Expand a config into its eager, compiled and ayla_compiled rows."""
    inputs = config.make_inputs()
    # Two lambdas, not one: dynamo caches on the code object.
    compiled = _compile(
        lambda *args: config.compute(*args), inputs, use_ayla=False)
    ayla = _compile(
        lambda *args: config.compute(*args), inputs, use_ayla=True)

    eager = config.compute
    return [
        BenchConfig(name=f"{config.name} eager",
                    make_inputs=config.make_inputs, compute=eager),
        BenchConfig(name=f"{config.name} compiled",
                    make_inputs=config.make_inputs, compute=compiled),
        BenchConfig(name=f"{config.name} ayla_compiled",
                    make_inputs=config.make_inputs, compute=ayla),
    ]


def main() -> None:
    # Both span the run: the backend is outside the cache key, and dynamo
    # guards on grad mode.
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        expanded = [
            variant for config in CONFIGS for variant in variants(config)
        ]
        print_results(run_bench(expanded))


if __name__ == "__main__":
    main()
