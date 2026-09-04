"""Compare eager, ``torch.compile``, and ``torch.compile`` over ayla.

The row that matters is `ayla_compiled` against `compiled`: the same graph
lowered the same way everywhere except the matmul. What gets measured lives
in `corpus/`, one module per graph.

    python benchmarks/bench_torch_compile/bench_torch_compile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

# `benchmarks/` holds `common`, and running a script only puts that script's
# own directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ayla import torch_compile  # noqa: E402
from common import BenchConfig, print_results, run_bench  # noqa: E402
# TODO: Imports every corpus module, each building its configs at import
# time. Select and load on demand once that gets slow.
from corpus import CONFIGS  # noqa: E402


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
