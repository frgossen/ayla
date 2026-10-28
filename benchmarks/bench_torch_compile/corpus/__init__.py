"""What the benchmark runs: one module per graph worth measuring.

Each module builds its own configs; this collects them, so adding a graph is
adding a file and one line here.
"""
from __future__ import annotations

from benchmarks.common import BenchConfig

from . import gemm_sigmoid, muse_swiglu


def build_configs() -> list[BenchConfig]:
    """Every config in the corpus."""
    return [
        *gemm_sigmoid.build_configs(),
        *muse_swiglu.build_configs(),
    ]
