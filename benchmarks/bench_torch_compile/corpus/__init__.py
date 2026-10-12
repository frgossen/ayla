"""What the benchmark runs: one module per graph worth measuring.

Each module builds its own configs; this collects them, so adding a graph is
adding a file and one line here.
"""
from __future__ import annotations

from benchmarks.common import BenchConfig

from . import gemm_sigmoid, muse_glimmer_block


def build_configs(*, muse_num_tokens: int | None = None) -> list[BenchConfig]:
    """The supported GEMM cases, or one captured block at its input shape."""
    if muse_num_tokens is not None:
        return [muse_glimmer_block.build_config(muse_num_tokens)]
    return gemm_sigmoid.build_configs()
