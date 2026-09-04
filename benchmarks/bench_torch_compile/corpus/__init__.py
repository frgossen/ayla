"""What the benchmark runs: one module per graph worth measuring.

Each module builds its own configs and lists them; this collects them, so
adding a graph is adding a file and one line here.
"""
from __future__ import annotations

from corpus import gemm_sigmoid

CONFIGS = [
    *gemm_sigmoid.CONFIGS,
]
