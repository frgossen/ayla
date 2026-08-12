"""Shared preparation helper for Dual GEMM tuning adapters."""
from pathlib import Path
import sys

import torch

AYLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AYLA_ROOT))

from ayla.gemm_collection.dual_gemm_4096 import dual_gemm_4096_module
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


def prepare_config(inputs, **config):
    launcher = make_ptx_launcher(make_ptx_kernel(dual_gemm_4096_module(**config)))
    output = torch.empty(
        (4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16,
    )

    def run():
        launcher(inputs["a"], inputs["b1"], inputs["b2"], output)
        return output

    return run
