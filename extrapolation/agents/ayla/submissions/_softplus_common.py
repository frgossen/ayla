"""Shared candidate preparation for Softplus schedule tuning."""
from pathlib import Path
import sys

import torch

AYLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AYLA_ROOT))

from ayla.gemm_collection.softplus_4096 import softplus_4096_kernel
from ayla.ptx_runtime import make_ptx_launcher


def prepare_config(inputs, *, interleave: int, **kernel_config):
    tau = float(inputs["tau"])
    launcher = make_ptx_launcher(
        softplus_4096_kernel(
            tau=tau, interleave=interleave, **kernel_config,
        )
    )
    output = torch.empty(
        (4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16,
    )

    def run():
        launcher(inputs["a"], inputs["b"], output)
        return output

    return run
