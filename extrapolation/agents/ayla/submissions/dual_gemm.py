"""Ayla fused Dual GEMM submission."""
from pathlib import Path
import sys

import torch

AYLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AYLA_ROOT))

from ayla.gemm_collection.dual_gemm_4096 import dual_gemm_4096_module
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


NAME = "ayla_fused_dual_gemm_k4_second128"
PROBLEM = "dual_gemm"
METADATA = {
    "backend": "Ayla PTX",
    "fused": True,
    "m_tile": 64,
    "n_tile": 256,
    "column_partitions": 2,
    "first_gemm_stages": 4,
    "second_gemm_k_tile": 128,
    "second_gemm_stages": 1,
    "intermediate": "BF16 128B-swizzled block-major shared memory",
    "intermediate_hbm_bytes": 0,
    "output_slices": 2,
    "output_tma_inner_padding": 16,
    "output_store_buffers": 2,
}


def prepare(inputs):
    kernel = make_ptx_kernel(dual_gemm_4096_module(
        first_stages=4,
        second_k_tile=128,
        output_slices=2,
        output_padding=16,
        output_store_buffers=2,
    ))
    launcher = make_ptx_launcher(kernel)
    output = torch.empty(
        (4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16,
    )

    def run():
        launcher(inputs["a"], inputs["b1"], inputs["b2"], output)
        return output

    return run
