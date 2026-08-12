"""Two-warpgroup Ayla fused Dual GEMM candidate."""
from pathlib import Path
import sys

import torch

AYLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AYLA_ROOT))

from ayla.gemm_collection.dual_gemm_4096 import dual_gemm_4096_module
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


NAME = "ayla_fused_dual_gemm_2wg"
PROBLEM = "dual_gemm"
METADATA = {
    "backend": "Ayla PTX",
    "fused": True,
    "m_tile": 64,
    "n_tile_per_warpgroup": 256,
    "warp_groups": 2,
    "column_partitions": 1,
    "first_gemm_stages": 3,
    "intermediate": "BF16 swizzled shared memory",
    "output_store_buffers": 1,
}


def prepare(inputs):
    kernel = make_ptx_kernel(dual_gemm_4096_module(
        columns_per_cta=4096, warp_groups=2, output_store_buffers=1,
    ))
    launcher = make_ptx_launcher(kernel)
    output = torch.empty(
        (4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16,
    )

    def run():
        launcher(inputs["a"], inputs["b1"], inputs["b2"], output)
        return output

    return run
