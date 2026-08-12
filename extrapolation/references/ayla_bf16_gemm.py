"""Ayla's retained 4-tile Hopper BF16 GEMM baseline."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as build,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

NAME = "ayla_k3_4tiles_split2_dbuf2"
PROBLEM = "bf16_gemm"
METADATA = {
    "backend": "Ayla PTX",
    "output_tiles_per_cta": 4,
    "operand_stages": 3,
    "output_slices": 2,
    "output_tma_inner_padding": 16,
    "output_store_buffers": 2,
}


def prepare(inputs):
    kernel = make_ptx_kernel(build(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=2,
        output_tma_inner_padding=16,
        output_store_buffers=2,
        operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
    ))
    launcher = make_ptx_launcher(kernel)
    output = torch.empty((4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16)

    def run():
        launcher(inputs["a"], inputs["b"], output)
        return output

    return run
