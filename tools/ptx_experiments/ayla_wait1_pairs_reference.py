"""Benchmark adapter for the artifact-level paired-WGMMA Ayla experiment."""
import dataclasses
from pathlib import Path

import torch

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as build,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


NAME = "ayla_wait1_paired_stages"
PROBLEM = "bf16_gemm"
METADATA = {"backend": "Ayla PTX; paired wait_group 1 with delayed SMEM release"}
PTX = Path(__file__).parents[2] / "generated_ptx/compiler_comparison/ayla_wait1_pairs.ptx"


def prepare(inputs):
    kernel = make_ptx_kernel(build(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=2,
        output_tma_inner_padding=16,
        output_store_buffers=2,
        operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
    ))
    launcher = make_ptx_launcher(dataclasses.replace(kernel, ptx=PTX.read_text()))
    output = torch.empty((4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16)

    def run():
        launcher(inputs["a"], inputs["b"], output)
        return output

    return run
