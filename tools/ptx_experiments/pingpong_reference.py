"""Benchmark-harness adapter for the disposable ping-pong PTX prototype."""
import dataclasses
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as build,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from pingpong_named_barrier import transform


NAME = "Ayla asymmetric WG ping-pong"
PROBLEM = "bf16_gemm"
METADATA = {"backend": "disposable Ayla PTX ping-pong experiment"}


def prepare(inputs):
    kernel = make_ptx_kernel(build(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=2,
        output_tma_inner_padding=16,
        output_store_buffers=2,
        operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
    ))
    ptx, changed = transform(kernel.ptx)
    assert changed == 25
    launcher = make_ptx_launcher(dataclasses.replace(kernel, ptx=ptx))
    output = torch.empty((4096, 4096), device=inputs["a"].device, dtype=torch.bfloat16)

    def run():
        launcher(inputs["a"], inputs["b"], output)
        return output

    return run
