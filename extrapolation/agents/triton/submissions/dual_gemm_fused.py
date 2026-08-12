"""Single-program fused BF16 dual GEMM for the canonical H100 problem.

Each program owns one row tile and one partition of the output columns.  It
computes A @ B1.T once for that partition, rounds the intermediate to BF16
(matching torch.mm's output contract), and keeps the tile on chip while it
walks its output-column tiles of B2.  The selected two-way partition repeats
the first stage to fill the H100's SMs; this is faster than a 64-CTA grid.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


NAME = "triton_fused_dual_gemm"
PROBLEM = "dual_gemm"

BLOCK_M = int(os.environ.get("DUAL_BLOCK_M", "64"))
BLOCK_N = int(os.environ.get("DUAL_BLOCK_N", "128"))
BLOCK_K = int(os.environ.get("DUAL_BLOCK_K", "128"))
NUM_WARPS = int(os.environ.get("DUAL_NUM_WARPS", "4"))
NUM_STAGES = int(os.environ.get("DUAL_NUM_STAGES", "4"))
SPLIT_N = int(os.environ.get("DUAL_SPLIT_N", "2"))

METADATA = {
    "backend": "Triton TMA + WGMMA fused dual GEMM",
    "block_m": BLOCK_M,
    "block_n": BLOCK_N,
    "block_k": BLOCK_K,
    "num_warps": NUM_WARPS,
    "num_stages": NUM_STAGES,
    "split_n": SPLIT_N,
    "intermediate": "on-chip BF16",
    "grid_strategy": "two programs per M tile, partitioned N tiles",
    "actual_flop_multiplier": (SPLIT_N + 1) / 2,
}


@triton.jit
def _dual_gemm_kernel(
    a_desc,
    b1_desc,
    b2_desc,
    out_desc,
    M: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    K1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // SPLIT) * BM
    split = pid % SPLIT

    first = tl.zeros((BM, N1), dtype=tl.float32)
    for k in range(0, K1, BK):
        a = a_desc.load([row, k])
        b1 = b1_desc.load([0, k])
        first = tl.dot(a, b1.T, first)

    # This conversion is semantically required: torch.mm materializes a BF16
    # intermediate before the second GEMM.
    intermediate = first.to(tl.bfloat16)

    for n in range(split * BN, N2, SPLIT * BN):
        b2 = b2_desc.load([n, 0])
        second = tl.dot(intermediate, b2.T)
        out_desc.store([row, n], second.to(tl.bfloat16))


def prepare(inputs):
    a, b1, b2 = inputs["a"], inputs["b1"], inputs["b2"]
    m, k1 = a.shape
    n1 = b1.shape[0]
    n2 = b2.shape[0]
    if (m, k1, n1, n2) != (4096, 4096, 128, 4096):
        raise ValueError("this candidate is specialized for the canonical dual_gemm shape")

    output = torch.empty((m, n2), device=a.device, dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b1_desc = TensorDescriptor.from_tensor(b1, [n1, BLOCK_K])
    b2_desc = TensorDescriptor.from_tensor(b2, [BLOCK_N, n1])
    out_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])
    grid = (triton.cdiv(m, BLOCK_M) * SPLIT_N,)

    def launch():
        _dual_gemm_kernel[grid](
            a_desc,
            b1_desc,
            b2_desc,
            out_desc,
            M=m,
            N1=n1,
            N2=n2,
            K1=k1,
            BM=BLOCK_M,
            BN=BLOCK_N,
            BK=BLOCK_K,
            SPLIT=SPLIT_N,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    # Compile and allocate all descriptor state before benchmark timing.
    launch()
    torch.cuda.synchronize(a.device)

    def run():
        launch()
        return output

    return run
