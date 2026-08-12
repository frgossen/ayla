"""Pinned Hopper TMA-persistent Triton BF16 GEMM.

Adapted from Triton's official ``09-persistent-matmul.py`` tutorial. B is
physically [N,K], so the kernel computes A @ B.T without a material transpose.
"""
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

NAME = "triton_tma_persistent_bf16"
PROBLEM = "bf16_gemm"

BM = int(os.environ.get("TRITON_BM", "128"))
BN = int(os.environ.get("TRITON_BN", "256"))
BK = int(os.environ.get("TRITON_BK", "64"))
STAGES = int(os.environ.get("TRITON_STAGES", "3"))
WARPS = int(os.environ.get("TRITON_WARPS", "8"))
GROUP_M = int(os.environ.get("TRITON_GROUP_M", "8"))
METADATA = {
    "backend": "Triton TMA persistent",
    "block_m": BM,
    "block_n": BN,
    "block_k": BK,
    "num_stages": STAGES,
    "num_warps": WARPS,
    "group_m": GROUP_M,
}


@triton.jit
def _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _kernel(a_desc, b_desc, c_desc, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            NUM_SMS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = a_desc.load([offset_m, k])
            b = b_desc.load([offset_n, k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        out_m, out_n = _pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        c_desc.store([out_m * BLOCK_M, out_n * BLOCK_N], accumulator.to(tl.bfloat16))


def prepare(inputs):
    a, b = inputs["a"], inputs["b"]
    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BN, BK])
    c_desc = TensorDescriptor.from_tensor(output, [BM, BN])
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(sms, triton.cdiv(a.shape[0], BM) * triton.cdiv(b.shape[0], BN)),)

    def run():
        _kernel[grid](
            a_desc, b_desc, c_desc,
            M=a.shape[0], N=b.shape[0], K=a.shape[1], NUM_SMS=sms,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=GROUP_M,
            num_stages=STAGES, num_warps=WARPS,
        )
        return output

    return run
