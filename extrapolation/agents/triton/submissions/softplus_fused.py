"""Persistent Hopper GEMM with a fused stable Softplus epilogue."""

import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


NAME = "triton_tma_persistent_softplus"
PROBLEM = "softplus"

# Best configuration from the validated 4K Hopper GEMM sweep.  Keeping the
# accumulator tile at 128x128 also gives enough independent Softplus work to
# cover much of the transcendental latency before the descriptor store.
BM = int(os.environ.get("SOFTPLUS_BM", "128"))
BN = int(os.environ.get("SOFTPLUS_BN", "128"))
BK = int(os.environ.get("SOFTPLUS_BK", "64"))
STAGES = int(os.environ.get("SOFTPLUS_STAGES", "6"))
WARPS = int(os.environ.get("SOFTPLUS_WARPS", "8"))
GROUP_M = int(os.environ.get("SOFTPLUS_GROUP_M", "8"))
SYMMETRIC = int(os.environ.get("SOFTPLUS_SYMMETRIC", "1"))

METADATA = {
    "backend": "Triton TMA persistent, fused stable Softplus",
    "block_m": BM,
    "block_n": BN,
    "block_k": BK,
    "num_stages": STAGES,
    "num_warps": WARPS,
    "group_m": GROUP_M,
    "tau_specialized": True,
    "epilogue": (
        "tau*(max(z,0)+log(1+exp(-abs(z))))"
        if SYMMETRIC
        else "tau*log(1+exp(min(z,20))); select x above threshold"
    ),
}


@triton.jit
def _tile_coords(tile_id, num_pid_in_group, num_pid_m,
                 GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + tile_id % group_size_m
    pid_n = tile_id % num_pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _softplus_kernel(
    a_desc,
    b_desc,
    c_desc,
    TAU: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # The delayed destination coordinates let descriptor stores pipeline with
    # the following persistent iteration, as in Triton's Hopper tutorial.
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _tile_coords(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, K, BLOCK_K):
            a = a_desc.load([offset_m, k])
            b = b_desc.load([offset_n, k])
            acc = tl.dot(a, b.T, acc)

        # Both variants bound the exponential operand.  The selected symmetric
        # form is non-positive everywhere; the retained threshold form clamps
        # before tl.where because tl.where does not short-circuit evaluation.
        scaled = acc / TAU
        if SYMMETRIC:
            # This form never evaluates a positive exponential.  Above the
            # threshold exp(-scaled) is too small to change FP32 1.0, so the
            # expression rounds to acc exactly, matching the threshold branch.
            nonlinear = TAU * (
                tl.maximum(scaled, 0.0)
                + tl.log(1.0 + tl.exp(-tl.abs(scaled)))
            )
            result = nonlinear
        else:
            exp_arg = tl.minimum(scaled, 20.0)
            nonlinear = TAU * tl.log(1.0 + tl.exp(exp_arg))
            result = tl.where(scaled > 20.0, acc, nonlinear)

        tile_id_c += NUM_SMS
        out_m, out_n = _tile_coords(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        c_desc.store(
            [out_m * BLOCK_M, out_n * BLOCK_N], result.to(tl.bfloat16)
        )


def prepare(inputs):
    a, b, tau = inputs["a"], inputs["b"], inputs["tau"]
    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BN, BK])
    c_desc = TensorDescriptor.from_tensor(output, [BM, BN])
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(sms, triton.cdiv(a.shape[0], BM) * triton.cdiv(b.shape[0], BN)),)

    def run():
        _softplus_kernel[grid](
            a_desc,
            b_desc,
            c_desc,
            TAU=tau,
            M=a.shape[0],
            N=b.shape[0],
            K=a.shape[1],
            NUM_SMS=sms,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_SIZE_M=GROUP_M,
            SYMMETRIC=SYMMETRIC,
            num_stages=STAGES,
            num_warps=WARPS,
        )
        return output

    # Compile in prepare(), outside the closure measured by the harness.
    run()
    torch.cuda.synchronize(a.device)
    return run
