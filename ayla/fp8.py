"""Hopper FP8 tensor-scaled GEMM with FP32 accumulation.

Inputs use E4M3FN or E5M2, including mixed pairs. Dequantization scales are
scalar FP32 CUDA tensors. The output is BF16 by default; FP16/FP32 are optional.
"""

from __future__ import annotations

from functools import cache

import torch

from .fp8_schedule_table import SCHEDULES
from .gemm_collection.gemm import make_fp8_gemm_kernel
from .ptx_runtime import make_ptx_launcher
from .torch_custom_call import OperandKind, make_torch_custom_call

_DTYPES = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
_OUTPUTS = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


@cache
def _cached_op(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    out_dtype: torch.dtype,
):
    config = (
        SCHEDULES.get((m, n, k), {})
        if dtype == rhs_dtype == torch.float8_e4m3fn and out_dtype == torch.bfloat16
        else {}
    )
    config = {"operand_k_tile": 128 if k % 128 == 0 else 64, **config}
    if out_dtype == torch.float32:
        # FP32 shared writeback is twice the BF16 size. Keep the default
        # operand ring and output buffer within Hopper's shared-memory limit.
        config.update(output_n_tile=128, warpgroup_n_tile=128, operand_stages=3)
    kernel = make_fp8_gemm_kernel(
        m=m,
        n=n,
        k=k,
        dtype=_DTYPES[dtype],
        rhs_dtype=_DTYPES[rhs_dtype],
        output_dtype=_OUTPUTS[out_dtype],
        scaled=True,
        **config,
    )
    launcher = make_ptx_launcher(kernel)
    return make_torch_custom_call(
        launcher,
        [
            OperandKind.R,
            OperandKind.R,
            OperandKind.RESULT,
            OperandKind.R,
            OperandKind.R,
        ],
        name=f"fp8_scaled_mm_{m}_{n}_{k}_{_DTYPES[dtype]}_{_DTYPES[rhs_dtype]}_{_OUTPUTS[out_dtype]}",
    )


@torch._dynamo.assume_constant_result
def _op_name(m, n, k, dtype, rhs_dtype, out_dtype):
    # Dynamo resolves shape specialization once and captures only the custom op.
    return _cached_op(m, n, k, dtype, rhs_dtype, out_dtype)._schema.name.split("::")[1]


def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return ``(a.float() @ b.float()) * scale_a * scale_b`` on Hopper.

    A[M,K] must be row major and B[K,N] column major (``weight.T``).
    K must be a multiple of 64 and N a multiple of 8; M/N may have partial tiles. Quantization and
    scale computation belong to the caller. Scales are dequantization factors,
    matching ``torch._scaled_mm``. FP8 outputs/block scales are unsupported.
    The custom op can be captured in CUDA graphs or ``torch.compile``.
    """
    if a.dtype not in _DTYPES or b.dtype not in _DTYPES:
        raise ValueError("inputs must be float8_e4m3fn or float8_e5m2")
    if out_dtype not in _OUTPUTS:
        raise ValueError("output dtype must be bfloat16, float16, or float32")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if not a.is_contiguous() or not b.T.is_contiguous():
        raise ValueError("expected row-major A and column-major B (weight.T)")
    if torch.compiler.is_compiling():
        # Kernels specialize their tile grid and tensor maps to concrete sizes.
        # Prevent automatic dynamic-shape promotion on later compilations.
        torch._dynamo.mark_static(a)
        torch._dynamo.mark_static(b)
    m, k = a.shape
    n = b.shape[1]
    if min(m, n, k) <= 0 or k % 64 or n % 8:
        raise ValueError(
            "dimensions must be positive, K a multiple of 64, and N a multiple of 8"
        )
    for tensor in (a, b, scale_a, scale_b):
        if tensor.device != a.device or not tensor.is_cuda:
            raise ValueError("inputs and scales must share one CUDA device")
    for scale in (scale_a, scale_b):
        if scale.dtype != torch.float32 or scale.ndim != 0:
            raise ValueError("dequantization scales must be scalar FP32 tensors")
    if torch.cuda.get_device_capability(a.device)[0] != 9:
        raise ValueError("FP8 WGMMA requires a Hopper (SM90) GPU")
    name = _op_name(m, n, k, a.dtype, b.dtype, out_dtype)
    return getattr(torch.ops.ayla, name).default(a, b.T, scale_a, scale_b)
