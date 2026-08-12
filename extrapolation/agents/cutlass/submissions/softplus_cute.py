"""Fused Hopper BF16 GEMM + stable Softplus using CUTLASS CuTe DSL."""

from pathlib import Path
import importlib.util
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


NAME = "cutlass_cute_fused_softplus"
PROBLEM = "softplus"

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "agents/cutlass-upstream/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm"
    / "dense_gemm_fp8_gelu_persistent.py"
)


def _configuration():
    return {
        "backend": "CUTLASS 4.7 CuTe DSL persistent WGMMA + fused Softplus",
        "tile_m": int(os.environ.get("CUTLASS_TILE_M", "128")),
        "tile_n": int(os.environ.get("CUTLASS_TILE_N", "256")),
        "cluster": [
            int(x) for x in os.environ.get("CUTLASS_CLUSTER", "1,1").split(",")
        ],
        "swizzle": int(os.environ.get("CUTLASS_SWIZZLE", "1")),
        "raster": os.environ.get("CUTLASS_RASTER", "m"),
        "math": os.environ.get("CUTLASS_SOFTPLUS_MATH", "approx"),
        "threshold": 20.0,
    }


METADATA = _configuration()


def stable_softplus_approx(acc_vec):
    """Stable tau=1 Softplus with Hopper approximate EX2/LG2 instructions.

    The exponential input is capped at the PyTorch threshold before it is
    evaluated. The final select preserves PyTorch's linear branch for x > 20.
    """

    threshold = cute.full_like(acc_vec, 20.0)
    exp_arg = cute.where(acc_vec > threshold, threshold, acc_vec)
    log2_e = cute.full_like(acc_vec, 1.4426950408889634)
    ln_2 = cute.full_like(acc_vec, 0.6931471805599453)
    one = cute.full_like(acc_vec, 1.0)
    smooth = cute.log2(
        one + cute.exp2(exp_arg * log2_e, fastmath=True), fastmath=True
    ) * ln_2
    return cute.where(acc_vec > threshold, acc_vec, smooth)


def stable_softplus_precise(acc_vec):
    """Stable tau=1 Softplus using the precise exp/log1p lowering."""

    threshold = cute.full_like(acc_vec, 20.0)
    exp_arg = cute.where(acc_vec > threshold, threshold, acc_vec)
    smooth = cute.log1p(cute.exp(exp_arg))
    return cute.where(acc_vec > threshold, acc_vec, smooth)


def _load_kernel_class():
    spec = importlib.util.spec_from_file_location("cutlass_hopper_fused_epilogue", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HopperFP8GeluPersistentGemmKernel


def _tensor(torch_tensor, leading_dim):
    value = from_dlpack(torch_tensor.unsqueeze(-1), assumed_align=16)
    value.element_type = cutlass.BFloat16
    return value.mark_layout_dynamic(leading_dim=leading_dim)


def prepare(inputs):
    a, b = inputs["a"], inputs["b"]
    tau = float(inputs["tau"])
    if tau != 1.0:
        raise ValueError("the canonical Softplus problem requires tau=1.0")

    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    m_a = _tensor(a, 1)       # MxKx1, K-major
    m_b = _tensor(b, 1)       # NxKx1, K-major; computes A @ B.T
    m_c = _tensor(output, 1)  # MxNx1, N-major

    tile = (METADATA["tile_m"], METADATA["tile_n"])
    cluster = tuple(METADATA["cluster"])
    raster_along_m = METADATA["raster"] == "m"
    epilogue_op = (
        stable_softplus_approx
        if METADATA["math"] == "approx"
        else stable_softplus_precise
    )

    kernel_class = _load_kernel_class()
    gemm = kernel_class(
        cutlass.Float32,
        tile,
        cluster,
        METADATA["swizzle"],
        raster_along_m,
    )
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        cluster[0] * cluster[1]
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compilation and specialization of the elementwise epilogue happen here,
    # outside the timed run closure.
    compiled = cute.compile(
        gemm,
        m_a,
        m_b,
        m_c,
        1.0,   # neutral scale_ab runtime argument
        False, # compile away the scale operation
        max_clusters,
        stream,
        epilogue_op,
    )

    def run():
        compiled(m_a, m_b, m_c, 1.0, stream)
        return output

    return run
