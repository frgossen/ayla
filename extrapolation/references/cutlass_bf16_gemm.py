"""CUTLASS 4.7 CuTe DSL Hopper persistent BF16 GEMM wrapper.

The kernel class is NVIDIA's official Hopper persistent dense GEMM example;
this file only supplies existing PyTorch tensors through DLPack and pins its
tile/scheduler configuration for the shared harness.
"""
from pathlib import Path
import importlib.util
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

NAME = "cutlass_cute_persistent_bf16"
PROBLEM = "bf16_gemm"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "agents/cutlass-upstream/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py"


def _configuration():
    return {
        "backend": "CUTLASS 4.7 CuTe DSL persistent GEMM",
        "tile_m": 128,
        "tile_n": int(os.environ.get("CUTLASS_TILE_N", "256")),
        "cluster": [int(x) for x in os.environ.get("CUTLASS_CLUSTER", "1,1").split(",")],
        "swizzle": int(os.environ.get("CUTLASS_SWIZZLE", "1")),
        "raster": os.environ.get("CUTLASS_RASTER", "m"),
    }


METADATA = _configuration()


def _load_kernel_class():
    spec = importlib.util.spec_from_file_location("cutlass_hopper_persistent", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HopperWgmmaGemmPersistentKernel


def _tensor(torch_tensor, leading_dim):
    value = from_dlpack(torch_tensor.unsqueeze(-1), assumed_align=16)
    value.element_type = cutlass.BFloat16
    return value.mark_layout_dynamic(leading_dim=leading_dim)


def prepare(inputs):
    a, b = inputs["a"], inputs["b"]
    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    m_a = _tensor(a, 1)       # MxKx1, K-major
    m_b = _tensor(b, 1)       # NxKx1, K-major
    m_c = _tensor(output, 1)  # MxNx1, N-major

    tile = (METADATA["tile_m"], METADATA["tile_n"])
    cluster = tuple(METADATA["cluster"])
    swizzle = METADATA["swizzle"]
    raster_along_m = METADATA["raster"] == "m"
    kernel_class = _load_kernel_class()
    gemm = kernel_class(cutlass.Float32, tile, cluster, swizzle, raster_along_m)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, m_a, m_b, m_c, max_clusters, stream)

    def run():
        # max_clusters is a cutlass.Constexpr specialized by cute.compile and
        # therefore is not a runtime argument of the compiled callable.
        compiled(m_a, m_b, m_c, stream)
        return output

    return run
