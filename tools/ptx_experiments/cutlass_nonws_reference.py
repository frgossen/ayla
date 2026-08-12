"""Experimental non-warp-specialized CuTe DSL BF16 GEMM adapter."""
from pathlib import Path
import importlib.util

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


NAME = "cutlass_cute_nonws_bf16"
PROBLEM = "bf16_gemm"
SOURCE = Path(
    "/home/arseniivanov/extrapolation/agents/cutlass/examples/python/"
    "CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py"
)
METADATA = {
    "backend": "CUTLASS 4.7 CuTe DSL non-warp-specialized GEMM",
    "tile_m": 128,
    "tile_n": 256,
    "cluster": [1, 1],
}


def _load_kernel_class():
    spec = importlib.util.spec_from_file_location("cutlass_hopper_nonws", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HopperWgmmaGemmKernel


def _tensor(torch_tensor):
    value = from_dlpack(torch_tensor.unsqueeze(-1), assumed_align=16)
    value.element_type = cutlass.BFloat16
    return value.mark_layout_dynamic(leading_dim=1)


def prepare(inputs):
    a, b = inputs["a"], inputs["b"]
    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    m_a, m_b, m_c = _tensor(a), _tensor(b), _tensor(output)
    gemm = _load_kernel_class()(cutlass.Float32, (128, 256), (1, 1))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, m_a, m_b, m_c, stream)

    def run():
        compiled(m_a, m_b, m_c, stream)
        return output

    return run
