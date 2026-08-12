"""CuTe DSL Hopper BF16 Dual GEMM candidate.

The two GEMMs use NVIDIA's Hopper persistent WGMMA kernel.  Compilation and
the intermediate/output allocations happen in prepare(), outside the timed
closure.  B1 and B2 retain the benchmark's physical [N, K] layout.
"""
from pathlib import Path
import importlib.util
import os

# Keep compiler artifacts inside the CUTLASS agent's writable tree.  These
# variables must be set before importing the DSL runtime.
_AGENT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(_AGENT_ROOT / "build-agent/cute-cache"))
os.environ.setdefault("CUDA_CACHE_PATH", str(_AGENT_ROOT / "build-agent/cuda-cache"))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


NAME = "cutlass_cute_dual_gemm"
PROBLEM = "dual_gemm"
ROOT = _AGENT_ROOT
SOURCE = (
    ROOT.parent
    / "cutlass-upstream/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm"
    / "dense_gemm_persistent.py"
)
GEMM0_TILE = tuple(int(x) for x in os.environ.get("CUTE_DUAL_TILE0", "64,128").split(","))
GEMM1_TILE = tuple(int(x) for x in os.environ.get("CUTE_DUAL_TILE1", "128,256").split(","))
USE_GRAPH = os.environ.get("CUTE_DUAL_GRAPH", "1") == "1"
METADATA = {
    "backend": "CUTLASS 4.7 CuTe DSL Hopper WGMMA",
    "implementation": "back-to-back persistent GEMM",
    "gemm0_tile": list(GEMM0_TILE),
    "gemm1_tile": list(GEMM1_TILE),
    "cluster": [1, 1],
    "cuda_graph": USE_GRAPH,
}


def _load_kernel_class():
    spec = importlib.util.spec_from_file_location("dual_hopper_persistent", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HopperWgmmaGemmPersistentKernel


def _tensor(value):
    tensor = from_dlpack(value.unsqueeze(-1), assumed_align=16)
    tensor.element_type = cutlass.BFloat16
    return tensor.mark_layout_dynamic(leading_dim=1)


def prepare(inputs):
    a, b1, b2 = inputs["a"], inputs["b1"], inputs["b2"]
    intermediate = torch.empty(
        (a.shape[0], b1.shape[0]), device=a.device, dtype=torch.bfloat16
    )
    output = torch.empty(
        (a.shape[0], b2.shape[0]), device=a.device, dtype=torch.bfloat16
    )

    m_a = _tensor(a)
    m_b1 = _tensor(b1)
    m_i = _tensor(intermediate)
    m_b2 = _tensor(b2)
    m_d = _tensor(output)

    kernel_class = _load_kernel_class()
    gemm0 = kernel_class(cutlass.Float32, GEMM0_TILE, (1, 1), 1, True)
    gemm1 = kernel_class(cutlass.Float32, GEMM1_TILE, (1, 1), 1, True)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    capture_stream = torch.cuda.Stream() if USE_GRAPH else torch.cuda.current_stream()
    stream = cuda.CUstream(capture_stream.cuda_stream)
    compiled0 = cute.compile(gemm0, m_a, m_b1, m_i, max_clusters, stream)
    compiled1 = cute.compile(gemm1, m_i, m_b2, m_d, max_clusters, stream)

    if USE_GRAPH:
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            compiled0(m_a, m_b1, m_i, stream)
            compiled1(m_i, m_b2, m_d, stream)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)

        def run():
            # Keep the capture stream alive for the lifetime of its graph.
            _ = capture_stream
            graph.replay()
            return output
    else:
        def run():
            compiled0(m_a, m_b1, m_i, stream)
            compiled1(m_i, m_b2, m_d, stream)
            return output

    return run
