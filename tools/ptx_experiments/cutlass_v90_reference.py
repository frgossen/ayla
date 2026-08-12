"""CuTe persistent GEMM reassembled unchanged by CUDA 13.0 PTXAS."""
from pathlib import Path

from cutlass_patched_cubin import prepare_patched


NAME = "cutlass_cute_v90_reassembled"
PROBLEM = "bf16_gemm"
METADATA = {"backend": "CuTe host ABI, original PTX reassembled by PTXAS 13.0"}
CUBIN = Path(__file__).parents[2] / "generated_ptx/compiler_comparison/cute_v90_baseline.cubin"


def prepare(inputs):
    return prepare_patched(inputs, CUBIN, "cute_v90")
