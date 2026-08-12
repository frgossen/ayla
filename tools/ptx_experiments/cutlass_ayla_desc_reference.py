"""CuTe persistent GEMM using Ayla-style WGMMA descriptor adds."""
from pathlib import Path

from cutlass_patched_cubin import prepare_patched


NAME = "cutlass_cute_ayla_descriptors"
PROBLEM = "bf16_gemm"
METADATA = {
    "backend": "CuTe host ABI, Ayla-style WGMMA descriptor construction",
    "sass_instructions": 992,
    "registers_per_thread": 168,
}
CUBIN = Path(__file__).parents[2] / "generated_ptx/compiler_comparison/cute_ayla_descriptors.cubin"


def prepare(inputs):
    return prepare_patched(inputs, CUBIN, "cute_ayla_desc")
