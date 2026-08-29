"""Native Blackwell tcgen05 4096x4096 BF16 GEMM."""

from ..gemm import blackwell_tcgen05_gemm_module
from .registry import register


@register
def blackwell_tcgen05_4096_module():
    return blackwell_tcgen05_gemm_module(
        operand_stages=8, transpose_cta_grid=True,
    )
