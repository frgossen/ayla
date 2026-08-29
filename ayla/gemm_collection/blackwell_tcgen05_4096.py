"""Native Blackwell tcgen05 4096x4096 BF16 GEMM."""

from ..gemm import blackwell_tcgen05_gemm_module
from .registry import register


@register
def blackwell_tcgen05_4096_module():
    return blackwell_tcgen05_gemm_module(
        operand_stages=8, output_n_tile=232, transpose_cta_grid=True,
        balanced_n_tile=224,
        lhs_l2_cache_hint="evict_first", rhs_l2_cache_hint="evict_last",
        lhs_l2_promotion="256B", rhs_l2_promotion="256B",
        split_tma_barriers=True, split_rhs_tma=True,
        prefetch_lhs_next=True, prefetch_rhs_next=True,
        output_l1_cache_policy="no_allocate",
        output_l2_eviction_priority="evict_first",
    )
