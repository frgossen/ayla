"""Registered variants of the configurable pipelined TMA GEMM."""

from ..gemm import auto_matmul_4096_wgmma256_wg_tma_store_module
from ..passes.pack_smem_allocs import pack_smem_allocs
from .registry import register


@register
def auto_matmul_4096_wgmma256_wg_tma_store_k3_module():
    return auto_matmul_4096_wgmma256_wg_tma_store_module()


@register
def auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_module():
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2,
    )


@register
def auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2_module():
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2,
        operand_stages=3,
        output_slices=2,
        output_store_buffers=2,
    )


@register
def auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2_module():
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=2,
        output_store_buffers=2,
        operand_move_attrs={
            "lhs": {
                "l2_cache_hint": "evict_first",
                "l2_promotion": "256B",
            },
            "dst": {"l2_cache_hint": "evict_first"},
        },
        ptx_optimization_level=1,
    )
    return pack_smem_allocs(module)
