"""Regression tests for the retained Muse benchmark schedules."""

from benchmarks.benchmark_bf16_muse_swiglu import _LARGE_M_CONFIGS


def test_retains_large_muse_schedules():
    projection_512 = _LARGE_M_CONFIGS[(512, "projection")]
    down_512 = _LARGE_M_CONFIGS[(512, "down")]
    down_2048 = _LARGE_M_CONFIGS[(2048, "down")]

    assert tuple(projection_512[key] for key in (
        "output_m_tile", "output_n_tile", "operand_k_tile",
        "operand_stages",
    )) == (128, 208, 64, 4)
    assert tuple(down_512[key] for key in (
        "output_m_tile", "output_n_tile", "operand_k_tile",
        "operand_stages",
    )) == (128, 208, 64, 4)
    assert tuple(down_2048[key] for key in (
        "output_m_tile", "output_n_tile", "operand_k_tile",
        "operand_stages", "output_slices", "output_tma_inner_size",
    )) == (192, 192, 64, 3, 3, 64)
