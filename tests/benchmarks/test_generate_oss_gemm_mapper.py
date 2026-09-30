import dataclasses

from benchmarks.generate_oss_gemm_mapper import (
    Point,
    _adapt_query_config,
    build_mapper,
    query_mapper,
)
from benchmarks.oss_gemm_corpus import fallback_config


def _points():
    narrow = dataclasses.replace(
        fallback_config(), name="narrow", output_m_tile=64,
    )
    wide = dataclasses.replace(
        narrow, name="wide", output_n_tile=128,
        warpgroup_n_tile=128, output_store="stmatrix",
        output_tma_inner_size=128,
    )
    return [
        Point(
            shape_id=f"shape-{index}", m=m, n=n, k=512,
            dtype="float16", frequency=1, percentage=105.0,
            config=narrow if n <= 256 else wide,
        )
        for index, (m, n) in enumerate([
            (128, 128), (256, 128), (512, 256), (1024, 256),
            (128, 512), (256, 512), (512, 1024), (1024, 1024),
        ])
    ]


def test_mapper_is_small_and_query_explains_decisions():
    mapper = build_mapper(
        _points(), max_leaves=3, min_leaf_shapes=2, seeds_per_leaf=2,
    )

    result = query_mapper(mapper, 512, 1024, 512, "float16")

    assert mapper["actual_leaves"] <= 3
    assert mapper["stored_seed_count"] <= 7
    assert mapper["training_legal_seed_coverage_pct"] == 100.0
    assert result["decision_path"]
    assert result["seeds"]
    assert result["seeds"][0]["source_shape_id"].startswith("shape-")


def test_query_adapts_exact_cta_count_to_grid_divisor():
    config = dataclasses.asdict(dataclasses.replace(
        fallback_config(),
        output_m_tile=192,
        output_n_tile=128,
        warpgroup_n_tile=128,
        output_cta_count=132,
    ))

    adapted, changes = _adapt_query_config(config, 12288, 768, 1024)

    assert adapted is not None
    assert adapted["output_cta_count"] == 128
    assert changes == ["output_cta_count 132 -> 128"]
