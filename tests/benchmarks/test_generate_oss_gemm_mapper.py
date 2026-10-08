import dataclasses

from benchmarks.generate_oss_gemm_mapper import (
    Point,
    _adapt_query_config,
    _config_from_mapping,
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
            shape_id=f"shape-{index}", op="mm", m=m, n=n, k=512,
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
    stored = result["seeds"][0]["configuration"]
    assert "wgmma_group_size" not in stored
    assert "wgmma_epilogue_pipeline" not in stored
    assert "rhs_tma_load_slices" not in stored
    assert "reschedule_async" in stored


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


def test_config_loader_preserves_explicit_optional_nulls():
    mapping = dataclasses.asdict(fallback_config())
    mapping["output_tma_inner_size"] = None

    config = _config_from_mapping(mapping)

    assert config.output_tma_inner_size is None


def test_query_rejects_illegal_blocked_output_tma_transfers():
    blocked = dataclasses.asdict(dataclasses.replace(
        fallback_config(),
        output_n_tile=48,
        warpgroup_n_tile=48,
        output_store="stmatrix",
        output_tma_inner_size=128,
    ))

    adapted, changes = _adapt_query_config(
        blocked, 128, 64, 128, allow_partial=True,
    )

    assert adapted is None
    assert changes == ["blocked output TMA requires 64-column divisibility"]


def test_mapper_routes_mm_and_addmm_independently():
    points = _points()
    points.extend(
        dataclasses.replace(point, shape_id=f"add-{point.shape_id}", op="addmm")
        for point in _points()
    )

    mapper = build_mapper(
        points, max_leaves=6, min_leaf_shapes=2, seeds_per_leaf=2,
    )

    mm = query_mapper(mapper, 512, 1024, 512, "float16", "mm")
    addmm = query_mapper(mapper, 512, 1024, 512, "float16", "addmm")
    assert mm["group"] == "mm/float16/k128"
    assert addmm["group"] == "addmm/float16/k128"


def test_mapper_preserves_a_transpose_winner_in_its_leaf():
    points = _points()
    points[-1] = dataclasses.replace(
        points[-1],
        config=dataclasses.replace(points[-1].config, transpose_output=True),
    )

    mapper = build_mapper(
        points, max_leaves=1, min_leaf_shapes=2, seeds_per_leaf=2,
    )
    result = query_mapper(mapper, 512, 1024, 512, "float16")

    assert any(
        seed["configuration"].get("transpose_output") is True
        for seed in result["seeds"]
    )


def test_mapper_preserves_an_rhs_promotion_winner_in_its_leaf():
    points = _points()
    points[-1] = dataclasses.replace(
        points[-1],
        config=dataclasses.replace(
            points[-1].config, rhs_l2_promotion="256B",
        ),
    )

    mapper = build_mapper(
        points, max_leaves=1, min_leaf_shapes=2, seeds_per_leaf=2,
    )
    result = query_mapper(mapper, 512, 1024, 512, "float16")

    assert any(
        seed["configuration"].get("rhs_l2_promotion") == "256B"
        for seed in result["seeds"]
    )
