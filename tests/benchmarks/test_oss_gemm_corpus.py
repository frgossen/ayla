import dataclasses
import json

from benchmarks.benchmark_oss_pytorch_gemm_shapes import (
    _adapt_transfer_schedule,
    transfer_configs,
)

from benchmarks.oss_gemm_corpus import (
    Shape,
    TensorSpec,
    architecture_gap_configs,
    benchmark_shapes,
    cublas_inspired_configs,
    cluster_tuning_configs,
    coarse_tile_configs,
    exploration_configs,
    exact_cta_configs,
    exact_cta_refinement_configs,
    fallback_config,
    gap_tuning_configs,
    load_shapes,
    local_refinement_configs,
    multicast_configs,
    narrow_tile_configs,
    narrow_wg2d_configs,
    pipeline_tile_configs,
    regime_name,
    refinement_configs,
    resource_tuning_configs,
    rhs_tma_slicing_configs,
    smem_reuse_configs,
    seed_configs,
    static_loop_configs,
    targeted_tuning_configs,
    tile_refinement_configs,
    two_dimensional_wg_configs,
    workgroup_parallelism_configs,
    validate_shape,
)
from benchmarks.report_oss_pytorch_gemm_heuristics import (
    render,
    shape_improvement_rows,
)


def _shape(
    *,
    op="mm",
    batch=None,
    m=128,
    n=256,
    k=64,
    dtype="float16",
    lhs_stride=None,
    rhs_stride=None,
):
    lhs_stride = lhs_stride or (k, 1)
    rhs_stride = rhs_stride or (1, k)
    return Shape(
        op=op, batch=batch, m=m, n=n, k=k,
        args=(
            TensorSpec((m, k), tuple(lhs_stride), dtype),
            TensorSpec((k, n), tuple(rhs_stride), dtype),
        ),
        count=7, suites=("huggingface",), source_records=1,
    )


def test_load_shapes_pools_suites_but_preserves_layout(tmp_path):
    entries = [
        {
            "op": "mm", "m": 128, "n": 256, "k": 64, "count": count,
            "suite": suite,
            "args": [
                {"shape": [128, 64], "stride": [64, 1], "dtype": "float16"},
                {"shape": [64, 256], "stride": rhs_stride, "dtype": "float16"},
            ],
        }
        for count, suite, rhs_stride in [
            (2, "huggingface", [1, 64]),
            (3, "timm", [1, 64]),
            (5, "torchbench", [256, 1]),
        ]
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(entries))

    shapes = load_shapes(corpus)

    assert len(shapes) == 2
    pooled = next(shape for shape in shapes if shape.args[1].stride == (1, 64))
    assert pooled.count == 5
    assert pooled.suites == ("huggingface", "timm")
    assert pooled.source_records == 2


def test_validation_reports_independent_reason_codes():
    assert validate_shape(_shape()).supported
    assert validate_shape(_shape(dtype="float8_e4m3fn")).reason_codes == ("dtype",)
    assert validate_shape(_shape(rhs_stride=(256, 1))).supported
    assert validate_shape(_shape(k=65)).reason_codes == ("alignment",)
    assert validate_shape(_shape(n=255)).reason_codes == ("alignment",)
    assert validate_shape(_shape(op="addmm")).reason_codes == ("op",)
    assert validate_shape(_shape(op="bmm", batch=8)).reason_codes == ("batching",)
    assert validate_shape(
        _shape(m=128, n=128, k=65_536),
    ).reason_codes == ("split_k",)
    assert validate_shape(
        _shape(m=256, n=256, k=32_768),
    ).reason_codes == ("split_k",)
    assert validate_shape(
        _shape(m=192, n=64, k=401_536),
    ).reason_codes == ("split_k",)
    assert validate_shape(
        _shape(m=512, n=512, k=8192),
    ).reason_codes == ("split_k",)
    assert validate_shape(
        _shape(m=128, n=512, k=4096),
    ).reason_codes == ("split_k",)
    assert validate_shape(
        _shape(m=2, n=768, k=3072),
    ).reason_codes == ("split_k", "small_m")
    assert validate_shape(
        _shape(m=2, n=2048, k=128),
    ).reason_codes == ("small_m",)
    assert validate_shape(_shape(m=128, n=129, k=65_536)).reason_codes == (
        "split_k", "alignment",
    )
    assert validate_shape(_shape(m=768, n=768, k=294_848)).supported
    assert validate_shape(_shape(m=768, n=768, k=8192)).supported
    assert validate_shape(_shape(m=2048, n=256, k=8192)).supported
    assert validate_shape(_shape(m=1024, n=1024, k=4096)).supported
    assert validate_shape(_shape(m=8192, n=768, k=768)).supported


def test_regime_and_seeds_expose_every_schedule_knob():
    shape = _shape(m=4096, n=256, k=128)
    assert regime_name(shape) == (
        "float16/canonical_nt/large/tall/k128"
    )
    seeds = seed_configs(shape)
    assert len(seeds) == 3
    assert all(seed.operand_k_tile == 64 for seed in seeds)
    assert all(seed.builder_kwargs()["operand_move_attrs"] for seed in seeds)
    assert any(
        config.operand_k_tile == 128
        for config in exploration_configs(shape)
    )

    learned = seed_configs(_shape(m=512, n=512, k=8192))
    assert {config.output_n_tile for config in learned} == {64, 96}
    assert {config.operand_k_tile for config in learned} == {128}

    copied_tall = seed_configs(_shape(m=12_608, n=768, k=3072))
    assert copied_tall[0].name == "copy_m192_n192_2tile_k64_s4"
    copied_persistent = seed_configs(_shape(m=16_384, n=4096, k=1024))
    assert copied_persistent[0].name == "copy_m128_n256_16tile_k64_s4"

    tall_narrow = seed_configs(_shape(m=61_568, n=256, k=1024))
    assert all(config.warpgroup_m_tile == 64 for config in tall_narrow)
    huge_m = seed_configs(_shape(m=401_536, n=512, k=64))
    assert huge_m[0].multicast
    assert huge_m[0].cluster_n_ctas == 4


def test_benchmark_shapes_pool_data_orders():
    nt = _shape()
    nn = _shape(rhs_stride=(256, 1))

    [pooled] = benchmark_shapes([validate_shape(nt), validate_shape(nn)])

    assert pooled.count == nt.count + nn.count
    assert pooled.args[0].stride == (64, 1)
    assert pooled.args[1].stride == (1, 64)


def test_fallback_avoids_tma_output_and_multi_tile_constraints():
    fallback = fallback_config()
    assert fallback.output_tiles_per_cta == 1
    assert fallback.output_store == "scalar"
    assert fallback.output_tma_inner_size is None
    assert {config.operand_stages for config in refinement_configs(_shape())} == {
        2, 4,
    }
    assert {
        (config.output_m_tile, config.output_n_tile)
        for config in tile_refinement_configs()
    } == {(64, 128), (128, 128)}
    assert any(
        config.dst_l2_cache_hint == "evict_last"
        for config in tile_refinement_configs()
    )


def test_targeted_tuning_covers_small_tiles_and_conditional_k128():
    configs = targeted_tuning_configs(_shape(k=128))
    assert any(
        config.output_m_tile == config.output_n_tile == 64
        for config in configs
    )
    assert any(config.operand_k_tile == 128 for config in configs)
    assert all(
        config.operand_k_tile == 64
        for config in targeted_tuning_configs(_shape(k=64))
    )


def test_cublas_inspired_configs_cover_observed_geometry():
    configs = cublas_inspired_configs(_shape(m=16_384, n=4096, k=1024))
    assert any(
        config.output_n_tile == 48 and config.operand_stages == 15
        for config in configs
    )
    assert any(
        config.output_n_tile == 80 and config.operand_stages == 11
        for config in configs
    )
    assert any(
        config.output_n_tile == 152 and config.operand_stages == 3
        for config in configs
    )
    assert any(
        config.output_n_tile == 160 and config.operand_stages == 5
        for config in configs
    )
    assert any(
        config.output_m_tile == config.output_n_tile == 192
        and config.output_tiles_per_cta == 8
        for config in configs
    )
    assert any(
        config.output_m_tile == 128 and config.output_n_tile == 256
        and config.output_tiles_per_cta == 16
        for config in configs
    )


def test_resource_tuning_varies_tile_shape_and_pipeline_depth():
    configs = resource_tuning_configs(_shape(m=4096, n=4096, k=4096))
    reused = [config for config in configs if config.unsafe_reuse_smem]
    assert reused
    assert all(config.output_m_tile == 64 for config in reused)
    assert {config.operand_stages for config in reused} == {2, 3, 4, 5, 6}
    assert any(
        config.output_m_tile == 256 and not config.unsafe_reuse_smem
        for config in configs
    )


def test_coarse_tile_configs_cover_unmeasured_geometry():
    configs = coarse_tile_configs(_shape(m=1024, n=1024, k=4096))
    by_name = {config.name: config for config in configs}

    assert "coarse_m64_n112_k64_s3" in by_name
    assert "coarse_m128_n224_k64_s4" in by_name
    assert "coarse_m192_n160_k64_s3" in by_name
    assert "coarse_m256_n128_k64_s3" in by_name
    assert by_name["coarse_m128_n128_k64_s4"].output_tma_inner_size == 128


def test_coarse_tile_configs_respect_problem_bounds():
    configs = coarse_tile_configs(_shape(m=128, n=96, k=4096))

    assert configs
    assert all(config.output_m_tile <= 128 for config in configs)
    assert all(config.output_n_tile <= 96 for config in configs)


def test_two_dimensional_wg_configs_cover_both_linearization_orders():
    configs = two_dimensional_wg_configs(_shape(m=1024, n=1024, k=4096))

    assert {config.warpgroup_fast_axis for config in configs} == {"m", "n"}
    assert any(
        config.output_n_tile == 256 and config.warpgroup_n_tile == 64
        for config in configs
    )
    assert any(config.operand_k_tile == 128 for config in configs)
    assert not two_dimensional_wg_configs(_shape(m=192, n=320, k=4096))


def test_multicast_configs_choose_the_shared_operand_from_cluster_axis():
    configs = multicast_configs(_shape(m=1024, n=1024, k=4096))

    assert {config.cluster_m_ctas for config in configs} == {1, 2, 4}
    assert {config.cluster_n_ctas for config in configs} == {1, 2, 4}
    for config in configs:
        kwargs = config.builder_kwargs()
        attrs = kwargs["operand_move_attrs"]
        assert isinstance(attrs, dict)
        if config.cluster_m_ctas > 1:
            assert attrs["rhs"]["multicast"] is True
        else:
            assert attrs["lhs"]["multicast"] is True


def test_pipeline_tile_configs_refine_depth_and_k_tile():
    configs = pipeline_tile_configs(_shape(m=1024, n=1024, k=4096))
    by_name = {config.name: config for config in configs}

    assert "pipe_m128_n112_k64_s6" in by_name
    assert "pipe_m192_n80_k64_s5" in by_name
    assert "pipe_m64_n112_k128_s4" in by_name
    assert "pipe_m256_n96_k128_s2" in by_name
    assert all(
        config.operand_k_tile == 64
        for config in pipeline_tile_configs(_shape(k=192))
    )


def test_gap_tuning_configs_cover_missing_geometry_and_epilogues():
    configs = gap_tuning_configs(_shape(m=1024, n=1024, k=4096))
    by_name = {config.name: config for config in configs}

    assert "gap_m64_n56_k128_s4" in by_name
    assert "gap_m64_n72_k64_s6" in by_name
    assert "gap_m128_n64_block_k64_s3" in by_name
    assert "gap_wg2d_m128_n256_wn128_k64_s2_nfast" in by_name
    assert "gap_m128_n80_k64_s6_split2_dbuf2" in by_name
    assert by_name["gap_m128_n80_k64_s6_split2_dbuf2"].output_slices == 2


def test_narrow_tile_configs_cross_m_tile_and_pipeline_depth():
    configs = narrow_tile_configs(_shape(m=1024, n=1024, k=4096))
    names = {config.name for config in configs}

    assert "narrow_m64_n48_k64_s8" in names
    assert "narrow_m128_n56_k128_s4" in names
    assert "narrow_m192_n72_k128_s3" in names
    assert "narrow_m256_n48_k64_s4" in names


def test_exact_cta_configs_cover_balanced_grid_and_axis_orders():
    configs = exact_cta_configs(_shape(m=12_544, n=2048, k=512))

    assert {config.output_tile_fast_axis for config in configs} == {"m", "n"}
    assert any(
        config.output_cta_count == 132
        and config.output_m_tile == 192
        and config.output_n_tile == 128
        for config in configs
    )
    assert all(config.output_tiles_per_cta == 1 for config in configs)


def test_exact_cta_refinement_covers_slices_and_depth():
    configs = exact_cta_refinement_configs(
        _shape(m=12_544, n=2048, k=512),
    )

    assert any(config.output_slices == 2 for config in configs)
    assert any(config.output_store_buffers == 2 for config in configs)
    assert any(config.operand_stages == 4 for config in configs)
    assert all(config.output_cta_count is not None for config in configs)


def test_local_refinement_changes_one_winner_neighborhood():
    shape = _shape(m=1024, n=1024, k=4096)
    base = next(
        config for config in gap_tuning_configs(shape)
        if config.name == "gap_m128_n64_block_k128_s4"
    )
    configs = local_refinement_configs(shape, base)

    assert configs
    assert any(config.operand_stages == 3 for config in configs)
    assert any(config.output_n_tile == 72 for config in configs)
    assert any(config.output_slices == 2 for config in configs)
    assert len({config.name for config in configs}) == len(configs)


def test_rhs_tma_slicing_varies_only_load_transaction_count():
    shape = _shape(m=768, n=768, k=8192)
    base = dataclasses.replace(
        seed_configs(shape)[0],
        output_n_tile=96,
        warpgroup_n_tile=96,
        rhs_tma_load_slices=None,
    )

    configs = rhs_tma_slicing_configs(shape, base)

    assert {config.rhs_tma_load_slices for config in configs} == {2, 3, 4, 6, 8}
    for config in configs:
        assert dataclasses.replace(
            config, name=base.name, rhs_tma_load_slices=None,
        ) == base


def test_workgroup_parallelism_preserves_cta_tile_and_adds_workers():
    shape = _shape(m=768, n=768, k=8192)
    base = dataclasses.replace(
        seed_configs(shape)[0],
        output_m_tile=64,
        output_n_tile=96,
        warpgroup_n_tile=96,
    )

    configs = workgroup_parallelism_configs(shape, base)

    assert {config.warpgroup_n_tile for config in configs} == {24, 32, 48}
    assert all(config.output_n_tile == 96 for config in configs)


def test_smem_reuse_only_changes_reuse_policy():
    shape = _shape(m=1024, n=1024, k=1024)
    base = seed_configs(shape)[1]

    [config] = smem_reuse_configs(shape, base)

    assert config.unsafe_reuse_smem
    assert dataclasses.replace(
        config, name=base.name, unsafe_reuse_smem=False,
    ) == base


def test_local_refinement_preserves_partial_compact_static_grids():
    shape = _shape(m=128_100, n=1536, k=512)
    base = next(
        config for config in static_loop_configs(shape)
        if config.output_m_tile == 192
        and config.output_n_tile == 128
        and config.output_cta_count == 132
    )

    configs = local_refinement_configs(shape, base)

    assert any(config.output_cta_count == 120 for config in configs)
    assert any(config.output_m_tile == 128 for config in configs)
    assert all(not config.unroll_output_tiles for config in configs)


def test_local_refinement_can_compact_a_regular_winner():
    shape = _shape(m=8192, n=768, k=768)
    base = dataclasses.replace(
        seed_configs(shape)[0],
        output_m_tile=128,
        output_n_tile=192,
        warpgroup_n_tile=192,
    )

    configs = local_refinement_configs(shape, base)

    assert any(
        config.output_cta_count == 132
        and not config.unroll_output_tiles
        for config in configs
    )
    assert any(config.output_slices == 3 for config in configs)


def test_cluster_tuning_configs_cover_narrow_clustered_tiles():
    configs = cluster_tuning_configs(_shape(m=768, n=768, k=8192))
    names = {config.name for config in configs}

    assert "cluster_n8_m64_n96_k128_s4" in names
    assert "cluster_m4_m64_n72_k128_s6" in names
    assert all(config.multicast for config in configs)


def test_narrow_wg2d_configs_use_two_and_three_compute_groups():
    configs = narrow_wg2d_configs(_shape(m=768, n=768, k=8192))

    assert any(
        config.output_m_tile == 128 and config.output_n_tile == 72
        and config.operand_k_tile == 128 and config.operand_stages == 3
        for config in configs
    )
    assert any(
        config.output_m_tile == 192 and config.output_n_tile == 72
        and config.warpgroup_m_tile == 64
        for config in configs
    )
    assert all(config.warpgroup_m_tile == 64 for config in configs)


def test_static_loop_configs_cover_large_and_partial_output_grids():
    configs = static_loop_configs(_shape(m=128_100, n=1536, k=512))

    assert configs
    assert all(config.output_cta_count in (128, 132) for config in configs)
    assert all(not config.unroll_output_tiles for config in configs)
    assert any(
        config.output_m_tile == 192 and config.output_n_tile == 128
        for config in configs
    )


def test_architecture_gap_configs_cover_deep_and_tall_families():
    deep = architecture_gap_configs(_shape(m=768, n=768, k=8192))
    tall = architecture_gap_configs(_shape(m=128_100, n=1536, k=512))

    assert any(
        config.output_m_tile == 64 and config.output_n_tile == 64
        and config.operand_stages == 13
        for config in deep
    )
    assert any(
        config.output_m_tile == 64 and config.output_n_tile == 80
        for config in deep
    )
    assert any(config.cluster_n_ctas == 8 for config in deep)
    assert any(
        config.output_n_tile == 96
        and config.warpgroup_n_tile == 32
        and config.cluster_n_ctas == 8
        for config in deep
    )
    assert any(
        config.output_m_tile == 256 and config.output_n_tile == 160
        and config.multicast
        for config in tall
    )


def test_transfer_adapts_exact_cta_count_to_target_divisor():
    target = _shape(m=12288, n=768, k=1024)
    base = dataclasses.replace(
        seed_configs(target)[0],
        output_m_tile=192,
        output_n_tile=128,
        warpgroup_n_tile=128,
        output_cta_count=132,
    )

    adapted = _adapt_transfer_schedule(target, base)

    assert adapted is not None
    assert adapted.output_cta_count == 128
    assert adapted.name.startswith("xfer_m192_n128_k")


def test_transfer_preserves_compact_static_grid_without_divisibility():
    target = _shape(m=128_100, n=1536, k=512)
    base = next(
        config for config in static_loop_configs(target)
        if config.output_m_tile == 192
        and config.output_n_tile == 128
        and config.output_cta_count == 132
    )

    adapted = _adapt_transfer_schedule(target, base)

    assert adapted is not None
    assert adapted.output_cta_count == 132
    assert not adapted.unroll_output_tiles


def test_transfer_uses_unmeasured_neighbor_winner():
    target = _shape(m=12288, n=768, k=1024)
    neighbor = _shape(m=12672, n=768, k=1024)
    winner = dataclasses.replace(
        seed_configs(neighbor)[0],
        name="neighbor_exact",
        output_m_tile=192,
        output_n_tile=128,
        warpgroup_n_tile=128,
        output_cta_count=132,
    )
    rows = [
        {
            "shape_id": neighbor.shape_id,
            "implementation": "torch",
            "status": "ok",
            "latency_ms": 1.0,
            "dtype": neighbor.dtype,
            "m": neighbor.m,
            "n": neighbor.n,
            "k": neighbor.k,
        },
        {
            "shape_id": neighbor.shape_id,
            "implementation": "ayla",
            "status": "ok",
            "latency_ms": 0.9,
            "configuration": dataclasses.asdict(winner),
            "m": neighbor.m,
            "n": neighbor.n,
            "k": neighbor.k,
        },
    ]

    configs = transfer_configs(target, rows)

    assert len(configs) == 1
    assert configs[0].output_cta_count == 128


def test_report_ranks_configs_within_regime(tmp_path):
    shape = _shape()
    rows = []
    for name, percentage in [("seed_a", 90.0), ("seed_b", 110.0)]:
        dataclass_config = seed_configs(shape)[0]
        config_dict = {
            **dataclass_config.__dict__,
            "name": name,
        }
        rows.append({
            "shape_id": shape.shape_id,
            "regime": regime_name(shape),
            "frequency": shape.count,
            "implementation": "ayla",
            "status": "ok",
            "torch_pct": percentage,
            "configuration": config_dict,
            "dtype": "float16",
            "layout": "canonical_nt",
            "m_bucket": "xs",
            "n_bucket": "small",
            "k_bucket": "xs",
            "aspect": "balanced",
            "alignment": "k64",
        })

    document = render(rows, tmp_path / "raw.json")

    assert document.index("`seed_b`") < document.index("`seed_a`")
    assert "110.0%" in document
    assert "cluster_m_ctas=1" in document


def test_shape_improvement_rows_select_fastest_config():
    common = {
        "shape_id": "shape-a",
        "m": 128,
        "n": 256,
        "k": 64,
        "dtype": "float16",
        "layout": "canonical_nt",
        "frequency": 7,
        "regime": "test",
        "status": "ok",
    }
    rows = [
        {
            **common, "implementation": "torch", "latency_ms": 1.0,
            "tflops": 10.0, "configuration": {"name": "torch.mm"},
        },
        {
            **common, "implementation": "ayla", "latency_ms": 0.9,
            "tflops": 11.0, "configuration": {"name": "slow"},
        },
        {
            **common, "implementation": "ayla", "latency_ms": 0.8,
            "tflops": 12.0, "configuration": {"name": "fast"},
        },
    ]

    result = shape_improvement_rows(rows)

    assert len(result) == 1
    assert result[0]["configuration_name"] == "fast"
    assert result[0]["percentage_of_torch"] == 125.0
    assert result[0]["beats_torch"] is True
