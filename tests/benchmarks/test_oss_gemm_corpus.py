import dataclasses
import json

from benchmarks.benchmark_oss_pytorch_gemm_shapes import (
    _adapt_transfer_schedule,
    _load_explicit_candidates,
    _shapes_with_current_best,
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
    documented_gap_ablation_configs,
    exploration_configs,
    exact_cta_configs,
    exact_cta_refinement_configs,
    fallback_config,
    gap_tuning_configs,
    grid_swap_ablation_configs,
    load_shapes,
    local_refinement_configs,
    multicast_ablation_configs,
    multicast_configs,
    multicast_incumbent_neighborhood_configs,
    multicast_resweep_configs,
    multicast_winner_transfer_configs,
    narrow_tile_configs,
    narrow_wg2d_configs,
    output_n_grouping_ablation_configs,
    pipeline_tile_configs,
    regime_name,
    refinement_configs,
    resource_tuning_configs,
    rhs_tma_slicing_configs,
    smem_reuse_configs,
    seed_configs,
    small_cta_writeback_configs,
    small_cta_writeback_refinement_configs,
    small_cta_writeback_transfer_configs,
    static_loop_configs,
    tail_end_configs,
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
    matrix_args = (
        (
            TensorSpec((batch, m, k), (m * k, k, 1), dtype),
            TensorSpec((batch, k, n), (k * n, n, 1), dtype),
        )
        if op == "bmm" and batch is not None
        else (
            TensorSpec((m, k), tuple(lhs_stride), dtype),
            TensorSpec((k, n), tuple(rhs_stride), dtype),
        )
    )
    return Shape(
        op=op, batch=batch, m=m, n=n, k=k,
        args=(
            (TensorSpec((n,), (1,), dtype),) if op == "addmm" else ()
        ) + matrix_args,
        count=7, suites=("huggingface",), source_records=1,
    )


def test_recheck_uses_only_shapes_with_an_explicit_ranked_config():
    ranked = _shape(m=128)
    missing = _shape(m=256)

    selected = _shapes_with_current_best(
        [ranked, missing],
        lambda shape: {} if shape is ranked else None,
    )

    assert selected == [ranked]


def test_explicit_candidate_manifest_keeps_all_unique_target_configs(tmp_path):
    first = dataclasses.asdict(fallback_config())
    second = {**first, "operand_stages": first["operand_stages"] + 1}
    manifest = tmp_path / "candidates.json"
    manifest.write_text(json.dumps({"transfers": [
        {"target": "shape-a", "configuration": first},
        {"target": "shape-a", "configuration": second},
        {"target": "shape-a", "configuration": first},
    ]}))

    candidates = _load_explicit_candidates(manifest)

    assert len(candidates["shape-a"]) == 2
    assert all(config.name.startswith("documented_") for config in candidates["shape-a"])


def test_documented_gap_ablation_covers_previously_missing_values():
    shape = _shape(m=4096, n=4096, k=4096)
    base = dataclasses.replace(
        fallback_config(),
        output_m_tile=64,
        output_n_tile=192,
        warpgroup_n_tile=192,
        output_store="stmatrix",
        operand_stages=4,
    )

    configs = documented_gap_ablation_configs(shape, base)
    deep_configs = documented_gap_ablation_configs(
        shape,
        dataclasses.replace(
            fallback_config(), output_m_tile=64, output_n_tile=64,
            warpgroup_n_tile=64, operand_stages=4,
        ),
    )

    assert any(config.reschedule_async for config in configs)
    assert {config.output_tiles_per_cta for config in configs} >= {4, 5, 6}
    assert {config.output_n_tiles_per_cta for config in configs} >= {2, 4, 6}
    assert any(config.output_slices == 3 for config in configs)
    assert {config.operand_stages for config in deep_configs} >= {7, 8, 9}


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
    assert validate_shape(_shape(op="addmm")).supported
    assert validate_shape(_shape(op="bmm", batch=8)).supported
    assert validate_shape(_shape(op="bmm", batch=1)).reason_codes == ("batch",)
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
        _shape(m=384, n=1536, k=25_088),
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
    assert validate_shape(
        _shape(m=768, n=768, k=294_848),
    ).reason_codes == ("split_k",)
    assert validate_shape(_shape(m=768, n=768, k=8192)).supported
    assert validate_shape(_shape(m=384, n=1536, k=12_544)).supported
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


def test_benchmark_shapes_keep_mm_and_addmm_as_distinct_workloads():
    mm = _shape()
    addmm = _shape(op="addmm")

    pooled = benchmark_shapes([validate_shape(mm), validate_shape(addmm)])

    assert {shape.op for shape in pooled} == {"mm", "addmm"}
    assert next(shape for shape in pooled if shape.op == "addmm").args[0] == (
        TensorSpec((256,), (1,), "float16")
    )


def test_benchmark_shapes_preserve_bmm_batch_and_canonicalize_bt():
    bmm = _shape(op="bmm", batch=8)

    [pooled] = benchmark_shapes([validate_shape(bmm)])

    assert pooled.batch == 8
    assert pooled.args == (
        TensorSpec((8, 128, 64), (8192, 64, 1), "float16"),
        TensorSpec((8, 256, 64), (16384, 64, 1), "float16"),
    )


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


def test_resource_tuning_varies_safe_tile_shape_and_pipeline_depth():
    configs = resource_tuning_configs(_shape(m=4096, n=4096, k=4096))
    assert configs
    assert all(not config.unsafe_reuse_smem for config in configs)
    assert {config.operand_stages for config in configs} == {2, 3, 4}
    assert any(
        config.output_m_tile == 256 and config.output_n_tile == 128
        for config in configs
    )


def test_tail_end_configs_use_only_safe_existing_schedule_controls():
    deep = tail_end_configs(_shape(m=768, n=768, k=4096))
    assert any(
        config.output_n_tile == 96
        and config.warpgroup_n_tile == 32
        and config.operand_stages == 10
        and config.output_store == "scalar"
        for config in deep
    )
    assert any(config.cyclic_operand_pipeline for config in deep)
    assert any(config.wgmma_pending_groups == 1 for config in deep)
    assert all(
        max(config.cluster_m_ctas, config.cluster_n_ctas) <= 4
        for config in deep
    )

    tall = tail_end_configs(_shape(m=128100, n=1536, k=512))
    assert any(
        config.output_m_tile == 192
        and config.warpgroup_m_tile == 64
        for config in tall
    )
    assert any(config.output_cta_count == 132 for config in tall)
    assert any(config.multicast for config in tall)

    wide = tail_end_configs(_shape(m=512, n=50272, k=768))
    assert any(config.output_n_tile == 384 for config in wide)

    for config in (*deep, *tall, *wide):
        estimated_smem = (
            2
            * (config.output_m_tile + config.output_n_tile)
            * config.operand_k_tile
            * config.operand_stages
            + 2
            * config.output_m_tile
            * config.output_n_tile
            * config.output_store_buffers
        )
        assert estimated_smem <= 225 * 1024
        assert not config.unsafe_reuse_smem
        assert not (
            config.cluster_m_ctas > 1 and config.cluster_n_ctas > 1
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


def test_two_dimensional_wg_configs_use_the_existing_linearization_order():
    configs = two_dimensional_wg_configs(_shape(m=1024, n=1024, k=4096))

    assert {config.warpgroup_fast_axis for config in configs} == {"n"}
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


def test_multicast_ablation_pairs_each_geometry_with_a_local_control():
    configs = multicast_ablation_configs(_shape(m=1568, n=2048, k=512))
    local = [config for config in configs if not config.multicast]
    clustered = [config for config in configs if config.multicast]

    assert local
    assert clustered
    local_geometries = {
        (
            config.output_m_tile,
            config.output_n_tile,
            config.operand_k_tile,
            config.operand_stages,
        )
        for config in local
    }
    assert all(
        (
            config.output_m_tile,
            config.output_n_tile,
            config.operand_k_tile,
            config.operand_stages,
        ) in local_geometries
        for config in clustered
    )
    assert any(config.cluster_n_ctas == 4 for config in clustered)
    assert all(
        config.cluster_m_ctas == config.cluster_n_ctas == 1
        for config in local
    )


def test_multicast_ablation_respects_pipeline_and_smem_bounds():
    short_k = multicast_ablation_configs(_shape(m=4096, n=4096, k=128))

    assert short_k
    for config in short_k:
        assert config.operand_stages <= 128 // config.operand_k_tile
        estimated_smem = (
            2
            * (config.output_m_tile + config.output_n_tile)
            * config.operand_k_tile
            * config.operand_stages
            + 2 * config.output_m_tile * config.output_n_tile
        )
        assert estimated_smem <= 225 * 1024


def test_multicast_resweep_pairs_both_axes_and_cluster_sizes():
    shape = _shape(m=4096, n=4096, k=512)
    configs = multicast_resweep_configs(shape, seed_configs(shape)[0])
    local = [config for config in configs if not config.multicast]
    clustered = [config for config in configs if config.multicast]

    assert local
    assert clustered
    assert {config.cluster_m_ctas for config in clustered} >= {1, 2, 4, 8}
    assert {config.cluster_n_ctas for config in clustered} >= {1, 2, 4, 8}

    def geometry(config):
        values = dataclasses.asdict(config)
        for field in (
            "name", "multicast", "cluster_m_ctas", "cluster_n_ctas",
        ):
            values.pop(field)
        return values

    local_geometries = [geometry(config) for config in local]
    assert all(geometry(config) in local_geometries for config in clustered)


def test_multicast_resweep_includes_feasible_deep_pipelines():
    shape = _shape(m=768, n=768, k=8192)
    configs = multicast_resweep_configs(shape, seed_configs(shape)[0])

    assert configs
    assert {config.operand_stages for config in configs} >= {6, 8}
    assert any(config.multicast for config in configs)
    for config in configs:
        estimated_smem = (
            2
            * (config.output_m_tile + config.output_n_tile)
            * config.operand_k_tile
            * config.operand_stages
            + 2 * config.output_m_tile * config.output_n_tile
        )
        assert estimated_smem <= 225 * 1024


def test_multicast_winner_transfer_keeps_exact_local_control():
    configs = multicast_winner_transfer_configs(
        _shape(m=8192, n=2048, k=128),
    )
    local = [config for config in configs if not config.multicast]
    multicast = [config for config in configs if config.multicast]
    assert len(local) == 7
    assert multicast

    def geometry(config):
        values = dataclasses.asdict(config)
        for field in (
            "name", "multicast", "cluster_m_ctas", "cluster_n_ctas",
        ):
            values.pop(field)
        return values

    local_geometries = [geometry(config) for config in local]
    assert all(geometry(config) in local_geometries for config in multicast)
    assert {config.cluster_m_ctas for config in multicast} >= {1, 2, 4, 8}
    assert {config.cluster_n_ctas for config in multicast} >= {1, 2, 4, 8}


def test_multicast_neighborhood_pairs_every_cluster_and_cache_policy():
    shape = _shape(m=32768, n=256, k=256)
    configs = multicast_incumbent_neighborhood_configs(
        shape, seed_configs(shape)[0],
    )
    local = [config for config in configs if not config.multicast]
    clustered = [config for config in configs if config.multicast]
    assert clustered

    def local_key(config):
        values = dataclasses.asdict(config)
        for field in (
            "name", "multicast", "cluster_m_ctas", "cluster_n_ctas",
        ):
            values.pop(field)
        return values

    local_keys = [local_key(config) for config in local]
    assert all(local_key(config) in local_keys for config in clustered)
    assert {
        (config.lhs_l2_cache_hint, config.rhs_l2_cache_hint)
        for config in local
        if config.output_m_tile == 128 and config.output_n_tile == 128
    } >= {
        ("evict_first", "evict_first"),
        ("evict_first", "evict_last"),
        ("evict_last", "evict_first"),
        ("evict_last", "evict_last"),
    }


def test_multicast_neighborhood_keeps_deep_sliced_incumbent():
    shape = _shape(m=512, n=1024, k=4096)
    base = dataclasses.replace(
        seed_configs(shape)[0],
        output_m_tile=64,
        output_n_tile=96,
        warpgroup_n_tile=96,
        operand_k_tile=64,
        operand_stages=8,
        output_slices=2,
        output_tma_inner_size=None,
        cyclic_operand_pipeline=True,
    )

    configs = multicast_incumbent_neighborhood_configs(shape, base)

    assert any(
        not config.multicast
        and config.output_slices == 2
        and config.operand_stages == 8
        for config in configs
    )
    assert any(
        config.multicast
        and config.output_slices == 2
        and config.operand_stages == 8
        for config in configs
    )
    assert all(
        config.output_tma_inner_size is None
        or config.output_n_tile // config.output_slices >= 64
        for config in configs
    )


def test_search_policy_does_not_become_redundant_builder_flags():
    config = dataclasses.replace(
        seed_configs(_shape(m=1024, n=1024, k=4096))[0],
        cluster_m_ctas=2,
        multicast=True,
        output_cta_count=128,
        output_tile_fast_axis="m",
        warpgroup_fast_axis="m",
    )

    kwargs = config.builder_kwargs()

    assert kwargs["operand_move_attrs"]["rhs"]["multicast"] is True
    assert not {
        "cluster_m_ctas",
        "cluster_n_ctas",
        "output_cta_count",
        "output_tile_fast_axis",
        "unroll_output_tiles",
        "warpgroup_fast_axis",
    } & kwargs.keys()


def test_output_n_grouping_routes_the_existing_builder_control():
    shape = _shape(m=3136, n=2048, k=1024)
    base = dataclasses.replace(
        seed_configs(shape)[0],
        output_m_tile=64,
        output_n_tile=128,
        warpgroup_n_tile=128,
        operand_stages=2,
    )

    configs = output_n_grouping_ablation_configs(shape, base)
    grouped = next(config for config in configs if config.name == "ngroup_g2_cy1")

    assert grouped.builder_kwargs()["output_m_tiles_per_cta"] == 1
    assert grouped.builder_kwargs()["output_n_tiles_per_cta"] == 2
    assert grouped.cyclic_operand_pipeline


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
    assert "gap_wg2d_m128_n256_wn128_k64_s2" in by_name
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


def test_grid_swap_ablation_supports_static_output_groups():
    shape = _shape(m=30_522, n=768, k=2048)
    base = dataclasses.replace(
        fallback_config(),
        output_tiles_per_cta=6,
    )

    configs = grid_swap_ablation_configs(shape, base)

    assert len(configs) == 2
    assert {config.swap_output_grid_axes for config in configs} == {False, True}
    assert all(config.output_tiles_per_cta == 6 for config in configs)


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


def test_rhs_tma_slicing_search_is_retired():
    shape = _shape(m=768, n=768, k=8192)
    base = dataclasses.replace(
        seed_configs(shape)[0],
        output_n_tile=96,
        warpgroup_n_tile=96,
        rhs_tma_load_slices=None,
    )

    configs = rhs_tma_slicing_configs(shape, base)

    assert configs == ()


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
    assert {
        config.output_n_tile // config.warpgroup_n_tile
        for config in configs
    } == {2, 3, 4}
    assert all(
        config.output_slices
        == config.output_n_tile // config.warpgroup_n_tile
        for config in configs
    )
    assert all(config.output_tma_inner_size is None for config in configs)


def test_smem_reuse_search_is_retired():
    shape = _shape(m=1024, n=1024, k=1024)
    base = seed_configs(shape)[1]

    assert smem_reuse_configs(shape, base) == ()


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


def test_small_cta_writeback_refinement_preserves_confirmed_base():
    shape = _shape(m=128_112, n=1024, k=256)

    configs = small_cta_writeback_refinement_configs(shape)
    base = next(config for config in configs if config.name == "smallcta_ref_base")

    assert base.output_cta_count == 396
    assert (base.output_m_tile, base.output_n_tile) == (128, 128)
    assert (base.warpgroup_m_tile, base.warpgroup_n_tile) == (128, 128)
    assert (base.operand_k_tile, base.operand_stages) == (64, 2)
    assert base.output_slices == 8
    assert base.output_store_buffers == 1
    assert base.output_tile_fast_axis == "n"


def test_small_cta_writeback_search_is_one_compute_warpgroup():
    configs = small_cta_writeback_configs(
        _shape(m=128_112, n=1024, k=256),
    )

    assert configs
    assert all(config.warpgroup_m_tile == config.output_m_tile for config in configs)
    assert all(config.warpgroup_n_tile == config.output_n_tile for config in configs)
    assert all(config.output_store_buffers == 1 for config in configs)
    assert all(not config.multicast for config in configs)


def test_small_cta_writeback_transfer_requires_enough_logical_tiles():
    eligible = small_cta_writeback_transfer_configs(
        _shape(m=61_568, n=1024, k=256),
    )
    too_small = small_cta_writeback_transfer_configs(
        _shape(m=256, n=128, k=256),
    )

    assert {config.name for config in eligible} >= {
        "smallcta_xfer_m64_r4",
        "smallcta_xfer_m128_r3",
        "smallcta_xfer_m192_r2",
    }
    assert too_small == ()
