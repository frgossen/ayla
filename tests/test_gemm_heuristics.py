from ayla.gemm_heuristics import select_gemm_schedules


def test_small_cta_seed_retains_the_measured_fixed_grid():
    for shape in (
        (128_100, 1536, 512),
        (128_112, 1024, 256),
        (197_951, 512, 256),
    ):
        selection = select_gemm_schedules(*shape)
        seed = next(
            seed for seed in selection.seeds
            if seed.captured_config["name"]
            == "heuristic_smallcta_m128_n128_k64_s2_split8_r3"
        )

        assert seed.builder_config is not None
        assert seed.builder_config["output_grid_size"] == 396
        assert seed.builder_config["output_m_tiles_per_cta"] == 1
        assert seed.builder_config["output_n_tiles_per_cta"] == 1
        assert seed.builder_config["warpgroup_m_tile"] == 128
        assert seed.builder_config["warpgroup_n_tile"] == 128
        assert seed.builder_config["output_slices"] == 8
        assert "fixed output grid" in seed.adaptations[0]


def test_incremental_winner_preserves_existing_leaf_seeds():
    selection = select_gemm_schedules(1152, 3072, 768)
    names = {seed.captured_config["name"] for seed in selection.seeds}

    assert "hybrid1152_s3_cy1" in names
    assert "waveconfirm_native" in names
