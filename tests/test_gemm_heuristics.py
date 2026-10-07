import pytest

from ayla.gemm_heuristics import (
    apply_gemm_schedule_seed_post_build,
    select_gemm_schedules,
)
from ayla.gpu_dialect import CollectiveAxis


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


def test_rescheduled_seed_exposes_and_executes_post_builder_passes(
    monkeypatch,
):
    selection = select_gemm_schedules(4160, 1024, 3072)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config.get("reschedule_async") is True
    )
    assert seed.post_builder_passes == ("schedule_async", "canonicalize")

    calls = []
    monkeypatch.setattr(
        "ayla.passes.schedule_async.ScheduleAsync.run",
        lambda self, module: calls.append("schedule_async") or module,
    )
    monkeypatch.setattr(
        "ayla.passes.canonicalize.Canonicalize.run",
        lambda self, module: calls.append("canonicalize") or module,
    )
    module = object()

    assert apply_gemm_schedule_seed_post_build(module, seed) is module
    assert calls == ["schedule_async", "canonicalize"]


def test_addmm_uses_an_independent_operation_group():
    mm = select_gemm_schedules(128, 4096, 4096)
    addmm = select_gemm_schedules(
        128, 4096, 4096, operation="addmm",
    )

    assert addmm.leaf.startswith("addmm_")
    assert mm.leaf != addmm.leaf


def test_bmm_uses_batch_aware_operation_group_and_partial_edge_tiles():
    partial = select_gemm_schedules(
        64, 16, 128, operation="bmm", batch=128,
    )
    small_batch = select_gemm_schedules(
        768, 64, 64, operation="bmm", batch=12,
    )
    large_batch = select_gemm_schedules(
        768, 64, 64, operation="bmm", batch=144,
    )

    assert partial.leaf.startswith("bmm_")
    assert partial.seeds
    assert any(
        seed.builder_config is not None
        and seed.builder_config["output_n_tile"] > 16
        for seed in partial.seeds
    )
    assert small_batch.leaf != large_batch.leaf


def test_batch_is_rejected_for_non_bmm_operations():
    with pytest.raises(ValueError, match="only meaningful for BMM"):
        select_gemm_schedules(128, 128, 128, batch=2)


def test_bmm_filters_illegal_blocked_output_tma_transfers():
    selection = select_gemm_schedules(
        128, 32, 128, operation="bmm", batch=128,
    )

    assert selection.seeds
    assert all(
        seed.builder_config is None
        or seed.builder_config["output_store"] == "scalar"
        or seed.builder_config["output_tma_inner_size"] is None
        for seed in selection.seeds
    )


def test_addmm_transfer_winners_seed_matching_mm_leaves():
    shapes = (
        (4096, 4096, 1024),
        (512, 6144, 1536),
        (1152, 768, 3072),
        (512, 1024, 4096),
        (12608, 256, 768),
        (32768, 256, 1024),
        (12608, 768, 256),
        (16448, 512, 512),
        (8192, 2048, 256),
        (25664, 384, 128),
        (25216, 320, 320),
        (50176, 768, 256),
    )

    for shape in shapes:
        selection = select_gemm_schedules(*shape)
        assert any(seed.source_shape == shape for seed in selection.seeds)


def test_n_grouped_winner_routes_existing_builder_controls():
    selection = select_gemm_schedules(3136, 2048, 1024)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["name"] == "ngroup3136wide_g2_cy1"
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 1
    assert seed.builder_config["output_n_tiles_per_cta"] == 2
    assert seed.builder_config["cyclic_operand_pipeline"] is True


def test_swapped_n_grouped_winner_routes_existing_builder_controls():
    selection = select_gemm_schedules(2856, 3072, 768)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["name"] == "ng2856tune_swap1"
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 1
    assert seed.builder_config["output_n_tiles_per_cta"] == 3
    assert seed.builder_config["output_grid_axes"] == (
        CollectiveAxis.Y, CollectiveAxis.X,
    )


def test_256_thread_m_grouped_winners_route_existing_builder_controls():
    for shape, group in (
        ((50265, 1024, 512), 6),
        ((6272, 1280, 320), 2),
        ((16448, 512, 1536), 2),
    ):
        selection = select_gemm_schedules(*shape)
        seed = next(
            seed for seed in selection.seeds
            if seed.source_shape == shape
            and seed.captured_config["output_m_tile"] == 128
            and seed.captured_config["output_n_tile"] == 256
            and seed.captured_config["swap_output_grid_axes"] is True
        )

        assert seed.builder_config is not None
        assert seed.builder_config["output_m_tiles_per_cta"] == group
        assert seed.builder_config["output_n_tiles_per_cta"] == 1
        assert seed.builder_config["cyclic_operand_pipeline"] is True
        assert seed.builder_config["output_grid_axes"] == (
            CollectiveAxis.Y, CollectiveAxis.X,
        )


def test_additional_swapped_m_groups_route_existing_builder_controls():
    for shape, group in (((9521, 512, 7936), 2),):
        selection = select_gemm_schedules(*shape)
        seed = next(
            seed for seed in selection.seeds
            if seed.source_shape == shape
            and seed.captured_config["swap_output_grid_axes"] is True
            and seed.captured_config["output_tiles_per_cta"] == group
        )

        assert seed.builder_config is not None
        assert seed.builder_config["output_m_tiles_per_cta"] == group
        assert seed.builder_config["output_grid_axes"] == (
            CollectiveAxis.Y, CollectiveAxis.X,
        )


def test_large_shape_static_groups_route_existing_builder_controls():
    cases = (
        ((6400, 2048, 512), 5, 1),
        ((512, 50272, 768), 1, 6),
    )
    for shape, m_group, n_group in cases:
        selection = select_gemm_schedules(*shape)
        seed = next(
            seed for seed in selection.seeds
            if seed.source_shape == shape
            and seed.captured_config["output_m_tile"] == 128
        )

        assert seed.builder_config is not None
        assert seed.builder_config["output_m_tiles_per_cta"] == m_group
        assert seed.builder_config["output_n_tiles_per_cta"] == n_group
        assert seed.builder_config["ptx_optimization_level"] == 1


def test_30522_group_five_family_routes_existing_builder_controls():
    for k in (2048, 4096, 8192):
        shape = (30522, 768, k)
        selection = select_gemm_schedules(*shape)
        seed = next(
            seed for seed in selection.seeds
            if seed.captured_config["output_m_tile"] == 192
            and seed.captured_config["output_n_tile"] == 192
        )

        assert seed.builder_config is not None
        assert seed.builder_config["output_m_tiles_per_cta"] == 5
        assert seed.builder_config["output_n_tiles_per_cta"] == 1
        assert seed.builder_config["output_grid_axes"] == (
            CollectiveAxis.Y, CollectiveAxis.X,
        )


def test_4000_n_group_six_routes_existing_builder_controls():
    selection = select_gemm_schedules(4000, 4000, 1024)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["name"] == "confirm4000_g6_swap"
    )

    assert seed.source_shape == (4000, 4000, 1024)
    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 1
    assert seed.builder_config["output_n_tiles_per_cta"] == 6
    assert seed.builder_config["operand_stages"] == 4
    assert seed.builder_config["cyclic_operand_pipeline"] is False
    assert seed.builder_config["output_grid_axes"] == (
        CollectiveAxis.Y, CollectiveAxis.X,
    )
