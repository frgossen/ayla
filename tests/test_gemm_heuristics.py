import dataclasses

import pytest

from ayla.gemm_heuristics import (
    apply_gemm_schedule_seed_post_build,
    select_gemm_schedules,
)
from ayla.gpu_dialect import CollectiveAxis


def test_fixed_grid_seed_exposes_required_unroll():
    selection = select_gemm_schedules(
        8192, 768, 768, operation="addmm",
    )
    seed = next(
        seed for seed in selection.seeds
        if seed.fixed_output_unroll_factor is not None
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_grid_size"] > 0
    assert seed.fixed_output_unroll_factor == 2
    assert "fixed output grid" in seed.adaptations[0]


def test_compact_seeds_omit_dropped_experimental_controls():
    selection = select_gemm_schedules(2048, 768, 768)
    redundant = {
        "output_tile_fast_axis",
        "rhs_tma_load_slices",
        "unsafe_reuse_smem",
        "warpgroup_fast_axis",
        "wgmma_epilogue_pipeline",
        "wgmma_group_size",
        "wgmma_pending_groups",
    }

    assert selection.seeds
    assert all(
        redundant.isdisjoint(seed.captured_config)
        for seed in selection.seeds
    )


def test_rescheduled_seed_exposes_and_executes_post_builder_passes(
    monkeypatch,
):
    selection = select_gemm_schedules(4160, 1024, 3072)
    seed = dataclasses.replace(
        selection.seeds[0],
        fixed_output_unroll_factor=None,
        post_builder_passes=("schedule_async", "canonicalize"),
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


def test_rhs_promotion_is_forwarded_to_the_builder():
    selection = select_gemm_schedules(2048, 768, 768)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config.get("rhs_l2_promotion") is not None
    )

    assert seed.builder_config is not None
    rhs = seed.builder_config["operand_move_attrs"]["rhs"]
    assert rhs["l2_promotion"] == seed.captured_config["rhs_l2_promotion"]


def test_n_grouped_winner_routes_existing_builder_controls():
    selection = select_gemm_schedules(2048, 2048, 128)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["output_n_tiles_per_cta"] > 1
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 1
    assert seed.builder_config["output_n_tiles_per_cta"] == 4


def test_swapped_grid_winner_routes_existing_builder_controls():
    selection = select_gemm_schedules(2048, 768, 768)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["swap_output_grid_axes"] is True
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_grid_axes"] == (
        CollectiveAxis.Y, CollectiveAxis.X,
    )


def test_two_dimensional_warpgroup_winner_is_preserved():
    selection = select_gemm_schedules(
        2048, 768, 768, operation="addmm",
    )
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config.get("warpgroup_m_tile") is not None
    )

    assert seed.builder_config is not None
    assert seed.builder_config["warpgroup_m_tile"] == 64


def test_group_five_family_routes_existing_builder_controls():
    selection = select_gemm_schedules(4096, 1024, 256)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["output_tiles_per_cta"] == 5
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 5
    assert seed.builder_config["output_n_tiles_per_cta"] == 1


def test_transpose_winner_is_selected_for_the_original_problem():
    selection = select_gemm_schedules(8192, 768, 768)
    seed = next(seed for seed in selection.seeds if seed.transpose_output)

    assert seed.captured_config["transpose_output"] is True
    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tile"] == 64
    assert seed.builder_config["output_n_tile"] == 128
