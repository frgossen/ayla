import dataclasses
from collections import Counter

import pytest

from ayla.gemm_heuristics import (
    _adapt_config,
    apply_gemm_schedule_seed_post_build,
    collapse_gemm_heuristic,
    default_reverse_output_rows,
    expand_gemm_heuristic,
    select_gemm_schedules,
)
from ayla.gpu_dialect import CollectiveAxis


def test_packaged_heuristic_covers_both_corpora_with_exact_dtype_dispatch():
    from ayla.gemm_heuristics import _mapper

    mapper = _mapper()
    expected = {
        ("mm", "float16"): 405,
        ("addmm", "float16"): 205,
        ("bmm", "float16"): 139,
        ("mm", "bfloat16"): 262,
        ("addmm", "bfloat16"): 123,
        ("bmm", "bfloat16"): 46,
    }
    assert Counter(
        (entry["operation"], entry["dtype"]) for entry in mapper["exact"]
    ) == expected
    keys = set()
    for entry in mapper["exact"]:
        key = tuple(entry[field] for field in (
            "operation", "dtype", "batch", "m", "n", "k",
        ))
        assert key not in keys
        keys.add(key)
        selection = select_gemm_schedules(
            entry["m"], entry["n"], entry["k"],
            operation=entry["operation"], dtype=entry["dtype"],
            batch=entry["batch"],
        )
        assert selection.leaf.startswith("exact:")
        assert len(selection.seeds) == 1
        seed = selection.seeds[0]
        assert seed.captured_config == entry["seed"]["configuration"]
        assert seed.builder_config is not None
        assert seed.source_percentage_of_torch == (
            entry["seed"]["source_percentage_of_torch"]
        )
    assert mapper["source_shape_count"] == len(keys) == 1180


def test_overlapping_fp16_and_bf16_shapes_keep_independent_winners():
    fp16 = select_gemm_schedules(128, 4096, 4096, dtype="float16")
    bf16 = select_gemm_schedules(128, 4096, 4096, dtype="bfloat16")

    assert fp16.seeds[0].source_shape == bf16.seeds[0].source_shape
    assert fp16.seeds[0].captured_config != bf16.seeds[0].captured_config
    assert fp16.seeds[0].captured_config["name"] == "heuristic_exact_winner"
    assert bf16.seeds[0].captured_config["name"] == "heuristic_bf16_exact_winner"


def test_bf16_active_wgmma_controls_are_preserved_in_builder_mapping():
    seed = select_gemm_schedules(
        30522, 128, 32768, dtype="bfloat16",
    ).seeds[0]

    assert seed.builder_config is not None
    assert seed.builder_config["wgmma_group_size"] == 4
    assert seed.builder_config["wgmma_pending_groups"] == 1
    seed = select_gemm_schedules(
        25216, 768, 192, dtype="bfloat16",
    ).seeds[0]
    assert seed.builder_config is not None
    assert seed.builder_config["wgmma_epilogue_pipeline"] is True


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


def test_single_iteration_fixed_grid_requires_no_unroll_passes():
    module = object()
    seed = dataclasses.replace(
        select_gemm_schedules(8192, 768, 768, operation="addmm").seeds[0],
        fixed_output_unroll_factor=1,
        post_builder_passes=(),
    )

    assert apply_gemm_schedule_seed_post_build(module, seed) is module


def test_fixed_grid_unroll_is_rescheduled_after_unrolling(monkeypatch):
    seed = dataclasses.replace(
        select_gemm_schedules(8192, 768, 768, operation="addmm").seeds[0],
        fixed_output_unroll_factor=2,
        post_builder_passes=(),
    )
    calls = []
    loop = type("Loop", (), {"attrs": {}})()
    monkeypatch.setattr(
        "ayla.passes.licm.LICM.run",
        lambda self, module: calls.append("licm") or module,
    )
    monkeypatch.setattr(
        "ayla.base.ir.find_ops_of_type",
        lambda module, op_type: [loop],
    )
    monkeypatch.setattr(
        "ayla.base.ir.find_ancestor_of_type",
        lambda loop, op_type: None,
    )
    monkeypatch.setattr(
        "ayla.passes.unroll_loop.UnrollLoop.run",
        lambda self, module: calls.append("unroll") or module,
    )
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
    assert calls == [
        "licm",
        "licm",
        "unroll",
        "canonicalize",
        "schedule_async",
        "canonicalize",
    ]


def test_rolled_wave_pipeline_is_composed_for_selected_mm(monkeypatch):
    from ayla.passes.pipeline_loop import PipelineLoop

    seed = select_gemm_schedules(4096, 16384, 1024).seeds[0]
    assert seed.pipeline_output_waves is True
    assert seed.fixed_output_unroll_factor is None
    assert seed.builder_config is not None
    assert seed.builder_config["output_grid_fast_axis"] == "n"

    class FakeAsync:
        def __init__(self):
            self.attrs = {"label": "mm_lhs_smem_iter0_async"}

    class FakeAwait:
        pass

    class FakeFor:
        def __init__(self):
            self.attrs = {}
            self.body = type("Body", (), {"ops": [FakeAsync(), FakeAwait()]})()

    loop = FakeFor()
    calls = []
    monkeypatch.setattr("ayla.gpu_dialect.AsyncOp", FakeAsync)
    monkeypatch.setattr("ayla.gpu_dialect.AwaitOp", FakeAwait)
    monkeypatch.setattr("ayla.gpu_dialect.ForOp", FakeFor)
    monkeypatch.setattr(
        "ayla.base.ir.find_ops_of_type", lambda module, op_type: [loop],
    )
    monkeypatch.setattr(
        "ayla.base.ir.find_ancestor_of_type", lambda loop, op_type: None,
    )
    monkeypatch.setattr(
        PipelineLoop,
        "run",
        lambda self, module: calls.append("pipeline_loop") or module,
    )
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
    assert calls == ["pipeline_loop", "schedule_async", "canonicalize"]
    assert loop.attrs["label"] == "heuristic_output_grid"
    assert loop.body.ops[0].attrs["label"] == (
        "heuristic_wave_prologue_async"
    )


def test_compact_round_trip_preserves_exact_dispatch():
    expanded = {
        "version": 2,
        "workload": "gemm",
        "source_shape_count": 1,
        "layout": "canonical_nt",
        "features": ["m"],
        "source_fields": ["m", "n", "k"],
        "fallback": {"name": "fallback"},
        "exact": [{
            "operation": "mm", "dtype": "float16", "batch": 1,
            "m": 128, "n": 256, "k": 512,
            "seed": {
                "configuration": {"name": "winner", "operand_stages": 4},
                "source": {"m": 128, "n": 256, "k": 512},
                "source_percentage_of_torch": 110.0,
            },
        }],
        "groups": {"mm/float16/k128": {
            "leaf": "fallback-leaf",
            "seeds": [{
                "configuration": {"name": "fallback"},
                "source": {"m": 64, "n": 64, "k": 128},
                "source_percentage_of_torch": 100.0,
            }],
        }},
    }

    assert expand_gemm_heuristic(
        collapse_gemm_heuristic(expanded),
    ) == expanded


def test_compact_seeds_omit_dropped_experimental_controls():
    selection = select_gemm_schedules(2048, 768, 768)
    redundant = {
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

    assert addmm.leaf.startswith("exact:addmm:")
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
    selection = select_gemm_schedules(
        512, 512, 64, operation="bmm", batch=72,
    )
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config.get("rhs_l2_promotion") is not None
    )

    assert seed.builder_config is not None
    rhs = seed.builder_config["operand_move_attrs"]["rhs"]
    assert rhs["l2_promotion"] == seed.captured_config["rhs_l2_promotion"]


def test_n_grouped_winner_routes_existing_builder_controls():
    selection = select_gemm_schedules(4096, 4096, 64)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["output_n_tiles_per_cta"] > 1
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 1
    assert seed.builder_config["output_n_tiles_per_cta"] == 4


def test_swapped_grid_winner_routes_existing_builder_controls():
    selection = select_gemm_schedules(
        100480, 128, 128, operation="addmm",
    )
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["swap_output_grid_axes"] is True
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_grid_axes"] == (
        CollectiveAxis.Y, CollectiveAxis.X,
    )


def test_fixed_grid_addmm_winner_is_preserved():
    selection = select_gemm_schedules(
        4096, 2304, 768, operation="addmm",
    )
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config.get("output_cta_count") == 132
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_grid_size"] == 132
    assert seed.builder_config.get("warpgroup_m_tile") is None


def test_group_five_family_routes_existing_builder_controls():
    selection = select_gemm_schedules(30522, 768, 2048)
    seed = next(
        seed for seed in selection.seeds
        if seed.captured_config["output_tiles_per_cta"] == 5
    )

    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tiles_per_cta"] == 5
    assert seed.builder_config["output_n_tiles_per_cta"] == 1


def test_transpose_winner_is_selected_for_the_original_problem():
    selection = select_gemm_schedules(
        196, 384, 768, operation="bmm", batch=64,
    )
    seed = next(seed for seed in selection.seeds if seed.transpose_output)

    assert seed.captured_config["transpose_output"] is True
    assert seed.builder_config is not None
    assert seed.builder_config["output_m_tile"] == 192
    assert seed.builder_config["output_n_tile"] == 200


def test_default_output_row_order_reverses_only_large_non_batched_outputs():
    assert default_reverse_output_rows(12608, 3072)
    assert not default_reverse_output_rows(8192, 768)
    assert not default_reverse_output_rows(
        512, 512, operation="bmm", batch=256,
    )


def test_unmeasured_large_output_reverses_output_rows():
    selection = select_gemm_schedules(20000, 3072, 768)

    seeds = [seed for seed in selection.seeds if seed.builder_config is not None]
    assert seeds
    for seed in seeds:
        rows = "N" if seed.transpose_output else "M"
        assert rows in seed.builder_config["output_reversed_dims"]
        assert "output rows visited last-to-first" in seed.adaptations


def test_caller_can_forbid_reversed_output_rows():
    selection = select_gemm_schedules(
        20000, 3072, 768, reverse_output_rows=False,
    )

    for seed in selection.seeds:
        assert seed.builder_config is not None
        assert "output_reversed_dims" not in seed.builder_config


def test_exact_winner_keeps_its_measured_output_row_order():
    [seed] = select_gemm_schedules(12608, 3072, 768).seeds
    assert seed.builder_config is not None
    assert seed.builder_config["output_reversed_dims"] == ("M",)
    assert "output rows visited last-to-first" not in seed.adaptations

    [forced] = select_gemm_schedules(
        12608, 3072, 768, reverse_output_rows=False,
    ).seeds
    assert forced.builder_config is not None
    assert "output_reversed_dims" not in forced.builder_config
    assert "output rows visited first-to-last" in forced.adaptations


def test_addmm_exact_winner_keeps_its_measured_output_row_order():
    [seed] = select_gemm_schedules(
        100352, 512, 64, operation="addmm",
    ).seeds

    assert seed.builder_config is not None
    assert seed.builder_config["output_reversed_dims"] == ("M",)


def test_bmm_exact_winner_remains_forward_order():
    [seed] = select_gemm_schedules(
        196, 640, 128, operation="bmm", batch=128,
    ).seeds

    assert seed.builder_config is not None
    assert "output_reversed_dims" not in seed.builder_config


def test_measured_output_tile_reverse_is_routed_to_the_builder():
    selection = select_gemm_schedules(12608, 3072, 768)
    captured = {
        **selection.seeds[0].captured_config, "output_tile_reverse": "m",
    }
    builder, _ = _adapt_config(
        captured, 12608, 3072, 768, preserve_output_grid=True,
    )
    assert builder is not None
    assert builder["output_reversed_dims"] == ("M",)
