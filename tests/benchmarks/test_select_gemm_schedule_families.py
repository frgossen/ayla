import pytest

from benchmarks.select_gemm_schedule_families import (
    DEFAULT_MAX_CANDIDATES,
    family,
    select_families,
    table_entry,
    update_table,
)


def _config(**overrides: object) -> dict[str, object]:
    return {
        "output_tiles_per_cta": 1, "output_m_tile": 128, "output_n_tile": 256,
        "warpgroup_n_tile": 256, "operand_stages": 3, "operand_k_tile": 64,
        "output_slices": 1, "output_store_buffers": 1,
        "output_tma_inner_size": 128, "output_store": "stmatrix",
        "lhs_l2_cache_hint": "evict_first", "rhs_l2_cache_hint": "evict_last",
        "dst_l2_cache_hint": "evict_first",
    } | overrides


def test_tuning_knobs_stay_within_a_family() -> None:
    base = family(_config())
    for knob in (
        {"lhs_l2_cache_hint": "evict_last"}, {"rhs_l2_promotion": "256B"},
        {"operand_stages": 5}, {"output_slices": 2},
        {"ptx_optimization_level": 2}, {"reschedule_async": True},
        {"swap_output_grid_axes": True},
    ):
        assert family(_config(**knob)) == base, knob
    assert family(_config(output_cta_count=132)) == family(_config(output_cta_count=128))


@pytest.mark.parametrize("change", [
    {"output_m_tile": 192}, {"output_n_tile": 128, "warpgroup_n_tile": 128},
    {"operand_k_tile": 128}, {"transpose_output": True},
    {"output_tiles_per_cta": 2}, {"output_cta_count": 132},
    {"multicast": True, "cluster_n_ctas": 2},
])
def test_geometry_starts_a_new_family(change: dict[str, object]) -> None:
    assert family(_config(**change)) != family(_config())


def test_families_keep_the_best_variant_within_the_band() -> None:
    ranked = [
        (110.0, _config(operand_stages=4)),
        (109.0, _config(operand_stages=5)),  # same family as the best
        (108.0, _config(output_m_tile=192)),
        (106.0, _config(output_n_tile=128, warpgroup_n_tile=128)),
        (100.0, _config(operand_k_tile=128)),  # below 95% of 110
    ]
    seeds = select_families([], ranked, families=10, band=0.95)
    assert seeds == [ranked[0][1], ranked[2][1], ranked[3][1]]
    assert select_families([], ranked, families=2, band=0.95) == seeds[:2]


def test_installed_seeds_come_first_and_claim_their_family() -> None:
    installed = [_config(operand_stages=2)]
    ranked = [(110.0, _config(operand_stages=4)), (108.0, _config(output_m_tile=192))]
    assert select_families(installed, ranked, families=10, band=0.95) == [
        installed[0], ranked[1][1],
    ]


def test_table_entries_omit_defaults_and_reject_unbuildable_configs() -> None:
    entry = table_entry(_config(name="x", multicast=False, wgmma_group_size=None))
    assert entry == dict(sorted(_config().items()))
    assert table_entry(_config(wgmma_epilogue_pipeline=True)) is None
    assert table_entry(_config(split_k_parts=2)) is None
    assert table_entry(
        _config(output_cta_count=132, multicast=True, cluster_m_ctas=2),
    ) is None


def test_fifteen_candidate_limit_counts_installed_variants() -> None:
    installed = [_config(operand_stages=2), _config(operand_stages=4)]
    ranked = [(110.0, _config(output_m_tile=128 + 32 * i)) for i in range(1, 20)]
    seeds = select_families(installed, ranked, families=15, band=0.95)
    assert DEFAULT_MAX_CANDIDATES == 15
    assert seeds == installed + [config for _, config in ranked[:13]]
    with pytest.raises(ValueError, match="exceeds the candidate limit"):
        select_families(seeds, ranked, families=15, band=0.95, candidates=14)


def test_update_preserves_admission_and_graph_metadata(tmp_path, monkeypatch) -> None:
    from ayla import gemm_schedule_table
    from benchmarks import select_gemm_schedule_families as selector

    config = table_entry(_config())
    shape = (1, 128, 256, 256)
    group = ("mm", "bfloat16")
    suffix = '\nDEFERRED_OUTPUT_WIDTHS = frozenset({2})\nfrom .graph_gemm_schedules import GRAPH_SCHEDULES\n'
    text = selector._format_table('"""test"""\n', [config], {group: {shape: (0,)}})
    path = tmp_path / "table.py"
    path.write_text(text + suffix)
    monkeypatch.setattr(selector, "TABLE", path)
    monkeypatch.setattr(gemm_schedule_table, "CONFIGS", (config,))
    monkeypatch.setattr(gemm_schedule_table, "SCHEDULES", {group: {shape: (0,)}})
    assert update_table({(*group, *shape): [(110.0, table_entry(_config(output_m_tile=192)))]},
                        families=15, band=0.95) == (1, 1)
    assert path.read_text().endswith(suffix)
