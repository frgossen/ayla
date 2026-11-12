import pytest

from benchmarks.merge_gemm_schedule_tables import merge_tables, replace_tables


def _table(configs, shapes):
    return dict(CONFIGS=configs, SCHEDULES={("mm", "float16"): shapes})


def test_merge_preserves_winner_and_deduplicates_by_config():
    shape = (1, 128, 256, 256)
    other = (1, 256, 256, 256)
    target = _table([dict(tile=128), dict(tile=256)], {shape: (1, 0)})
    source = _table([dict(tile=128), dict(tile=192), dict(tile=256)], {shape: (0, 1, 2), other: (1,)})
    configs, schedules = merge_tables(target, source)
    assert configs == [dict(tile=128), dict(tile=256), dict(tile=192)]
    assert schedules["mm", "float16"] == {shape: (1, 0, 2), other: (2,)}
    assert target["SCHEDULES"]["mm", "float16"][shape] == (1, 0)
    assert merge_tables(dict(CONFIGS=configs, SCHEDULES=schedules), source) == (configs, schedules)


def test_merge_accepts_fifteen_and_rejects_overflow_without_truncation():
    shape = (1, 128, 256, 256)
    configs = [dict(tile=i) for i in range(16)]
    target = _table(configs[:10], {shape: tuple(range(10))})
    source = _table(configs[:15], {shape: tuple(range(15))})
    merged_configs, schedules = merge_tables(target, source)
    assert len(merged_configs) == len(schedules["mm", "float16"][shape]) == 15
    with pytest.raises(ValueError, match="16 candidates exceed limit 15"):
        merge_tables(target, _table(configs, {shape: tuple(range(16))}))
    assert len(target["CONFIGS"]) == 10


def test_table_replacement_preserves_non_candidate_metadata():
    text = ('"""metadata"""\n# fmt: off\nCONFIGS: tuple = ()\n'
            'SCHEDULES: dict = {}\n\nDEFERRED_OUTPUT_WIDTHS = frozenset({2})\n'
            'from .graph_gemm_schedules import GRAPH_SCHEDULES\n')
    suffix = text[text.index('\nDEFERRED_OUTPUT_WIDTHS'):]
    updated = replace_tables(text, [dict(tile=128)], {("mm", "float16"): {(1, 128, 256, 256): (0,)}})
    assert updated.startswith('"""metadata"""\n# fmt: off\n')
    assert updated.endswith(suffix)
    assert 'dict(tile=128)' in updated
