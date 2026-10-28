from benchmarks.select_oss_gemm_replicated import (
    _has_distinct_configuration,
    _representative_incumbent,
)


def test_candidate_must_change_more_than_its_name() -> None:
    original = {"configuration": {"name": "base", "stages": 4}}
    renamed = {"configuration": {"name": "candidate", "stages": 4}}
    changed = {"configuration": {"name": "candidate", "stages": 5}}

    assert not _has_distinct_configuration(original, renamed)
    assert _has_distinct_configuration(original, changed)


def test_incumbent_representative_uses_original_median() -> None:
    rows = [
        {
            "original": {"torch_pct": pct, "name": name},
            "original_torch": {"name": f"torch-{name}"},
            # Deliberately make the candidate-speedup ordering disagree.
            "pct_speedup": speedup,
        }
        for pct, speedup, name in [
            (70.0, 1.01, "stale"),
            (101.5, 1.20, "median"),
            (102.0, 1.10, "fast"),
        ]
    ]

    representative = _representative_incumbent(rows)

    assert representative["original"]["name"] == "median"
    assert representative["original_torch"]["name"] == "torch-median"
