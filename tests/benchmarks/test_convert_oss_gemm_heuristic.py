from copy import deepcopy

import pytest

from benchmarks.convert_oss_gemm_heuristic import _merge_mapper


def _fixtures():
    expanded = {
        "version": 2, "workload": "gemm", "layout": "canonical_nt",
        "source_fields": ["m", "n", "k"], "features": ["m", "n", "k"],
        "source_shape_count": 1,
        "exact": [{
            "operation": "mm", "dtype": "float16", "batch": 1,
            "m": 128, "n": 128, "k": 128,
            "seed": {"configuration": {"name": "fp16-winner"}},
        }],
        "groups": {"mm/float16/k128": {"leaf": "fp16", "seeds": []}},
    }
    mapper = {
        "version": 2, "layout": "canonical_nt",
        "features": ["m", "n", "k"], "source_shape_count": 1,
        "fallback_seeds": [{"configuration": {"name": "fallback"}}],
        "groups": {"mm/bfloat16/k128": {
            "kind": "leaf", "id": "bf16", "seeds": [],
        }},
        "exact": [{
            "shape": {
                "operation": "mm", "dtype": "bfloat16", "batch": 1,
                "m": 128, "n": 128, "k": 128,
            },
            "seed": {
                "configuration": {"name": "bf16-winner"},
                "source_shape": {"m": 128, "n": 128, "k": 128},
                "source_percentage_of_torch": 106.0,
            },
        }],
    }
    return expanded, mapper


def test_merge_retains_exact_winners_for_the_same_shape_in_both_dtypes():
    expanded, mapper = _fixtures()
    old = deepcopy(expanded)

    merged = _merge_mapper(expanded, mapper)

    assert merged["exact"][0] == old["exact"][0]
    assert merged["exact"][1]["dtype"] == "bfloat16"
    assert merged["exact"][1]["seed"]["configuration"]["name"] == "bf16-winner"
    assert merged["groups"]["mm/float16/k128"] == old["groups"]["mm/float16/k128"]
    assert merged["source_shape_count"] == 2


def test_duplicate_exact_winner_is_rejected_without_mutating_the_base():
    expanded, mapper = _fixtures()
    mapper["exact"][0]["shape"]["dtype"] = "float16"
    old = deepcopy(expanded)

    with pytest.raises(ValueError, match="exact GEMM winner already exists"):
        _merge_mapper(expanded, mapper)

    assert expanded == old
