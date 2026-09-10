"""Tests for utilities shared by transformation passes."""
import re

from ayla.base.ir import Op
from ayla.passes.utils import label_matches


def test_label_matches_full_regex_filters():
    op = Op(operands=[], results=[], attrs={"label": "stage_1"})

    assert label_matches(op, None)
    assert label_matches(op, "stage_1")
    assert label_matches(op, r"stage_\d")
    assert label_matches(op, re.compile(r"stage_\d"))
    assert not label_matches(op, r"stage")


def test_label_matches_treats_a_missing_label_as_empty():
    unlabeled = Op(operands=[], results=[])
    assert not label_matches(unlabeled, "stage")
    assert label_matches(unlabeled, "")
    assert label_matches(unlabeled, r".*")
    assert label_matches(unlabeled, r"^$")


def test_label_matches_rejects_an_explicit_non_string_label():
    assert not label_matches(
        Op(operands=[], results=[], attrs={"label": 1}),
        r"\d",
    )
