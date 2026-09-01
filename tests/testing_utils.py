"""Helpers shared across the test suite.

Deliberately not part of the ayla package: `pyproject.toml` ships `ayla*` and
excludes `tests*`, so anything pytest-flavoured belongs here rather than in
the wheel. Fixtures and hooks go in `conftest.py` instead -- importing from a
conftest is not something pytest supports.
"""
import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"
)


def assert_contains_in_order(text: str, *fragments: str) -> None:
    """Assert that `text` contains each fragment's lines in order.

    Each fragment is a multi-line string (typically a dedented raw string).
    Lines are stripped and matched in sequence — arbitrary lines may appear
    between fragments and between lines within a fragment.
    """
    lines = [l.strip() for l in text.splitlines()]
    pos = 0
    for fragment in fragments:
        for expected_line in fragment.strip().splitlines():
            expected_line = expected_line.strip()
            if not expected_line:
                continue
            found = False
            while pos < len(lines):
                if lines[pos] == expected_line:
                    found = True
                    pos += 1
                    break
                pos += 1
            assert found, (
                f"expected line not found: {expected_line!r}\n"
                f"searched from position {pos} in:\n{text}"
            )
