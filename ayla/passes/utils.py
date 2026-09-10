"""Utilities shared by GPU dialect transformation passes."""

from __future__ import annotations

import re

from ..base.ir import Op


def label_matches(
    op: Op, pattern: str | re.Pattern[str] | None,
) -> bool:
    """Whether an op's string label fully matches ``pattern``.

    ``None`` selects every op. A missing label is treated as an empty string,
    so a pattern such as ``.*`` also selects unlabeled ops. Non-string labels
    never match.
    """
    if pattern is None:
        return True
    label = op.attrs.get("label", "")
    return isinstance(label, str) and re.fullmatch(pattern, label) is not None
