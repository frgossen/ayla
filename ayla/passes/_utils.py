"""Utilities shared by IR transformation passes."""
from __future__ import annotations

import re

from ..base.ir import Op


def label_matches(
    op: Op,
    label_filter: str | None,
) -> bool:
    """Whether ``op``'s effective label fully matches ``label_filter``.

    ``None`` selects every op. An op without a label has the effective label
    ``""``; an explicitly non-string label never matches.
    """
    if label_filter is None:
        return True
    label = op.attrs.get("label", "")
    if not isinstance(label, str):
        return False
    return re.fullmatch(label_filter, label) is not None
