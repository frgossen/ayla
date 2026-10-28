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


def build_slice_tile(value, dim: int, index: int, size: int):
    from ..base.ir import Value, clone_type
    from ..gpu_dialect import MemTensorType, TileOp

    assert isinstance(value.type, MemTensorType)
    typ = clone_type(value.type)
    assert isinstance(typ, MemTensorType)
    typ.sizes = [
        size if axis == dim else extent for axis, extent in enumerate(typ.sizes)
    ]
    indices: list[Value | int] = [
        index if axis == dim else 0 for axis in range(typ.rank)
    ]
    return TileOp.build(value, indices, typ)
