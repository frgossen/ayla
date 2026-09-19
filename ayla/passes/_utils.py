"""Utilities shared by IR transformation passes."""
from __future__ import annotations

import re

from ..base.ir import Block, Op, Value, clone_type
from ..gpu_dialect import (
    AllocOp,
    BlockIdsOp,
    CollectiveLevel,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    LaneIdOp,
    RmemTensorType,
    SmemTensorType,
    ThreadIdsOp,
    TileOp,
    WarpGroupIdOp,
    WrmemTensorType,
)


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


def value_collective_level(value: Value) -> CollectiveLevel:
    """Return the narrowest hardware level across which ``value`` varies."""
    defn = value.definition
    if isinstance(defn, (Block, BlockIdsOp, IndexConstantOp)):
        return CollectiveLevel.BLOCK
    if isinstance(defn, WarpGroupIdOp):
        return CollectiveLevel.WG
    if isinstance(defn, (ThreadIdsOp, LaneIdOp)):
        return CollectiveLevel.THREAD
    if isinstance(defn, AllocOp):
        typ = value.type
        if isinstance(typ, SmemTensorType):
            return CollectiveLevel.BLOCK
        if isinstance(typ, WrmemTensorType):
            return CollectiveLevel.WG
        if isinstance(typ, (RmemTensorType, GmemTensorType)):
            return CollectiveLevel.THREAD
    if isinstance(defn, TileOp):
        level = value_collective_level(defn.arg)
        for index in defn.indices:
            if isinstance(index, Value):
                level = min(level, value_collective_level(index))
        return level
    if isinstance(defn, (IndexAddOp, IndexMulOp, IndexDivOp, IndexModOp)):
        level = CollectiveLevel.BLOCK
        for operand in defn.operands:
            level = min(level, value_collective_level(operand))
        return level
    return CollectiveLevel.BLOCK


def tile_introduces_narrow_level(
    tile: TileOp, target_level: CollectiveLevel,
) -> bool:
    return any(
        isinstance(index, Value)
        and value_collective_level(index) < target_level
        for index in tile.indices
    )


def rebuild_tile(tile: TileOp, new_arg: Value) -> TileOp:
    """Rebuild ``tile`` on ``new_arg``, preserving dimensions it narrows."""
    tile_type = tile.result().type
    arg_type = tile.arg.type
    new_tile_type = clone_type(new_arg.type)
    for index in range(len(new_tile_type.sizes)):
        if tile_type.sizes[index] != arg_type.sizes[index]:
            new_tile_type.sizes[index] = tile_type.sizes[index]
    return TileOp.build(
        new_arg, list(tile.indices), new_tile_type,
        maybe_partial=tile.maybe_partial,
    )


def build_slice_tile(value: Value, dim: int, index: int, size: int) -> TileOp:
    typ = clone_type(value.type)
    dim %= typ.rank
    typ.sizes[dim] = size
    indices = [0] * typ.rank
    indices[dim] = index
    return TileOp.build(value, indices, typ)
