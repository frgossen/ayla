"""Utilities shared by IR transformation passes."""

from __future__ import annotations

import re
from typing import Callable, cast

from ..base.ir import Op, Value, clone_op, clone_type, find_ancestor_of_type
from ..gpu_dialect import (
    AddrMemTensorType, AllocOp, CollectiveAxis, CollectiveLevel, KernelOp, MemTensorType, MoveOp, ReduceOp, TileOp, memory_effects, is_side_effect_free, effective_operands, has_ordering_effect,
)


def effect_roots(op: Op) -> tuple[set[Value], set[Value]] | None:
    """The buffers ``op`` reads and writes, or None if unknown."""
    # An await resolves to its lock peers via memory_effects.
    effects = memory_effects(op)
    if effects is None:
        return None
    reads = {
        root for effect in effects if effect.kind == "R"
        for root in effect.roots
    }
    writes = {
        root for effect in effects if effect.kind == "W"
        for root in effect.roots
    }
    return reads, writes


def can_reorder(
    upper: Op, lower: Op,
    roots: Callable[[Op], tuple[set[Value], set[Value]] | None] = effect_roots,
) -> bool:
    """Whether an adjacent swap preserves SSA, memory dependencies, and fences."""
    if has_ordering_effect(upper) or has_ordering_effect(lower):
        return False

    # SSA dependency: lower consumes a result of upper.
    upper_results = set(upper.results)
    if any(v in upper_results for v in effective_operands(lower)):
        return False

    upper_roots = roots(upper)
    lower_roots = roots(lower)
    if upper_roots is None or lower_roots is None:
        return False

    # RAW, WAR, or WAW conflict on the same buffer.
    upper_r, upper_w = upper_roots
    lower_r, lower_w = lower_roots
    if lower_w & upper_r or lower_r & upper_w or lower_w & upper_w:
        return False

    return True


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


def fresh_alloc(typ, anchor, attrs=None):
    """Allocate without colliding with explicit IDs in parsed input IR."""
    from ..base.ir import Block, Op, find_ops_of_type, root_ancestor
    from ..gpu_dialect import AllocOp

    root = root_ancestor(anchor)
    roots = root.ops if isinstance(root, Block) else [root]
    used = {
        op.attrs["id"]
        for node in roots
        if isinstance(node, Op)
        for op in find_ops_of_type(node, AllocOp)
    }
    allocation_id = AllocOp.fresh_id()
    while allocation_id in used:
        allocation_id = AllocOp.fresh_id()
    return AllocOp.build(typ, allocation_id, attrs=attrs)


def select_region(ops: list[Op], start: int, label_filter: str | None) -> list[Op]:
    """Select a closed, explicitly labelled region of tensor/scalar operations.

    Selection depends on memory effects and SSA closure, not composite
    function names. Control flow and synchronization require scheduling of
    their own and are not implicitly moved with a straight-line region.
    """
    if label_filter is None:
        return []
    selected = []
    for op in ops[start:]:
        if not label_matches(op, label_filter):
            break
        effects = memory_effects(op)
        if op.blocks or effects is None or (not is_side_effect_free(op) and (not effects or any(
            not isinstance(effect.value.type, MemTensorType) for effect in effects
        ))):
            raise ValueError(f'operation requires explicit scheduling outside the tensor region: {type(op).__name__}')
        selected.append(op)
    members = set(selected)
    kernel = find_ancestor_of_type(ops[start - 1], KernelOp)
    for op in selected:
        if any(user not in members and find_ancestor_of_type(user, KernelOp) is kernel
               for result in op.results for user in result.users()):
            raise ValueError('consumer region has an escaping result; include its users in the region')
    return selected


def tile_region(ops: list[Op], anchor: Value, replacements: dict[Value, Value],
                  tiles: list[tuple[int, int, Value]],
                  collective: dict[CollectiveAxis, CollectiveLevel]) -> list[Op]:
    """Tile parallel tensor axes, retaining singleton broadcast dimensions."""
    shape = cast(MemTensorType, anchor.type).static_shape
    mapping = dict(replacements)
    result: list[Op] = []

    def tiled_type(typ: MemTensorType, allocation=False):
        sizes = list(typ.static_shape)
        if not sizes:
            return typ
        if len(sizes) > len(shape):
            raise ValueError('consumer tensor rank cannot exceed producer rank')
        for axis, size, _ in tiles:
            # Reduction result extents are independent of the producer's
            # reduction axis, including partial outputs wider than one.
            if axis < typ.rank and typ.sizes[axis] == shape[axis] and typ.sizes[axis] != 1:
                sizes[axis] = size
        attrs: dict[str, object] = {'sizes': sizes}
        if allocation and isinstance(typ, AddrMemTensorType):
            attrs['strides'] = AddrMemTensorType.canonical_strides(sizes)
        return clone_type(typ, attrs)

    def apply_tiles(value: Value):
        current = value
        value_type = cast(MemTensorType, value.type)
        active_axes = {a for a in range(value_type.rank) if value_type.sizes[a] == shape[a] and shape[a] != 1}
        for axis, size, index in tiles:
            if axis not in active_axes:
                continue
            current_type = cast(MemTensorType, current.type)
            if current_type.sizes[axis] == 1 or current_type.sizes[axis] == size:
                continue
            sizes = list(current_type.sizes)
            sizes[axis] = size
            indices: list[int | Value] = [0] * current_type.rank
            indices[axis] = index
            attrs = {'label': op.attrs['label']} if 'label' in op.attrs else {}
            tile = TileOp.build(current, indices, clone_type(current_type, {'sizes': sizes}), attrs=attrs,
                                maybe_partial=isinstance(current_type, AddrMemTensorType) and cast(int, current_type.sizes[axis]) % size != 0)
            result.append(tile)
            current = tile.result()
        return current

    def operand(value: Value):
        if value in mapping:
            return mapping[value]
        if not isinstance(value.type, MemTensorType) or value.type.rank == 0:
            return value
        typ = tiled_type(value.type)
        if typ == value.type:
            mapping[value] = value
            return value
        current = apply_tiles(value)
        mapping[value] = current
        return current

    for op in ops:
        if isinstance(op, TileOp):
            # Preserve an outer view's indices before adding the new inner
            # partition. Its root can be larger than the current producer.
            remapped_arg = op.arg in mapping and mapping[op.arg].type != op.arg.type
            cloned = clone_op(op, mapping)
            result.append(cloned)
            if remapped_arg:
                parent = cast(MemTensorType, op.arg.type)
                remapped = cast(MemTensorType, mapping[op.arg].type)
                typ = cast(MemTensorType, op.result().type)
                sizes = list(typ.static_shape)
                # Only inherit axes already partitioned in the parent. An
                # existing view on another axis still needs its original size
                # to scale its index before adding the inner partition.
                for axis, (before, after) in enumerate(zip(
                    parent.static_shape, remapped.static_shape, strict=True
                )):
                    if before != after and sizes[axis] == before:
                        sizes[axis] = after
                cloned.result().type = clone_type(typ, {'sizes': sizes})
            mapping[op.result()] = apply_tiles(cloned.result())
            continue
        if isinstance(op, ReduceOp):
            src = cast(MemTensorType, op.src.type)
            dst = cast(MemTensorType, op.dst.type)
            axis = src.rank - 1
            if any(a == axis and size < cast(int, src.sizes[-1]) for a, size, _ in tiles) and src.sizes[-1] != dst.sizes[-1]:
                raise ValueError('consumer reduction spans producer tiles; tile the reduction before fusing')
        for value in op.operands:
            operand(value)
        attrs: dict[str, object] = {}
        if isinstance(op, (ReduceOp, MoveOp)):
            attrs['collective'] = dict(collective)
        if isinstance(op, AllocOp):
            attrs['id'] = AllocOp.fresh_id()
        cloned = clone_op(op, mapping, attrs)
        for original, replacement in zip(op.results, cloned.results, strict=True):
            if isinstance(original.type, MemTensorType):
                replacement.type = tiled_type(original.type, isinstance(op, AllocOp))
            mapping[original] = replacement
        result.append(cloned)
    return result
