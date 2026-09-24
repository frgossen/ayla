"""Common subexpression elimination.

Deduplicates ops that compute the same thing. Two ops are equivalent
if they have the same type, same attrs, and their operands are
identical (by identity, after prior CSE replacements).

Walks blocks top-down, building a cache keyed by op structure.
When a duplicate is found, the original's results are reused and the
duplicate is removed.
"""
from __future__ import annotations

from ..gpu_dialect import (
    ModuleOp,
    TileOp,
    is_side_effect_free,
)
from ..base.ir import Block, Op, Type, Value
from ..base.pipeline import Pass


def _op_key(op: Op) -> tuple:
    if isinstance(op, TileOp):
        arg = op.arg
        parent = arg.definition
        while (
            isinstance(parent, TileOp)
            and not parent.extra_attrs
            and not parent.maybe_partial
            and all(index == 0 for index in parent.indices)
        ):
            arg = parent.arg
            parent = arg.definition
        return (
            type(op),
            id(arg),
            tuple(
                index if isinstance(index, int) else id(index)
                for index in op.indices
            ),
            tuple(_type_key(result.type) for result in op.results),
            _attr_key(op.attrs),
        )
    return (
        type(op),
        tuple(id(v) for v in op.operands),
        tuple(_type_key(r.type) for r in op.results),
        _attr_key(op.attrs),
    )


def _type_key(t) -> tuple:
    assert isinstance(t, Type)
    return (t.TYPE_ID, _attr_key(t.attrs))


def _attr_key(v) -> tuple | object:
    if isinstance(v, dict):
        return tuple(
            (k, _attr_key(val)) for k, val in sorted(v.items())
        )
    if isinstance(v, (list, tuple)):
        return tuple(_attr_key(x) for x in v)
    if isinstance(v, Type):
        return _type_key(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    if v is None:
        return None
    raise NotImplementedError(
        f"_attr_key: unsupported attr type {type(v).__name__}"
    )


def _cse_block(block: Block, parent_canonical: dict[int, Value]) -> None:
    canonical_value = dict(parent_canonical)
    cache: dict[tuple, Op] = {}
    new_ops: list[Op] = []

    for op in block.ops:
        op.operands = [
            canonical_value.get(id(v), v) for v in op.operands
        ]

        for inner in op.blocks:
            _cse_block(inner, canonical_value)

        if is_side_effect_free(op):
            key = _op_key(op)
            if key in cache:
                original = cache[key]
                for old_r, new_r in zip(op.results, original.results):
                    canonical_value[id(old_r)] = new_r
                continue
            cache[key] = op
        new_ops.append(op)

    block.ops = new_ops


class CSE(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        for block in module.blocks:
            _cse_block(block, {})
        return module


def cse(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return CSE().run(module)
