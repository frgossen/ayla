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
    is_value_identical,
)
from ..base.ir import Block, Op, Region, Type, Value
from ..base.pipeline import Pass


class CSE(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        for region in module.regions:
            self.run_region(region, {})
        return module

    def run_region(self, region: Region, canonical_value: dict[int, Value]) -> None:
        for block in region.blocks:
            self.run_block(block, canonical_value)

    def run_block(self, block: Block, parent_canonical: dict[int, Value]) -> None:
        canonical_value = dict(parent_canonical)
        cache: dict[tuple, Op] = {}
        new_ops: list[Op] = []

        for op in block.ops:
            op.set_operands(
                [canonical_value.get(id(v), v) for v in op.operands]
            )

            for region in op.regions:
                self.run_region(region, canonical_value)

            if is_value_identical(op):
                key = self._op_key(op)
                if key in cache:
                    original = cache[key]
                    for old_r, new_r in zip(op.results, original.results):
                        canonical_value[id(old_r)] = new_r
                    continue
                cache[key] = op
            new_ops.append(op)

        block.ops = new_ops

    @staticmethod
    def _op_key(op: Op) -> tuple:
        return (
            type(op),
            tuple(id(v) for v in op.operands),
            tuple(CSE._type_key(r.type) for r in op.results),
            CSE._attr_key(op.attrs),
        )

    @staticmethod
    def _type_key(t) -> tuple:
        assert isinstance(t, Type)
        return (t.TYPE_ID, CSE._attr_key(t.attrs))

    @staticmethod
    def _attr_key(v) -> tuple | object:
        if isinstance(v, dict):
            return tuple(
                (k, CSE._attr_key(val)) for k, val in sorted(v.items())
            )
        if isinstance(v, (list, tuple)):
            return tuple(CSE._attr_key(x) for x in v)
        if isinstance(v, Type):
            return CSE._type_key(v)
        if isinstance(v, (int, float, str, bool)):
            return v
        if v is None:
            return None
        raise NotImplementedError(
            f"_attr_key: unsupported attr type {type(v).__name__}"
        )


def cse(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return CSE().run(module)
