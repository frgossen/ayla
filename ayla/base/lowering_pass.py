"""Base class for IR-to-IR lowering passes.

Subclasses register per-op rewriters via `self.register(OpType, fn)` in
their `__init__`. Each rewriter has signature::

    def fn(self, op) -> tuple[list[Op], list[pytree]]

returning:

- ``new_ops: list[Op]`` — ops to splice into the parent block in place
  of the original op.
- ``new_results: list[pytree]`` — one pytree per ``op.results[i]``.
  Pytree leaves are SSA Values; intermediate nodes are Python lists/
  tuples; a single Value is a 1-leaf pytree.

The walker threads ``self.value_map: dict[Value, pytree]`` through the
walk, recording each result's replacement so downstream rewriters can
look up their operands. Ops without a registered rewriter pass through:
the walker recurses into their regions and identity-maps their results.
"""
from __future__ import annotations

from typing import Callable

from .ir import Block, Op, Region, Value
from ..gpu_dialect import ModuleOp
from .pipeline import Pass


class LoweringPass(Pass):
    """Walks the IR top-down, dispatching ops via a registry."""

    def __init__(self) -> None:
        self.value_map: dict[Value, object] = {}
        self._rewriters: dict[
            type,
            Callable[["LoweringPass", Op], tuple[list[Op], list[object]]],
        ] = {}

    def register(
        self,
        op_type: type,
        fn: Callable[
            ["LoweringPass", Op], tuple[list[Op], list[object]]
        ],
    ) -> None:
        """Register a rewriter for ops of `op_type`."""
        assert op_type not in self._rewriters, (
            f"{type(self).__name__}: rewriter for {op_type.__name__} "
            "already registered"
        )
        self._rewriters[op_type] = fn

    def run(self, module: ModuleOp) -> ModuleOp:
        new_ops, new_results = self.lower_op(module)
        [new_module] = new_ops
        assert isinstance(new_module, ModuleOp), (
            f"{type(self).__name__}: expected a ModuleOp, got "
            f"{type(new_module).__name__}"
        )
        assert not new_results, (
            f"{type(self).__name__}: module has no results"
        )
        return new_module

    def lower_op(
        self, op: Op,
    ) -> tuple[list[Op], list[object]]:
        """Dispatch via the registry. Falls through to pass-through
        (rewrite operands, recurse into regions, identity-map results)
        when no rewriter is registered for `type(op)`.

        Pass-through assumes every operand's replacement in `value_map`
        is a single `Value` (not a decomposed pytree). Ops whose
        operands get split into multiple values must register a custom
        lowering that knows how to consume the pytree.
        """
        fn = self._rewriters.get(type(op))
        if fn is not None:
            result = fn(self, op)
            if result is not None:
                return result
        new_operands: list[Value] = []
        for v in op.operands:
            replacement = self.value_map[v]
            assert isinstance(replacement, Value), (
                f"{op.op_id}: operand was lowered to a non-Value "
                f"pytree ({replacement!r}); register a custom lowering"
            )
            new_operands.append(replacement)
        op.operands = new_operands
        for r in op.regions:
            self.lower_region(r)
        return [op], list(op.results)

    def lower_region(self, region: Region) -> None:
        for block in region.blocks:
            self.lower_block(block)

    def lower_block(self, block: Block) -> None:
        # Block arguments are unchanged; register identity mappings so
        # downstream lookups succeed.
        for v in block.arguments:
            self.value_map[v] = v
        new_block_ops: list[Op] = []
        for op in block.ops:
            new_ops, new_results = self.lower_op(op)
            new_block_ops.extend(new_ops)
            assert len(new_results) == len(op.results), (
                f"{op.op_id}: rewriter returned {len(new_results)} "
                f"pytrees for {len(op.results)} results"
            )
            for result, new_result in zip(op.results, new_results):
                self.value_map[result] = new_result
        block.ops = new_block_ops
