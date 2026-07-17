"""Loop-invariant code motion.

Hoists side-effect-free ops out of a region — a ``for`` loop body or an
``async`` body — when all their operands are defined outside that region.

Which region ops to hoist out of is configurable. The default handles both;
the codegen pipeline runs it ``AsyncOp``-only, to lift the index arithmetic
``lower_to_tma_move`` materializes inside an ``async { … }`` back out (restoring
a single-op async body) without disturbing loops — at that point loop bodies
hold per-iteration lock allocs, which are side-effect-free but must not be
hoisted out of their loop.
"""
from __future__ import annotations

from .gpu_dialect import (
    AsyncOp,
    ForOp,
    ModuleOp,
    effective_operands,
    is_side_effect_free,
)
from .ir import Op, Value
from .lowering_pass import LoweringPass


class LICM(LoweringPass):

    def __init__(self, region_ops: tuple[type, ...] = (ForOp, AsyncOp)) -> None:
        super().__init__()
        for op_type in region_ops:
            self.register(op_type, type(self).lower_region_op)

    def lower_region_op(
        self, op: Op,
    ) -> tuple[list[Op], list[object]]:
        assert len(op.body.blocks) == 1
        body = op.body.entry_block
        # Defined inside the region: block arguments (e.g. a loop's iv) plus the
        # results of the body ops we keep.
        region_defined: set[Value] = set(body.arguments)
        hoisted: list[Op] = []
        remaining: list[Op] = []
        for body_op in body.ops:
            if _can_hoist(body_op, region_defined):
                hoisted.append(body_op)
            else:
                region_defined.update(body_op.results)
                remaining.append(body_op)
        body.ops = remaining
        for h in hoisted:
            for r in h.results:
                self.value_map[r] = r
        self.lower_region(op.body)
        return hoisted + [op], list(op.results)


def _can_hoist(op: Op, region_defined: set[Value]) -> bool:
    if any(v in region_defined for v in effective_operands(op)):
        return False
    return is_side_effect_free(op)


def licm(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LICM().run(module)
