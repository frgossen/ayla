"""Loop-invariant code motion.

Hoists side-effect-free ops out of a loop body -- ``for`` or ``ploop`` --
when all their operands are defined outside the loop. An op that does not
depend on the induction variable computes the same value at every point of
the iteration space, sequential or parallel alike.
"""
from __future__ import annotations

from ..gpu_dialect import (
    ForOp,
    ModuleOp,
    PLoopOp,
    effective_operands,
    is_side_effect_free,
)
from ..base.ir import Op, Value
from ..base.lowering_pass import LoweringPass, lowers


class LICM(LoweringPass):

    @lowers(ForOp, PLoopOp)
    def lower_block_op(
        self, op: Op,
    ) -> tuple[list[Op], list[object]]:
        body = op.body
        # Defined inside the loop: block arguments (e.g. the iv) plus the
        # results of the body ops we keep.
        body_defined: set[Value] = set(body.arguments)
        hoisted: list[Op] = []
        remaining: list[Op] = []
        for body_op in body.ops:
            if _can_hoist(body_op, body_defined):
                hoisted.append(body_op)
            else:
                body_defined.update(body_op.results)
                remaining.append(body_op)
        body.ops = remaining
        for h in hoisted:
            for r in h.results:
                self.value_map[r] = r
        self.lower_block(op.body)
        return hoisted + [op], list(op.results)


def _can_hoist(op: Op, body_defined: set[Value]) -> bool:
    if any(v in body_defined for v in effective_operands(op)):
        return False
    return is_side_effect_free(op)


def licm(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LICM().run(module)
