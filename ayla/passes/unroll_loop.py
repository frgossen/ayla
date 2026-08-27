"""Unroll loops by a given factor.

Replaces each ForOp with a new ForOp whose step is multiplied by
`unroll_factor`, containing `unroll_factor` copies of the original body.
Each copy's induction variable is offset by `i * original_step`.

When the trip count is not divisible by the factor, the leftover iterations are
emitted as a straight-line tail after the unrolled loop. If the factor exceeds
the trip count the whole loop becomes the tail (no loop remains).

Every copy is a plain clone of the body. Multi-buffering a loop is
``multi_buffer``'s job, so run LICM first to lift the loop's allocations out --
otherwise each copy gets its own.
"""
from __future__ import annotations

import re

from ..gpu_dialect import (
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexType,
    ModuleOp,
)
from ..base.ir import Block, Op, Value, clone_op, iter_ops
from ..base.lowering_pass import LoweringPass, lowers


class UnrollLoop(LoweringPass):
    """Unroll all ForOps by `unroll_factor`."""

    def __init__(self, unroll_factor: int, label_filter: str | None = None) -> None:
        super().__init__()
        self.unroll_factor = unroll_factor
        self.label_filter = label_filter

    @lowers(ForOp)
    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        if self.unroll_factor <= 1:
            return None
        if self.label_filter is not None and not re.fullmatch(self.label_filter, op.attrs.get("label", "")):
            return None

        # Recursively transform nested ops first.
        self.lower_block(op.body)

        lb, step = op.lb, op.step
        num_tail_iterations = op.trip_count % self.unroll_factor
        main_trip_count = op.trip_count - num_tail_iterations
        new_step = step * self.unroll_factor
        new_ub = lb + main_trip_count * step
        extra_attrs = {k: v for k, v in op.attrs.items() if k not in ("lb", "ub", "step")}

        new_iv = Value(name=op.iv.name, type=IndexType())
        main_body_ops: list[Op] = []
        for i in range(self.unroll_factor):
            offset = IndexConstantOp.build(value=i * step)
            iteration_iv = IndexAddOp.build(lhs=new_iv, rhs=offset.result())
            main_body_ops.extend([offset, iteration_iv])
            main_body_ops.extend(
                _clone_iteration(op, iteration_iv.result(), f"_iter{i}")
            )

        tail_ops: list[Op] = []
        for i in range(num_tail_iterations):
            tail_iv = IndexConstantOp.build(value=new_ub + i * step)
            tail_ops.append(tail_iv)
            tail_ops.extend(_clone_iteration(op, tail_iv.result(), f"_tail{i}"))

        # An empty main loop is left for folding to erase rather than special
        # cased here.
        new_body = Block(arguments=[new_iv], ops=main_body_ops)
        result: list[Op] = [
            ForOp.build(lb=lb, ub=new_ub, step=new_step, body=new_body, attrs=extra_attrs)
        ]
        result.extend(tail_ops)
        return result, []


def _clone_iteration(
    op: ForOp, iteration_iv: Value, label_suffix: str,
) -> list[Op]:
    """Clone the loop body once, rebinding the induction variable.

    Labels get the suffix so that later passes can name a single copy.
    """
    vmap: dict[Value, Value] = {op.iv: iteration_iv}
    cloned = [clone_op(body_op, vmap) for body_op in op.body.ops]
    for cloned_op in cloned:
        for nested in iter_ops(cloned_op):
            label = nested.attrs.get("label")
            if isinstance(label, str):
                nested.attrs["label"] = label + label_suffix
    return cloned


def unroll_loop(module: ModuleOp, unroll_factor: int, label_filter: str | None = None) -> ModuleOp:
    """Convenience entry point."""
    return UnrollLoop(unroll_factor, label_filter).run(module)
