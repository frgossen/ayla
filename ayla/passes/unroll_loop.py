"""Unroll loops by a given factor.

Replaces each ForOp with a new ForOp whose step is multiplied by
`unroll_factor`, containing `unroll_factor` copies of the original body.
Each copy's induction variable is offset by `i * original_step`.

Requires that (ub - lb) is divisible by (step * unroll_factor).
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
from ..base.ir import Block, Op, Region, Value, clone_op, iter_ops
from ..base.lowering_pass import LoweringPass


class UnrollLoop(LoweringPass):
    """Unroll all ForOps by `unroll_factor`."""

    def __init__(self, unroll_factor: int, label_filter: str | None = None) -> None:
        super().__init__()
        self.register(ForOp, type(self).lower_for)
        self.unroll_factor = unroll_factor
        self.label_filter = label_filter

    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        if self.unroll_factor <= 1:
            return None
        if self.label_filter is not None and not re.fullmatch(self.label_filter, op.attrs.get("label", "")):
            return None
        lb, ub, step = op.lb, op.ub, op.step
        if op.trip_count % self.unroll_factor != 0:
            raise ValueError(
                f"Unroll: trip count ({op.trip_count}) is not divisible "
                f"by unroll factor ({self.unroll_factor})"
            )

        # TODO: extend to handle multiple blocks.
        assert len(op.body.blocks) == 1, (
            f"Unroll: expected one block in for body, got {len(op.body.blocks)}"
        )

        # Recursively transform nested ops first.
        self.lower_region(op.body)

        new_step = step * self.unroll_factor

        new_iv = Value(name=op.iv.name, type=IndexType())
        vmap: dict[Value, Value] = {op.iv: new_iv}
        new_entry_block_ops: list[Op] = []

        for i in range(self.unroll_factor):
            if i > 0:
                offset_const = IndexConstantOp.build(value=i * step)
                offset_iv = IndexAddOp.build(lhs=new_iv, rhs=offset_const.result())
                new_entry_block_ops.extend([offset_const, offset_iv])
                vmap[op.iv] = offset_iv.result()

            cloned = [clone_op(op, vmap) for op in op.body.blocks[0].ops]
            _suffix_labels(cloned, f"_iter{i}")
            new_entry_block_ops.extend(cloned)

        new_body = Region(blocks=[Block(arguments=[new_iv], ops=new_entry_block_ops)])
        extra_attrs = {k: v for k, v in op.attrs.items() if k not in ("lb", "ub", "step")}
        new_for = ForOp.build(lb=lb, ub=ub, step=new_step, body=new_body, attrs=extra_attrs)
        op.drop_uses()  # original loop + body discarded, replaced by clones
        return [new_for], []



def _suffix_labels(ops: list[Op], suffix: str) -> None:
    for op in ops:
        for nested in iter_ops(op):
            label = nested.attrs.get("label")
            if isinstance(label, str):
                nested.attrs["label"] = label + suffix


def unroll_loop(module: ModuleOp, unroll_factor: int, label_filter: str | None = None) -> ModuleOp:
    """Convenience entry point."""
    return UnrollLoop(unroll_factor, label_filter).run(module)
