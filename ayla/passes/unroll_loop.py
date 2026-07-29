"""Unroll loops by a given factor.

Replaces each ForOp with a new ForOp whose step is multiplied by
`unroll_factor`, containing `unroll_factor` copies of the original body.
Each copy's induction variable is offset by `i * original_step`.

When the trip count is not divisible by the factor, emits a scalar remainder
after the unrolled main loop. Loop-local allocations are hoisted and the tail
reuses those from the first unrolled copy, avoiding an extra allocation.
"""
from __future__ import annotations

import re

from ..gpu_dialect import (
    AllocOp,
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
        remainder = op.trip_count % self.unroll_factor
        if self.unroll_factor > op.trip_count:
            raise ValueError(
                f"Unroll: factor ({self.unroll_factor}) exceeds trip count "
                f"({op.trip_count})"
            )

        # TODO: extend to handle multiple blocks.
        assert len(op.body.blocks) == 1, (
            f"Unroll: expected one block in for body, got {len(op.body.blocks)}"
        )

        # Recursively transform nested ops first.
        self.lower_region(op.body)

        main_trip_count = op.trip_count - remainder
        new_step = step * self.unroll_factor

        new_iv = Value(name=op.iv.name, type=IndexType())
        vmap: dict[Value, Value] = {op.iv: new_iv}
        new_entry_block_ops: list[Op] = []

        reusable_allocs: dict[Value, Value] = {}
        for i in range(self.unroll_factor):
            if i > 0:
                offset_const = IndexConstantOp.build(value=i * step)
                offset_iv = IndexAddOp.build(lhs=new_iv, rhs=offset_const.result())
                new_entry_block_ops.extend([offset_const, offset_iv])
                vmap[op.iv] = offset_iv.result()

            cloned = [clone_op(op, vmap) for op in op.body.blocks[0].ops]
            _suffix_labels(cloned, f"_iter{i}")
            new_entry_block_ops.extend(cloned)
            if i == 0:
                reusable_allocs = {
                    orig.result(): vmap[orig.result()]
                    for orig in op.body.entry_block.ops
                    if isinstance(orig, AllocOp) and orig.init is None
                }

        new_body = Region(blocks=[Block(arguments=[new_iv], ops=new_entry_block_ops)])
        extra_attrs = {k: v for k, v in op.attrs.items() if k not in ("lb", "ub", "step")}
        main_ub = lb + main_trip_count * step
        new_for = ForOp.build(lb=lb, ub=main_ub, step=new_step, body=new_body, attrs=extra_attrs)
        if remainder == 0:
            return [new_for], []

        # The tail runs after the loop, so it cannot reference allocs scoped to
        # the loop body. Hoist them out (they are loop-invariant static buffers)
        # so the remainder can reuse the first copy instead of allocating
        # additional storage.
        hoisted = [
            o for o in new_entry_block_ops
            if isinstance(o, AllocOp) and o.init is None
        ]
        new_body.entry_block.ops = [
            o for o in new_entry_block_ops
            if not (isinstance(o, AllocOp) and o.init is None)
        ]
        tail_vmap: dict[Value, Value] = dict(reusable_allocs)

        tail_ops: list[Op] = []
        for i in range(remainder):
            tail_iv = IndexConstantOp.build(main_ub + i * step)
            tail_ops.append(tail_iv)
            tail_vmap[op.iv] = tail_iv.result()
            cloned = [
                clone_op(body_op, tail_vmap)
                for body_op in op.body.entry_block.ops
                if not (isinstance(body_op, AllocOp) and body_op.init is None)
            ]
            _suffix_labels(cloned, f"_tail{i}")
            tail_ops.extend(cloned)
        return hoisted + [new_for] + tail_ops, []



def _suffix_labels(ops: list[Op], suffix: str) -> None:
    for op in ops:
        for nested in iter_ops(op):
            label = nested.attrs.get("label")
            if isinstance(label, str):
                nested.attrs["label"] = label + suffix


def unroll_loop(module: ModuleOp, unroll_factor: int, label_filter: str | None = None) -> ModuleOp:
    """Convenience entry point."""
    return UnrollLoop(unroll_factor, label_filter).run(module)
