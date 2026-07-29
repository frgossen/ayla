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
        num_tail_iterations = op.trip_count % self.unroll_factor

        # TODO: extend to handle multiple blocks.
        assert len(op.body.blocks) == 1, (
            f"Unroll: expected one block in for body, got {len(op.body.blocks)}"
        )

        # Recursively transform nested ops first.
        self.lower_region(op.body)

        main_trip_count = op.trip_count - num_tail_iterations
        new_step = step * self.unroll_factor

        new_iv = Value(name=op.iv.name, type=IndexType())
        new_entry_block_ops: list[Op] = []

        reusable_allocs: dict[Value, Value] = {}
        if main_trip_count:
            for i in range(self.unroll_factor):
                iteration_iv = new_iv
                if i > 0:
                    offset_const = IndexConstantOp.build(value=i * step)
                    offset_iv = IndexAddOp.build(lhs=new_iv, rhs=offset_const.result())
                    new_entry_block_ops.extend([offset_const, offset_iv])
                    iteration_iv = offset_iv.result()

                cloned, iteration_vmap = _clone_iteration(
                    op, iteration_iv, f"_iter{i}",
                )
                new_entry_block_ops.extend(cloned)
                if i == 0:
                    reusable_allocs = _uninitialized_alloc_map(
                        op, iteration_vmap,
                    )

        extra_attrs = {k: v for k, v in op.attrs.items() if k not in ("lb", "ub", "step")}
        new_ub = lb + main_trip_count * step
        new_for: ForOp | None = None
        if main_trip_count:
            new_body = Region(blocks=[Block(arguments=[new_iv], ops=new_entry_block_ops)])
            new_for = ForOp.build(
                lb=lb, ub=new_ub, step=new_step, body=new_body,
                attrs=extra_attrs,
            )

        # The tail runs after the loop, so it cannot reference allocs scoped to
        # the loop body. Hoist them out (they are loop-invariant static buffers)
        # so the remainder can reuse the first copy instead of allocating
        # additional storage.
        hoisted: list[Op] = []
        if num_tail_iterations and new_for is not None:
            hoisted = [
                o for o in new_entry_block_ops
                if isinstance(o, AllocOp) and o.init is None
            ]
            new_for.body.entry_block.ops = [
                o for o in new_entry_block_ops
                if not (isinstance(o, AllocOp) and o.init is None)
            ]

        tail_ops: list[Op] = []
        for i in range(num_tail_iterations):
            tail_iv = IndexConstantOp.build(new_ub + i * step)
            tail_ops.append(tail_iv)
            cloned, iteration_vmap = _clone_iteration(
                op,
                tail_iv.result(),
                f"_tail{i}",
                reusable_allocs=reusable_allocs,
            )
            tail_ops.extend(cloned)
            if not reusable_allocs:
                reusable_allocs = _uninitialized_alloc_map(
                    op, iteration_vmap,
                )

        main_ops = [new_for] if new_for is not None else []
        return hoisted + main_ops + tail_ops, []



def _clone_iteration(
    op: ForOp,
    iteration_iv: Value,
    label_suffix: str,
    reusable_allocs: dict[Value, Value] | None = None,
) -> tuple[list[Op], dict[Value, Value]]:
    vmap = dict(reusable_allocs or {})
    vmap[op.iv] = iteration_iv
    cloned = [
        clone_op(body_op, vmap)
        for body_op in op.body.entry_block.ops
        if not (
            reusable_allocs
            and isinstance(body_op, AllocOp)
            and body_op.init is None
        )
    ]
    _suffix_labels(cloned, label_suffix)
    return cloned, vmap


def _uninitialized_alloc_map(
    op: ForOp,
    vmap: dict[Value, Value],
) -> dict[Value, Value]:
    return {
        body_op.result(): vmap[body_op.result()]
        for body_op in op.body.entry_block.ops
        if isinstance(body_op, AllocOp) and body_op.init is None
    }


def _suffix_labels(ops: list[Op], suffix: str) -> None:
    for op in ops:
        for nested in iter_ops(op):
            label = nested.attrs.get("label")
            if isinstance(label, str):
                nested.attrs["label"] = label + suffix


def unroll_loop(module: ModuleOp, unroll_factor: int, label_filter: str | None = None) -> ModuleOp:
    """Convenience entry point."""
    return UnrollLoop(unroll_factor, label_filter).run(module)
