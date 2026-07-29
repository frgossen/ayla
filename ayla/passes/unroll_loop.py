"""Unroll loops by a given factor.

Replaces each ForOp with a new ForOp whose step is multiplied by
`unroll_factor`, containing `unroll_factor` copies of the original body.
Each copy's induction variable is offset by `i * original_step`.

When the trip count is not divisible by the factor, the leftover iterations are
emitted as a straight-line tail after the unrolled loop. If the factor exceeds
the trip count the whole loop becomes the tail (no loop remains). Static
loop-local allocations are hoisted so the tail can reuse the first copy's
buffers instead of allocating additional storage.
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

        # TODO: extend to handle multiple blocks.
        assert len(op.body.blocks) == 1, (
            f"Unroll: expected one block in for body, got {len(op.body.blocks)}"
        )

        # Recursively transform nested ops first.
        self.lower_region(op.body)

        lb, step = op.lb, op.step
        num_tail_iterations = op.trip_count % self.unroll_factor
        main_trip_count = op.trip_count - num_tail_iterations
        new_step = step * self.unroll_factor
        new_ub = lb + main_trip_count * step
        extra_attrs = {k: v for k, v in op.attrs.items() if k not in ("lb", "ub", "step")}

        new_iv = Value(name=op.iv.name, type=IndexType())
        main_body_ops: list[Op] = []
        reusable_allocs: dict[Value, Value] = {}
        num_main_copies = self.unroll_factor if main_trip_count else 0
        for i in range(num_main_copies):
            iteration_iv = new_iv
            if i > 0:
                offset_const = IndexConstantOp.build(value=i * step)
                offset_iv = IndexAddOp.build(lhs=new_iv, rhs=offset_const.result())
                main_body_ops.extend([offset_const, offset_iv])
                iteration_iv = offset_iv.result()
            cloned, vmap = _clone_iteration(op, iteration_iv, f"_iter{i}")
            main_body_ops.extend(cloned)
            if i == 0:
                reusable_allocs = _static_alloc_map(op, vmap)

        tail_ops: list[Op] = []
        for i in range(num_tail_iterations):
            tail_iv = IndexConstantOp.build(value=new_ub + i * step)
            tail_ops.append(tail_iv)
            cloned, vmap = _clone_iteration(op, tail_iv.result(), f"_tail{i}", reusable_allocs)
            tail_ops.extend(cloned)
            if not reusable_allocs:
                reusable_allocs = _static_alloc_map(op, vmap)

        result: list[Op] = []
        if main_trip_count:
            # The tail runs after the loop, so it cannot reference allocs scoped
            # to the loop body. Hoist them out (they are loop-invariant static
            # buffers) so the tail reuses the first copy instead of allocating.
            if num_tail_iterations:
                result.extend(o for o in main_body_ops if _is_static_alloc(o))
                main_body_ops = [o for o in main_body_ops if not _is_static_alloc(o)]
            new_body = Region(blocks=[Block(arguments=[new_iv], ops=main_body_ops)])
            result.append(ForOp.build(lb=lb, ub=new_ub, step=new_step, body=new_body, attrs=extra_attrs))
        result.extend(tail_ops)
        return result, []


def _is_static_alloc(op: Op) -> bool:
    """A loop-invariant allocation (no per-iteration initializer)."""
    return isinstance(op, AllocOp) and op.init is None


def _static_alloc_map(op: ForOp, vmap: dict[Value, Value]) -> dict[Value, Value]:
    """Map each original static alloc to its clone recorded in `vmap`."""
    return {
        body_op.result(): vmap[body_op.result()]
        for body_op in op.body.entry_block.ops
        if _is_static_alloc(body_op)
    }


def _clone_iteration(
    op: ForOp,
    iteration_iv: Value,
    label_suffix: str,
    reusable_allocs: dict[Value, Value] | None = None,
) -> tuple[list[Op], dict[Value, Value]]:
    """Clone the loop body once, rebinding the induction variable.

    Allocations listed in `reusable_allocs` are skipped and remapped to the
    already-emitted buffer instead of being re-allocated.
    """
    vmap: dict[Value, Value] = dict(reusable_allocs or {})
    vmap[op.iv] = iteration_iv
    cloned = [
        clone_op(body_op, vmap)
        for body_op in op.body.entry_block.ops
        if not (reusable_allocs and _is_static_alloc(body_op))
    ]
    _suffix_labels(cloned, label_suffix)
    return cloned, vmap


def _suffix_labels(ops: list[Op], suffix: str) -> None:
    for op in ops:
        for nested in iter_ops(op):
            label = nested.attrs.get("label")
            if isinstance(label, str):
                nested.attrs["label"] = label + suffix


def unroll_loop(module: ModuleOp, unroll_factor: int, label_filter: str | None = None) -> ModuleOp:
    """Convenience entry point."""
    return UnrollLoop(unroll_factor, label_filter).run(module)
