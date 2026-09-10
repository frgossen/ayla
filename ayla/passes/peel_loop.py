"""Peel static iterations from the front and back of loops.

Selected iterations become straight-line clones of the loop body with the
induction variable replaced by a constant. The unpeeled iterations remain in
a loop with narrowed bounds::

    for (k = 0 to 8) { body(k) }

with two front iterations and one back iteration becomes::

    body(0)
    body(1)
    for (k = 2 to 7) { body(k) }
    body(7)

Labels in the straight-line copies receive ``_frontN`` or ``_backN`` suffixes
so subsequent passes can specialize individual iterations.
"""
from __future__ import annotations

from ..base.ir import Op
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import ForOp, IndexConstantOp, ModuleOp
from ._loop_utils import clone_loop_iteration
from ._utils import label_matches


class PeelLoop(LoweringPass):
    """Peel matching loops at both ends by fixed iteration counts."""

    def __init__(
        self,
        label_filter: str,
        front_iterations: int = 0,
        back_iterations: int = 0,
    ) -> None:
        super().__init__(
            label_filter=label_filter,
            front_iterations=front_iterations,
            back_iterations=back_iterations,
        )
        if front_iterations < 0:
            raise ValueError("PeelLoop: front_iterations must be non-negative")
        if back_iterations < 0:
            raise ValueError("PeelLoop: back_iterations must be non-negative")

    @lowers(ForOp)
    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not label_matches(op, self.label_filter):
            return None
        if self.front_iterations == 0 and self.back_iterations == 0:
            return None

        trip_count = op.trip_count
        num_peeled = self.front_iterations + self.back_iterations
        if num_peeled > trip_count:
            label = op.attrs.get("label", "")
            raise ValueError(
                f"PeelLoop: requested {self.front_iterations} front and "
                f"{self.back_iterations} back iterations from loop "
                f"{label!r} with trip count {trip_count}"
            )

        # Transform nested loops before cloning this loop's body.
        self.lower_block(op.body)

        lb, ub, step = op.lb, op.ub, op.step
        middle_lb = lb + self.front_iterations * step
        middle_ub = ub - self.back_iterations * step

        front_ops: list[Op] = []
        for i in range(self.front_iterations):
            iteration_iv = IndexConstantOp.build(value=lb + i * step)
            front_ops.append(iteration_iv)
            front_ops.extend(
                clone_loop_iteration(op, iteration_iv.result(), f"_front{i}")
            )

        back_ops: list[Op] = []
        first_back_iteration = trip_count - self.back_iterations
        for i in range(self.back_iterations):
            iteration_iv = IndexConstantOp.build(
                value=lb + (first_back_iteration + i) * step,
            )
            back_ops.append(iteration_iv)
            back_ops.extend(
                clone_loop_iteration(op, iteration_iv.result(), f"_back{i}")
            )

        middle_ops: list[Op] = []
        if num_peeled < trip_count:
            op.lb = middle_lb
            op.ub = middle_ub
            middle_ops.append(op)

        return [*front_ops, *middle_ops, *back_ops], []


def peel_loop(
    module: ModuleOp,
    label_filter: str,
    front_iterations: int = 0,
    back_iterations: int = 0,
) -> ModuleOp:
    """Peel matching loops by fixed front and back iteration counts."""
    return PeelLoop(
        label_filter=label_filter,
        front_iterations=front_iterations,
        back_iterations=back_iterations,
    ).run(module)
