"""Give a loop-carried buffer several copies, chosen per iteration.

Clones a loop-local allocation ``depth`` times and routes its uses through a
``select`` on the iteration's buffer index. The loop is untouched::

    for (%k = lb to ub) {             for (%k = lb to ub) {
      %b = alloc()                      %b0 = alloc()
      move(%b, ...)          ==>        %b1 = alloc()
      use(%b)                           %i = (%k - lb) % 2
    }                                   %b = select(%i, %b0, %b1)
                                        move(%b, ...)
                                        use(%b)
                                      }

Pass a label_filter: a loop body also holds buffers this has no business
duplicating, such as an accumulator or a staging buffer belonging to an outer
loop. Folding cleans up the redundant index computation.
"""
from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexModOp,
    ModuleOp,
    SelectOp,
)
from ..base.ir import Op, clone_op, find_ancestor_of_type
from ..base.lowering_pass import LoweringPass, lowers
from ._utils import label_matches


class MultiBuffer(LoweringPass):
    """Clone loop-local allocations `depth` times and select per iteration."""

    depth: int
    label_filter: str | None

    def __init__(self, depth: int, label_filter: str | None = None) -> None:
        super().__init__(depth=depth, label_filter=label_filter)

    @lowers(AllocOp)
    def lower_alloc(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]] | None:
        loop = find_ancestor_of_type(op, ForOp)
        if self.depth <= 1 or loop is None:
            return None
        if not label_matches(op, self.label_filter):
            return None

        # Unrolling is the only thing that gives a loop a step other than 1, and
        # it runs after this pass, so a strided loop here means the pipeline is
        # wrong rather than that the index needs to divide the step out.
        assert loop.step == 1, (
            f"MultiBuffer: expected an unstrided loop, got step {loop.step}"
        )

        # Each copy is its own buffer, so each needs its own name.
        clones = [
            clone_op(op, {}, override_attrs={"id": AllocOp.fresh_id()})
            for _ in range(self.depth)
        ]
        # Which buffer this iteration uses: (iv - lb) % depth.
        lb = IndexConstantOp.build(value=-loop.lb)
        shifted = IndexAddOp.build(loop.iv, lb.result())
        span = IndexConstantOp.build(value=self.depth)
        index = IndexModOp.build(shifted.result(), span.result())

        select = SelectOp.build(
            index.result(), [clone.result() for clone in clones]
        )
        # Handing back the select as this allocation's result is what redirects
        # every use of it, wherever they are.
        return [*clones, lb, shifted, span, index, select], [select.result()]


def multi_buffer(
    module: ModuleOp, depth: int, label_filter: str | None = None,
) -> ModuleOp:
    """Convenience entry point."""
    return MultiBuffer(depth, label_filter).run(module)
