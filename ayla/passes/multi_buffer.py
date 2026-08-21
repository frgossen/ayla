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

Folding cleans up the redundant index computation.
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
from ..base.ir import Op, clone_op
from ..base.lowering_pass import LoweringPass


class MultiBuffer(LoweringPass):
    """Clone loop-local allocations `depth` times and select per iteration."""

    def __init__(self, depth: int, label_filter: str | None = None) -> None:
        super().__init__()
        self.register(AllocOp, type(self).lower_alloc)
        self.register(ForOp, type(self).lower_for)
        self.depth = depth
        self.label_filter = label_filter
        # The loops enclosing the allocation currently being rewritten.
        #
        # TODO: with a parent pointer on Op this goes away, and so does the
        # ForOp hook -- an allocation could just walk up to the loop whose
        # iterations index it. Second pass to want this; see folding.
        self._loops: list[ForOp] = []

    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]]:
        assert len(op.body.blocks) == 1, (
            f"MultiBuffer: expected one block in for body, got {len(op.body.blocks)}"
        )
        self._loops.append(op)
        try:
            self.lower_region(op.body)
        finally:
            self._loops.pop()
        return [op], list(op.results)

    def lower_alloc(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]] | None:
        if self.depth <= 1 or not self._loops:
            return None
        if self.label_filter is not None and op.attrs.get("label") != self.label_filter:
            return None
        # A per-iteration initializer would have to be re-run per buffer.
        if op.init is not None:
            return None

        loop = self._loops[-1]
        # Unrolling is the only thing that gives a loop a step other than 1, and
        # it runs after this pass, so a strided loop here means the pipeline is
        # wrong rather than that the index needs to divide the step out.
        assert loop.step == 1, (
            f"MultiBuffer: expected an unstrided loop, got step {loop.step}"
        )

        clones = [clone_op(op, {}) for _ in range(self.depth)]
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
