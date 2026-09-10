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
from ..base.ir import Op, clone_op
from ..base.lowering_pass import LoweringPass, lowers
from .utils import label_matches


def _enclosing_loop(op: Op) -> ForOp | None:
    """The innermost ``for`` whose body holds ``op``, if there is one.

    The loop that indexes an allocation's buffers is the one it sits inside,
    which the parent links reach directly from the allocation itself.
    """
    block = op.parent_block
    while block is not None:
        parent = block.parent_op
        if isinstance(parent, ForOp):
            return parent
        block = None if parent is None else parent.parent_block
    return None


class MultiBuffer(LoweringPass):
    """Clone loop-local allocations `depth` times and select per iteration."""

    def __init__(self, depth: int, label_filter: str | None = None) -> None:
        super().__init__()
        self.depth = depth
        self.label_filter = label_filter

    @lowers(AllocOp)
    def lower_alloc(
        self, op: AllocOp,
    ) -> tuple[list[Op], list[object]] | None:
        loop = _enclosing_loop(op)
        if self.depth <= 1 or loop is None:
            return None
        if not label_matches(op, self.label_filter):
            return None
        # A per-iteration initializer would have to be re-run per buffer.
        if op.init is not None:
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
