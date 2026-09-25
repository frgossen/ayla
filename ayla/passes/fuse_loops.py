"""Fuse compatible sequential loops, peeling an unequal tail."""

from __future__ import annotations

from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass
from ..gpu_dialect import (
    BarrierOp,
    ForOp,
    IndexType,
    ModuleOp,
    is_side_effect_free,
)
from ._loop_utils import clone_loop_iteration
from ._utils import can_reorder, label_matches


def _split_body(ops: list[Op]) -> tuple[list[Op], list[Op]] | None:
    """Split a loop body into pure setup followed by effectful work."""
    split = next(
        (i for i, op in enumerate(ops) if not is_side_effect_free(op)),
        len(ops),
    )
    if split == len(ops) or any(is_side_effect_free(op) for op in ops[split:]):
        return None
    return ops[:split], ops[split:]


def _all_reorderable(earlier: list[Op], later: list[Op]) -> bool:
    return all(can_reorder(first, second) for first in earlier for second in later)


class FuseLoops(Pass):
    """Fuse selected neighboring loops when reordering is provably safe.

    Non-loop operations may occur between the loops. They remain after the
    first loop's completion, which naturally places an epilogue before a
    longer second loop's peeled tail.
    """

    loop_label_filter: str | None
    body_label_filter: str | None
    tail_iterations: int

    def __init__(
        self,
        loop_label_filter: str | None = None,
        body_label_filter: str | None = None,
        tail_iterations: int = 0,
    ) -> None:
        if tail_iterations < 0:
            raise ValueError("FuseLoops: tail_iterations must be non-negative")
        super().__init__(
            loop_label_filter=loop_label_filter,
            body_label_filter=body_label_filter,
            tail_iterations=tail_iterations,
        )

    def _selected(self, op: Op) -> bool:
        if not isinstance(op, ForOp) or not label_matches(
            op, self.loop_label_filter,
        ):
            return False
        return self.body_label_filter is None or any(
            label_matches(body_op, self.body_label_filter)
            for body_op in op.body.ops
        )

    def _fuse(
        self, first: ForOp, between: list[Op], second: ForOp,
    ) -> list[Op] | None:
        if (
            first.lb != second.lb
            or first.step != second.step
            or first.extra_attrs != second.extra_attrs
            or first.ub <= first.lb
            or second.ub <= second.lb
            or (first.ub - first.lb) % first.step
            or (second.ub - second.lb) % second.step
        ):
            return None
        first_parts = _split_body(first.body.ops)
        second_parts = _split_body(second.body.ops)
        if first_parts is None or second_parts is None:
            return None
        if not _all_reorderable(first.body.ops, second.body.ops):
            return None
        if not _all_reorderable(between, second.body.ops):
            return None

        common_ub = min(first.ub, second.ub) - self.tail_iterations * first.step
        if common_ub <= first.lb:
            return None
        iv = Value(name=first.iv.name, type=IndexType())
        cloned_first = _split_body(clone_loop_iteration(first, iv))
        cloned_second = _split_body(clone_loop_iteration(second, iv))
        assert cloned_first is not None and cloned_second is not None
        first_setup, first_work = cloned_first
        second_setup, second_work = cloned_second
        common = ForOp.build(
            lb=first.lb,
            ub=common_ub,
            step=first.step,
            body=Block(
                arguments=[iv],
                ops=[*first_setup, *second_setup, *first_work, *second_work],
            ),
            attrs=first.extra_attrs,
        )

        result: list[Op] = [common]
        first_tail: list[Op] = []
        second_tail: list[Op] = []
        if first.ub > common_ub:
            first.lb = common_ub
            if self.tail_iterations:
                first.attrs["peeled"] = True
                for body_op in first.body.ops:
                    body_op.attrs["peeled"] = True
                if isinstance(label := first.attrs.get("label"), str):
                    first.attrs["label"] = f"{label}_first_tail"
            first_tail.append(first)
        if second.ub > common_ub:
            second.lb = common_ub
            if self.tail_iterations:
                second.attrs["peeled"] = True
                for body_op in second.body.ops:
                    body_op.attrs["peeled"] = True
                if isinstance(label := second.attrs.get("label"), str):
                    second.attrs["label"] = f"{label}_second_tail"
            second_tail.append(second)
        if self.tail_iterations and first_tail and second_tail:
            # The peeled loops may reuse storage owned by the common pipeline.
            # Keep asynchronous tail loads behind that lifetime boundary.
            result.append(BarrierOp.build("thread"))
            result.extend(first_tail)
            result.extend(second_tail)
            result.extend(between)
        else:
            result.extend(first_tail)
            result.extend(between)
            result.extend(second_tail)
        return result

    def run_block(self, block: Block) -> None:
        for op in list(block.ops):
            self.run_op(op)

        changed = True
        while changed:
            changed = False
            ops = list(block.ops)
            result: list[Op] = []
            i = 0
            while i < len(ops):
                if not self._selected(ops[i]):
                    result.append(ops[i])
                    i += 1
                    continue
                j = i + 1
                while j < len(ops) and not isinstance(ops[j], ForOp):
                    j += 1
                if j == len(ops) or not self._selected(ops[j]):
                    result.append(ops[i])
                    i += 1
                    continue
                first, second = ops[i], ops[j]
                assert isinstance(first, ForOp) and isinstance(second, ForOp)
                fused = self._fuse(first, ops[i + 1:j], second)
                if fused is None:
                    result.extend(ops[i:j])
                    i = j
                    continue
                result.extend(fused)
                changed = True
                i = j + 1
            block.ops = result
            if self.tail_iterations:
                break


def fuse_loops(
    module: ModuleOp,
    loop_label_filter: str | None = None,
    body_label_filter: str | None = None,
    tail_iterations: int = 0,
) -> ModuleOp:
    return FuseLoops(
        loop_label_filter, body_label_filter, tail_iterations,
    ).run(module)
