"""Shared helpers for transformations that clone loop iterations."""
from __future__ import annotations

from ..base.ir import Block, Op, Value, clone_op, iter_ancestors, iter_ops
from ..gpu_dialect import ForOp, IfOp, IndexAddOp, IndexConstantOp, IndexLtOp, IndexMulOp, is_side_effect_free


def clone_loop_iteration(
    op: ForOp,
    iteration_iv: Value,
    label_suffix: str | None = None,
) -> list[Op]:
    """Clone a loop body once, rebinding its induction variable.

    When provided, ``label_suffix`` is appended to every label in the cloned
    iteration so later passes can select an individual specialized copy.
    """
    vmap: dict[Value, Value] = {op.iv: iteration_iv}
    cloned = [clone_op(body_op, vmap) for body_op in op.body.ops]
    if label_suffix is not None:
        for cloned_op in cloned:
            for nested in iter_ops(cloned_op):
                label = nested.attrs.get("label")
                if isinstance(label, str):
                    nested.attrs["label"] = label + label_suffix
    return cloned


def loop_guard(loop: ForOp) -> IfOp | None:
    """Recognize a grid-stride body with one trailing validity guard."""
    *prefix, last = loop.body.ops or [None]
    if isinstance(last, IfOp) and all(is_side_effect_free(op) for op in prefix):
        return last
    return None


def guard_narrows(loop: ForOp, guard: IfOp) -> bool:
    """Prove that a later valid iteration implies the current one is valid."""
    def invariant(value: Value) -> bool:
        return (isinstance(value.definition, Block) and value.definition is not loop.body) or (
            isinstance(value.definition, Op) and loop not in iter_ancestors(value.definition)
        )

    def increasing(value: Value) -> bool:
        if value is loop.iv:
            return True
        definition = value.definition
        if isinstance(definition, IndexAddOp):
            return (increasing(definition.lhs) and invariant(definition.rhs)
                    or invariant(definition.lhs) and increasing(definition.rhs))
        if isinstance(definition, IndexMulOp):
            for variable, factor in ((definition.lhs, definition.rhs), (definition.rhs, definition.lhs)):
                if increasing(variable) and isinstance(factor.definition, IndexConstantOp) and factor.definition.value > 0:
                    return True
        return False

    condition = guard.condition.definition
    return (loop.step > 0 and isinstance(condition, IndexLtOp)
            and increasing(condition.lhs) and invariant(condition.rhs))
