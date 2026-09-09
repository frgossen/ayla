"""Shared helpers for transformations that clone loop iterations."""
from __future__ import annotations

from ..base.ir import Op, Value, clone_op, iter_ops
from ..gpu_dialect import ForOp


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
