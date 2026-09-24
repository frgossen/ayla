"""Schedule async ops for maximum overlap.

Sinks ``await`` ops as late as possible — just before the first consumer of
the guarded buffer — and hoists asynchronous moves as early as possible. TMA
``move`` ops may be scheduled before ``LowerToAsync`` wraps them. This
maximizes the window where async loads overlap with compute.

Hoisting an async op drags its full dependency cone of
side-effect-free ops (alloc, index arithmetic, tile ops) up with it,
so the async op rises as far as memory effects and external
(side-effecting) dependencies allow, rather than getting stuck behind
the ops that define its operands.
"""

from __future__ import annotations

from ..gpu_dialect import (
    AsyncOp,
    AwaitOp,
    TmaMoveOp,
    effective_operands,
    is_side_effect_free,
    ModuleOp,
    MoveOp,
)
from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass
from ._utils import can_reorder, label_matches


def _is_async_move(op: Op) -> bool:
    """The hoistable subset of async starts: data movement, not compute.

    Covers TMA moves before or after async lowering. An ``async`` wrapping a
    matmul is tracked as a lock peer but never hoisted.
    """
    if isinstance(op, TmaMoveOp) or (
        isinstance(op, MoveOp) and bool(op.attrs.get("tma"))
    ):
        return True
    return isinstance(op, AsyncOp) and all(
        isinstance(o, MoveOp) for o in op.body.ops
    )


def _is_await(op: Op) -> bool:
    return isinstance(op, AwaitOp)


def _side_effect_free_operand_closure(seeds: list[Op], ops: list[Op]) -> list[Op]:
    """The ``seeds`` plus their transitive side-effect-free dependency cone.

    Walks operand definitions reachable from the seeds, keeping those that are
    in this block and side-effect-free, and stopping at any op that is not — a
    real barrier that must not be hoisted (including an ``alloc`` *with* an
    ``init``, which is not side-effect-free). Returned in block order. In valid
    SSA a definition precedes its uses, so a reached producer always precedes
    its seed; pulling these cone ops in is what frees an async move from being
    pinned behind the ops that define its operands.
    """
    in_block = set(ops)
    closure = set(seeds)
    worklist: list[Value] = [v for seed in seeds for v in effective_operands(seed)]
    while worklist:
        d = worklist.pop().definition
        if not isinstance(d, Op) or d not in in_block or d in closure:
            continue
        if not is_side_effect_free(d):
            continue  # barrier — do not include or recurse past it.
        closure.add(d)
        worklist.extend(d.operands)
    return [o for o in ops if o in closure]


def _bubble(
    ops: list[Op],
    targets: list[Op],
    *,
    step: int,
) -> None:
    """Bubble each target through its neighbours as far as allowed.

    ``step`` is ``-1`` to move up (toward index 0) or ``+1`` to move down.
    Targets are processed front-to-back when hoisting and back-to-front when
    sinking, and never cross one another. Each move is a single adjacent swap
    gated by ``can_reorder``, which is inherently SSA-safe: a consumer cannot be
    swapped above its producer, since that swap would require
    ``can_reorder(producer, consumer)``, which the SSA check rejects. Valid SSA
    input therefore stays valid, and producers settle before the consumers
    stacked on them.
    """
    assert step in (-1, 1), f"step must be -1 or +1; got {step}"
    target_set = set(targets)
    ordered_targets = targets if step == -1 else reversed(targets)
    for op in ordered_targets:
        j = next(i for i, o in enumerate(ops) if o is op)
        while 0 <= j + step < len(ops):
            lo, hi = sorted((j, j + step))
            if ops[j + step] in target_set:
                break
            if not can_reorder(ops[lo], ops[hi]):
                break
            ops[j], ops[j + step] = ops[j + step], ops[j]
            j += step


def _hoist_async_moves(ops: list[Op], label_filter: str | None) -> None:
    # Targets: the async moves (matmuls are tracked as lock peers but never
    # moved) plus the side-effect-free dependency cone feeding each. Then hoist
    # each up.
    targets = _side_effect_free_operand_closure(
        [
            op for op in ops
            if _is_async_move(op) and label_matches(op, label_filter)
        ],
        ops,
    )
    _bubble(ops, targets, step=-1)


def _sink_awaits(ops: list[Op], label_filter: str | None) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    targets = [
        op for op in ops
        if _is_await(op) and label_matches(op, label_filter)
    ]
    _bubble(ops, targets, step=+1)


class ScheduleAsync(Pass):

    label_filter: str | None

    def __init__(self, label_filter: str | None = None) -> None:
        super().__init__(label_filter=label_filter)

    def run_block(self, block: Block) -> None:
        _hoist_async_moves(block.ops, self.label_filter)
        _sink_awaits(block.ops, self.label_filter)
        super().run_block(block)


def schedule_async(
    module: ModuleOp, label_filter: str | None = None,
) -> ModuleOp:
    """Convenience entry point."""
    return ScheduleAsync(label_filter).run(module)
