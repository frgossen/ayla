"""Schedule async ops for maximum overlap.

Sinks ``await`` ops as late as possible — just before the first
consumer of the guarded buffer — and hoists ``async_move`` ops as
early as possible. This maximizes the window where async loads
overlap with compute.

Hoisting an async op drags its full dependency cone of
side-effect-free ops (alloc, index arithmetic, tile ops) up with it,
so the async op rises as far as memory effects and external
(side-effecting) dependencies allow, rather than getting stuck behind
the ops that define its operands.
"""

from __future__ import annotations

from ..gpu_dialect import (
    AsyncOp,
    TmaMoveOp,
    AwaitOp,
    effective_operands,
    is_side_effect_free,
    memory_effects,
    ModuleOp,
    MoveOp,
)
from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass


def _is_async_move(op: Op) -> bool:
    """The hoistable subset of async starts: data movement, not compute.

    Covers an ``async`` wrapping only data movement, before or after TMA moves
    have been lowered to descriptor-based ``tma_move`` ops. An ``async``
    wrapping a matmul is tracked as a lock peer but never hoisted.
    """
    if not isinstance(op, AsyncOp):
        return False
    body_ops = op.body.entry_block.ops
    return bool(body_ops) and all(
        isinstance(o, (MoveOp, TmaMoveOp)) for o in body_ops
    )


def _is_await(op: Op) -> bool:
    return isinstance(op, AwaitOp)


def _is_lowered_tma_async(op: Op) -> bool:
    if not isinstance(op, AsyncOp):
        return False
    body_ops = op.body.entry_block.ops
    return bool(body_ops) and all(isinstance(o, TmaMoveOp) for o in body_ops)


def _can_swap(upper: Op, lower: Op) -> bool:

    # SSA dependency: lower consumes a result of upper.
    upper_results = set(upper.results)
    if any(v in upper_results for v in effective_operands(lower)):
        return False

    # Memory effects (an await resolves to its lock peers via memory_effects).
    upper_effs = memory_effects(upper)
    lower_effs = memory_effects(lower)
    if upper_effs is None or lower_effs is None:
        return False

    # RAW, WAR, or WAW conflict on the same buffer.
    upper_r = {root for kind, root, _scope in upper_effs if kind == "R"}
    upper_w = {root for kind, root, _scope in upper_effs if kind == "W"}
    lower_r = {root for kind, root, _scope in lower_effs if kind == "R"}
    lower_w = {root for kind, root, _scope in lower_effs if kind == "W"}
    if lower_w & upper_r or lower_r & upper_w or lower_w & upper_w:
        return False

    return True


def _side_effect_free_value_closure(
    values: list[Value], ops: list[Op], *, seeds: list[Op] | None = None,
) -> list[Op]:
    """The transitive side-effect-free producers of ``values`` plus ``seeds``.

    Walks definitions reachable from the values, keeping those that are
    in this block and side-effect-free, and stopping at any op that is not — a
    real barrier that must not be hoisted (including an ``alloc`` *with* an
    ``init``, which is not side-effect-free). Returned in block order. In valid
    SSA a definition precedes its uses, so a reached producer always precedes
    its seed; pulling these cone ops in is what frees an async move from being
    pinned behind the ops that define its operands.
    """
    in_block = set(ops)
    closure = set(seeds or [])
    worklist = list(values)
    while worklist:
        d = worklist.pop().definition
        if not isinstance(d, Op) or d not in in_block or d in closure:
            continue
        if not is_side_effect_free(d):
            continue  # barrier — do not include or recurse past it.
        closure.add(d)
        worklist.extend(d.operands)
    return [o for o in ops if o in closure]


def _side_effect_free_operand_closure(seeds: list[Op], ops: list[Op]) -> list[Op]:
    """The ``seeds`` plus their transitive side-effect-free operand cone."""
    return _side_effect_free_value_closure(
        [v for seed in seeds for v in effective_operands(seed)],
        ops,
        seeds=seeds,
    )


def _bubble(
    ops: list[Op],
    targets: list[Op],
    *,
    step: int,
) -> None:
    """Bubble each target through its neighbours as far as `_can_swap` allows.

    ``step`` is ``-1`` to move up (toward index 0) or ``+1`` to move down.
    Targets are processed in block order. Each move is a single adjacent swap
    gated by `_can_swap`, which is inherently SSA-safe: a consumer can never be
    swapped above its producer, since that swap would require
    ``_can_swap(producer, consumer)``, which the SSA check rejects. Valid SSA
    input therefore stays valid, and producers settle before the consumers
    stacked on them.
    """
    assert step in (-1, 1), f"step must be -1 or +1; got {step}"
    for op in targets:
        j = next(i for i, o in enumerate(ops) if o is op)
        while 0 <= j + step < len(ops):
            lo, hi = sorted((j, j + step))
            if not _can_swap(ops[lo], ops[hi]):
                break
            ops[j], ops[j + step] = ops[j + step], ops[j]
            j += step


def _hoist_async_moves(ops: list[Op]) -> None:
    # Before TMA lowering, hoist async moves with their full operand closure.
    # After lowering, leave the async issue at its established shared-memory
    # reuse boundary and hoist only the arithmetic feeding TMA coordinates —
    # not its lock, descriptor, or shared-memory allocation.
    seeds = [
        op for op in ops
        if _is_async_move(op) and not _is_lowered_tma_async(op)
    ]
    lowered_tmas = [op for op in ops if _is_lowered_tma_async(op)]
    coordinate_values = [
        index
        for async_op in lowered_tmas
        for body_op in async_op.body.entry_block.ops
        for index in body_op.indices
    ]
    target_set = set(_side_effect_free_operand_closure(seeds, ops))
    target_set.update(_side_effect_free_value_closure(coordinate_values, ops))
    targets = [op for op in ops if op in target_set]
    _bubble(ops, targets, step=-1)


def _sink_awaits(ops: list[Op]) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    targets = [op for op in ops if _is_await(op)]
    _bubble(ops, targets, step=+1)


class ScheduleAsync(Pass):

    def __init__(self, *, sink_awaits: bool = True) -> None:
        self.sink_awaits = sink_awaits

    def run_block(self, block: Block) -> None:
        # Reordering through Block.ops directly would make every adjacent swap
        # trigger _OpList's whole-block anchor reconciliation. Large unrolled
        # GEMMs perform many swaps, so schedule a plain list and publish the
        # final order once.
        ops = list(block.ops)
        _hoist_async_moves(ops)
        if self.sink_awaits:
            _sink_awaits(ops)
        block.ops = ops
        super().run_block(block)


def schedule_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return ScheduleAsync().run(module)
