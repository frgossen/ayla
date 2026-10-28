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

import functools

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    TmaMoveOp,
    effective_operands,
    has_ordering_effect,
    is_side_effect_free,
    memory_effects,
    ModuleOp,
    MoveOp,
)
from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass


def _is_async_move(op: Op) -> bool:
    """The hoistable subset of async starts: data movement, not compute.

    Covers a TMA descriptor move and an ``async`` wrapping only moves; an
    ``async`` wrapping a matmul is tracked as a lock peer but never hoisted.
    """
    if isinstance(op, TmaMoveOp):
        return True
    return isinstance(op, AsyncOp) and all(
        isinstance(o, MoveOp) for o in op.body.ops
    )


def _is_await(op: Op) -> bool:
    return isinstance(op, AwaitOp)


def _effect_roots(op: Op) -> tuple[set[Value], set[Value]] | None:
    """The buffers ``op`` reads and writes, or None if unknown."""
    # An await resolves to its lock peers via memory_effects.
    effects = memory_effects(op)
    if effects is None:
        return None
    reads = {
        root for effect in effects if effect.kind == "R"
        for root in effect.roots
    }
    writes = {
        root for effect in effects if effect.kind == "W"
        for root in effect.roots
    }
    return reads, writes


def _can_swap(upper: Op, lower: Op, effect_roots=_effect_roots) -> bool:

    if has_ordering_effect(upper) or has_ordering_effect(lower):
        return False

    # SSA dependency: lower consumes a result of upper.
    upper_results = set(upper.results)
    if any(v in upper_results for v in effective_operands(lower)):
        return False

    upper_roots = effect_roots(upper)
    lower_roots = effect_roots(lower)
    if upper_roots is None or lower_roots is None:
        return False

    # RAW, WAR, or WAW conflict on the same buffer.
    upper_r, upper_w = upper_roots
    lower_r, lower_w = lower_roots
    if lower_w & upper_r or lower_r & upper_w or lower_w & upper_w:
        return False

    return True


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
    # Reordering doesn't change an op's effects, and recomputing them for
    # every candidate swap dominates the pass's runtime.
    effect_roots = functools.cache(_effect_roots)
    for op in targets:
        j = next(i for i, o in enumerate(ops) if o is op)
        while 0 <= j + step < len(ops):
            lo, hi = sorted((j, j + step))
            if not _can_swap(ops[lo], ops[hi], effect_roots):
                break
            ops[j], ops[j + step] = ops[j + step], ops[j]
            j += step


def _hoist_async_moves(ops: list[Op]) -> None:
    # Targets: the async moves (matmuls are tracked as lock peers but never
    # moved) plus the side-effect-free dependency cone feeding each. Then hoist
    # each up.
    targets = _side_effect_free_operand_closure(
        [op for op in ops if _is_async_move(op)], ops
    )
    _bubble(ops, targets, step=-1)
    # Each move now sits just below its own allocations, between the others'.
    # An allocation without an initializer has no operands, so move those back
    # to the front, in order: lock initializations then stay together, and one
    # barrier publishes them all.
    allocations = [
        op for op in ops if isinstance(op, AllocOp) and op.init is None
    ]
    ops[:] = allocations + [op for op in ops if op not in set(allocations)]


def _sink_awaits(ops: list[Op]) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    targets = [op for op in ops if _is_await(op)]
    _bubble(ops, targets, step=+1)


class ScheduleAsync(Pass):

    def run_block(self, block: Block) -> None:
        # Reorder a plain list: every write to ``block.ops`` re-syncs the whole
        # block, which makes bubbling through it cubic.
        ops = list(block.ops)
        _hoist_async_moves(ops)
        _sink_awaits(ops)
        block.ops = ops
        super().run_block(block)


def schedule_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return ScheduleAsync().run(module)
