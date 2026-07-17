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

from .gpu_dialect import (
    AsyncOp,
    TmaMoveOp,
    AwaitOp,
    effective_operands,
    is_side_effect_free,
    memory_effects,
    ModuleOp,
    MoveOp,
)
from .ir import Block, Op, Value
from .pipeline import Pass


def _is_async_op(op: Op) -> bool:
    """An async start: issues work completed later by an ``await`` on its lock."""
    return isinstance(op, (AsyncOp, TmaMoveOp))


def _is_async_move(op: Op) -> bool:
    """The hoistable subset of async starts: data movement, not compute.

    Covers a TMA descriptor move and an ``async`` wrapping only moves; an
    ``async`` wrapping a matmul is tracked as a lock peer but never hoisted.
    """
    if isinstance(op, TmaMoveOp):
        return True
    return isinstance(op, AsyncOp) and all(
        isinstance(o, MoveOp) for o in op.body.entry_block.ops
    )


def _is_await(op: Op) -> bool:
    return isinstance(op, AwaitOp)


def _lock_to_asyncs(ops: list[Op]) -> dict[Value, list[Op]]:
    """Map each lock to the async starts sharing it (its peers), in block order.

    Built once per block and threaded through hoisting and await-sinking:
    peer-union effect resolution needs every peer of every lock, and reordering
    ops never invalidates the map (it holds op identities, not positions).
    """
    lock_to_asyncs: dict[Value, list[Op]] = {}
    for op in ops:
        if _is_async_op(op):
            lock_to_asyncs.setdefault(op.lock, []).append(op)
    return lock_to_asyncs


def await_aware_memory_effects(
    op: Op,
    lock_to_asyncs: dict[Value, list[Op]],
) -> list[tuple[str, Value]] | None:
    """``memory_effects`` extended to resolve lock synchronization.

    Delegates to the dialect's per-op ``memory_effects`` for ordinary ops, but
    resolves any lock-bearing op — an async start or an ``await`` — to the
    union of the effects of *all* same-block async starts sharing its lock
    (its "peers"). A wgmma group is many single-matmul ``async`` ops on one
    lock plus a single ``await``: every one must be treated as reading the
    whole group's operands, since those operands stay live until the await.
    Collapsing the peers to a single start (last-writer-wins) is exactly the
    bug that let a prefetch ``async_move`` hoist into an in-flight group.

    Returns ``None`` (worst-case conflict) for an unmodelled peer, or for a
    lock-bearing op with no registered peers.
    """
    lock = op.lock if (_is_await(op) or _is_async_op(op)) else None
    if lock is None:
        return memory_effects(op)
    peers = lock_to_asyncs.get(lock)
    if not peers:
        return None
    effects: list[tuple[str, Value]] = []
    for peer in peers:
        peer_effs = memory_effects(peer)
        if peer_effs is None:
            return None
        effects.extend(peer_effs)
    return effects


def _can_swap(
    upper: Op,
    lower: Op,
    lock_to_asyncs: dict[Value, list[Op]],
) -> bool:

    # SSA dependency: lower consumes a result of upper.
    upper_results = set(upper.results)
    if any(v in upper_results for v in effective_operands(lower)):
        return False

    # Memory effects (lock-bearing ops union over their lock peers).
    upper_effs = await_aware_memory_effects(upper, lock_to_asyncs)
    lower_effs = await_aware_memory_effects(lower, lock_to_asyncs)
    if upper_effs is None or lower_effs is None:
        return False

    # RAW, WAR, or WAW conflict on the same buffer.
    upper_r = {root for kind, root in upper_effs if kind == "R"}
    upper_w = {root for kind, root in upper_effs if kind == "W"}
    lower_r = {root for kind, root in lower_effs if kind == "R"}
    lower_w = {root for kind, root in lower_effs if kind == "W"}
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
    lock_to_asyncs: dict[Value, list[Op]],
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
            if not _can_swap(ops[lo], ops[hi], lock_to_asyncs):
                break
            ops[j], ops[j + step] = ops[j + step], ops[j]
            j += step


def _hoist_async_moves(ops: list[Op], lock_to_asyncs: dict[Value, list[Op]]) -> None:
    # Targets: the async moves (matmuls are tracked as lock peers but never
    # moved) plus the side-effect-free dependency cone feeding each. Then hoist
    # each up.
    targets = _side_effect_free_operand_closure(
        [op for op in ops if _is_async_move(op)], ops
    )
    _bubble(ops, targets, lock_to_asyncs, step=-1)


def _sink_awaits(ops: list[Op], lock_to_asyncs: dict[Value, list[Op]]) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    targets = [op for op in ops if _is_await(op)]
    _bubble(ops, targets, lock_to_asyncs, step=+1)


class ScheduleAsync(Pass):

    def run_block(self, block: Block) -> None:
        lock_to_asyncs = _lock_to_asyncs(block.ops)
        _hoist_async_moves(block.ops, lock_to_asyncs)
        _sink_awaits(block.ops, lock_to_asyncs)
        super().run_block(block)


def schedule_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return ScheduleAsync().run(module)
