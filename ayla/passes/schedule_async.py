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

Hoisting suits loads but not stores: a store hoisted early strands its
staging copy ahead of the next compute. So async *stores* are instead sunk
into the following compute window (before that compute's await), letting the
compute latency hide the store.
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
    GmemTensorType,
    root_buffer,
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
        isinstance(o, MoveOp) for o in op.body.entry_block.ops
    )


def _is_await(op: Op) -> bool:
    return isinstance(op, AwaitOp)


def _can_swap(upper: Op, lower: Op) -> bool:

    # SSA dependency: lower consumes a result of upper.
    upper_results = set(upper.results)
    if any(v in upper_results for v in effective_operands(lower)):
        return False

    # A bulk-store completion wait may pass a later bulk-store launch.  This
    # does not reorder either transfer: both launches retain their order and
    # the later wait still drains the hardware bulk group.  Reuse of the
    # earlier store's shared source remains protected because the intervening
    # synchronous write conflicts with the await and stops it before reuse.
    # Without this rule, root-buffer aliasing conservatively makes stores to
    # distinct output tiles appear as a gmem WAW conflict.
    if (
        isinstance(upper, AwaitOp)
        and upper.lock.type.kind == "bulk"
        and isinstance(lower, AsyncOp)
        and lower.lock.type.kind == "bulk"
    ):
        return True

    # Memory effects (an await resolves to its lock peers via memory_effects).
    upper_effs = memory_effects(upper)
    lower_effs = memory_effects(lower)
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
    *,
    step: int,
    barrier=None,
) -> None:
    """Bubble each target through its neighbours as far as `_can_swap` allows.

    ``step`` is ``-1`` to move up (toward index 0) or ``+1`` to move down.
    Targets are processed in block order. Each move is a single adjacent swap
    gated by `_can_swap`, which is inherently SSA-safe: a consumer can never be
    swapped above its producer, since that swap would require
    ``_can_swap(producer, consumer)``, which the SSA check rejects. Valid SSA
    input therefore stays valid, and producers settle before the consumers
    stacked on them. ``barrier``, if given, is a predicate on the neighbour a
    target is about to cross; the target stops before any op it matches.
    """
    assert step in (-1, 1), f"step must be -1 or +1; got {step}"
    for op in targets:
        j = next(i for i, o in enumerate(ops) if o is op)
        while 0 <= j + step < len(ops):
            if barrier is not None and barrier(ops[j + step]):
                break
            lo, hi = sorted((j, j + step))
            if not _can_swap(ops[lo], ops[hi]):
                break
            ops[j], ops[j + step] = ops[j + step], ops[j]
            j += step


def _hoist_async_moves(ops: list[Op]) -> None:
    # Targets: the async moves (matmuls are tracked as lock peers but never
    # moved, and stores are handled by `_sink_stores`) plus the side-effect-free
    # dependency cone feeding each. Then hoist each up.
    targets = _side_effect_free_operand_closure(
        [op for op in ops if _is_async_move(op) and not _is_async_store(op)],
        ops,
    )
    _bubble(ops, targets, step=-1)


def _sink_awaits(ops: list[Op]) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    targets = [op for op in ops if _is_await(op)]
    _bubble(ops, targets, step=+1)


def _is_async_store(op: Op) -> bool:
    """A data-movement async whose transfer writes global memory.

    Loads and stores are both async moves; loads hoist early, stores defer so
    compute overlaps them. The inner move's destination space tells them apart.
    """
    return isinstance(op, AsyncOp) and _is_async_move(op) and any(
        isinstance(inner, MoveOp) and isinstance(inner.dst.type, GmemTensorType)
        for inner in op.body.entry_block.ops
    )


def _sink_stores(ops: list[Op]) -> None:
    """Sink each async store, then its staging producer, into the next compute
    window — up to but not past that compute's await — so compute hides the
    store. A store with no following compute await is left where hoisting put
    it. Motion is the same `_bubble`/`_can_swap` swap used for awaits.
    """
    def compute_await(op: Op) -> bool:
        return isinstance(op, AwaitOp) and op.lock.type.kind == "wgmma"

    stores, producers = [], []
    for store in [op for op in ops if _is_async_store(op)]:
        i = ops.index(store)
        if not any(compute_await(op) for op in ops[i + 1:]):
            continue
        reads = {r for kind, r in (memory_effects(store) or []) if kind == "R"}
        producer = next((
            op for op in reversed(ops[:i])
            if isinstance(op, MoveOp) and root_buffer(op.dst) in reads
        ), None)
        stores.append(store)
        if producer is not None:
            producers.append(producer)

    _bubble(ops, stores, step=+1, barrier=compute_await)
    _bubble(ops, producers, step=+1, barrier=compute_await)


class ScheduleAsync(Pass):

    def run_block(self, block: Block) -> None:
        _hoist_async_moves(block.ops)
        _sink_awaits(block.ops)
        _sink_stores(block.ops)
        super().run_block(block)


def schedule_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return ScheduleAsync().run(module)
