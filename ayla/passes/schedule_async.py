"""Schedule async ops for maximum overlap.

Sinks ``await`` ops as late as possible — just before the first
consumer of the guarded buffer — and hoists ``async_move`` ops as
early as possible. This maximizes the window where async loads
overlap with compute. Loads from global memory into registers are
hoisted too, since only their first use waits; they go first, so the
async moves hoisted after them still issue ahead of them.

Hoisting an async op drags its full dependency cone of
side-effect-free ops (alloc, index arithmetic, tile ops) up with it,
so the async op rises as far as memory effects and external
(side-effecting) dependencies allow, rather than getting stuck behind
the ops that define its operands.
"""

from __future__ import annotations

import functools
from typing import cast

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    GmemTensorType,
    LockType,
    RegMemTensorType,
    TmaMoveOp,
    effective_operands,
    is_side_effect_free,
    ModuleOp,
    MoveOp,
    TileOp,
)
from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass
from ._utils import can_reorder, effect_roots


def _is_async_move(op: Op) -> bool:
    """The hoistable subset of async starts: data movement, not compute.

    Covers a TMA descriptor move and an ``async`` wrapping only moves; an
    ``async`` wrapping a matmul is tracked as a lock peer but never hoisted.
    """
    if isinstance(op, TmaMoveOp):
        return True
    return isinstance(op, AsyncOp) and all(isinstance(o, MoveOp) for o in op.body.ops)


def _is_register_load(op: Op) -> bool:
    return (
        isinstance(op, MoveOp)
        and isinstance(op.src.type, GmemTensorType)
        and isinstance(op.dst.type, RegMemTensorType)
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
    """Bubble each target through its neighbours as far as `can_reorder` allows.

    ``step`` is ``-1`` to move up (toward index 0) or ``+1`` to move down.
    Targets are processed in block order. Each move is a single adjacent swap
    gated by `can_reorder`, which is inherently SSA-safe: a consumer can never be
    swapped above its producer, since that swap would require
    ``can_reorder(producer, consumer)``, which the SSA check rejects. Valid SSA
    input therefore stays valid, and producers settle before the consumers
    stacked on them.
    """
    assert step in (-1, 1), f"step must be -1 or +1; got {step}"
    # Reordering doesn't change an op's effects, and recomputing them for
    # every candidate swap dominates the pass's runtime.
    cached_roots = functools.cache(effect_roots)
    for op in targets:
        j = next(i for i, o in enumerate(ops) if o is op)
        while 0 <= j + step < len(ops):
            lo, hi = sorted((j, j + step))
            if not can_reorder(ops[lo], ops[hi], cached_roots):
                break
            ops[j], ops[j + step] = ops[j + step], ops[j]
            j += step


def _hoist_async_moves(ops: list[Op]) -> None:
    # Targets: the register loads, then the async moves (matmuls are tracked as
    # lock peers but never moved), each with the side-effect-free dependency
    # cone feeding it. Then hoist each up. The async moves go second, so they
    # issue ahead of any register load they can pass.
    for is_target in (_is_register_load, _is_async_move):
        targets = _side_effect_free_operand_closure(
            [op for op in ops if is_target(op)], ops
        )
        _bubble(ops, targets, step=-1)
    # Each move now sits just below its own allocations, between the others'.
    # An allocation without an initializer has no operands, so move those back
    # to the front, in order: lock initializations then stay together, and one
    # barrier publishes them all.
    allocations = [op for op in ops if isinstance(op, AllocOp)]
    ops[:] = allocations + [op for op in ops if op not in set(allocations)]


def _sink_awaits(ops: list[Op]) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    targets = [cast(AwaitOp, op) for op in ops if _is_await(op)]
    _bubble(ops, [op for op in targets if cast(LockType, op.lock.type).kind != "bulk"], step=+1)
    # Bulk groups are per issuing thread. Independent output boxes can stay
    # pending while another fragment is published. Root-only effects otherwise
    # stop every await at the next write to the same output allocation.
    for wait in targets:
        assert isinstance(wait, AwaitOp)
        if cast(LockType, wait.lock.type).kind != "bulk" or wait.attrs.get("pending", 0):
            continue
        peers = [op for op in ops if isinstance(op, AsyncOp) and op.lock is wait.lock]
        if len(peers) != 1 or ops.index(peers[0]) > ops.index(wait):
            continue
        issue = peers[0]
        j = ops.index(wait)
        while j + 1 < len(ops):
            following = ops[j + 1]
            newer = sum(
                isinstance(op, AsyncOp) and op.lock.type == wait.lock.type
                for op in ops[ops.index(issue) + 1 : j]
            )
            if isinstance(following, AsyncOp) and following.lock.type == wait.lock.type:
                if newer == 7 or following.collective != issue.collective:
                    break
            if not can_reorder(wait, following) and not _independent_bulk_stores(
                issue, _bulk_peer(following, ops)
            ):
                break
            ops[j], ops[j + 1] = ops[j + 1], ops[j]
            j += 1
        pending = sum(
            isinstance(op, AsyncOp) and op.lock.type == wait.lock.type
            for op in ops[ops.index(issue) + 1 : ops.index(wait)]
        )
        if pending:
            wait.attrs["pending"] = pending


def _bulk_peer(op: Op, ops: list[Op]) -> Op:
    """Use the awaited store's region when comparing two completions."""
    if isinstance(op, AwaitOp) and cast(LockType, op.lock.type).kind == "bulk":
        peers = [
            issue
            for issue in ops
            if isinstance(issue, AsyncOp) and issue.lock is op.lock
        ]
        if len(peers) == 1:
            return peers[0]
    return op


def _independent_bulk_stores(first: AsyncOp, second: Op) -> bool:
    """Prove disjoint static output boxes under the same dynamic CTA view.

    This does not relax aliases on the shared source or move an await across
    control flow. Unknown views retain the conservative root dependency.
    """
    if (
        not isinstance(second, AsyncOp)
        or first.lock.type != second.lock.type
        or first.collective != second.collective
        or len(first.body.ops) != 1
        or len(second.body.ops) != 1
    ):
        return False
    a, b = first.body.ops[0], second.body.ops[0]
    if not (
        isinstance(a, MoveOp)
        and isinstance(b, MoveOp)
        and isinstance(a.dst.type, GmemTensorType)
        and isinstance(b.dst.type, GmemTensorType)
    ):
        return False
    ta, tb = a.dst.definition, b.dst.definition
    if not (
        isinstance(ta, TileOp)
        and isinstance(tb, TileOp)
        and ta.arg is tb.arg
        and all(i is not None for i in [*ta.static_indices, *tb.static_indices])
    ):
        return False
    return any(
        ia * da + sa <= ib * db or ib * db + sb <= ia * da
        for ia, ib, sa, sb, da, db in zip(
            ta.static_indices,
            tb.static_indices,
            a.dst.type.sizes,
            b.dst.type.sizes,
            ta.index_steps,
            tb.index_steps,
            strict=True,
        )
    )


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
