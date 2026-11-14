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
    RegMemTensorType,
    TmaMoveOp,
    effective_operands,
    has_ordering_effect,
    is_side_effect_free,
    memory_effects,
    memory_effects_may_alias,
    MatmulOp,
    ordered_async_roots,
    execution_issuer_warp,
    ModuleOp,
    KernelOp,
    LockType,
    MoveOp,
    SmemTensorType,
    TileOp,
    ForOp,
)
from ..base.ir import dependency_slice, Block, Op, Value
from ..base.pipeline import Pass


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


def _effect_roots(op: Op) -> tuple[set[Value], set[Value]] | None:
    """The buffers ``op`` reads and writes, or None if unknown."""
    # An await resolves to its lock peers via memory_effects.
    effects = memory_effects(op)
    if effects is None:
        return None
    reads = {root for effect in effects if effect.kind == "R" for root in effect.roots}
    writes = {root for effect in effects if effect.kind == "W" for root in effect.roots}
    return reads, writes


def _disjoint_store_views(a: Value, b: Value) -> bool:
    """Prove disjoint static boxes below a common (possibly dynamic) view.

    This proof is local to store scheduling. Casts, dynamic subtiles, partial
    boxes outside their parent, and different coordinate systems stay unknown.
    """
    if not isinstance(a.type, GmemTensorType) or not isinstance(b.type, GmemTensorType):
        return False

    def boxes(value):
        origin = [0] * value.type.rank
        size = list(value.type.sizes)
        if any(s is None for s in size):
            return {}
        result = {value: tuple(zip(origin, size))}
        while isinstance(value.definition, TileOp):
            tile = value.definition
            parent = tile.arg.type
            child = value.type
            indices = tile.static_indices
            if (not isinstance(parent, GmemTensorType) or parent.rank != child.rank
                    or parent.dtype != child.dtype or parent.strides != child.strides
                    or any(i is None or i < 0 for i in indices)
                    or any(p is None or i * s + s > p
                           for i, s, p in zip(indices, child.sizes, parent.sizes))):
                break
            origin = [o + i * s for o, i, s in zip(origin, indices, child.sizes)]
            value = tile.arg
            result[value] = tuple(zip(origin, size))
        return result

    first, second = boxes(a), boxes(b)
    for ancestor in first.keys() & second.keys():
        if any(x + xs <= y or y + ys <= x
               for (x, xs), (y, ys) in zip(first[ancestor], second[ancestor])):
            return True
    return False


def _bulk_issue(op: Op) -> bool:
    return (isinstance(op, AsyncOp) and op.lock.type.kind == 'bulk'
            and bool(op.body.ops) and all(isinstance(m, MoveOp)
                and m.attrs.get('tma') and isinstance(m.src.type, SmemTensorType)
                and isinstance(m.dst.type, GmemTensorType) for m in op.body.ops))


def _can_swap(upper: Op, lower: Op, effect_roots=_effect_roots, effects=memory_effects,
              *, queue_stores: bool = False) -> bool:

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
    # A completion wait may pass another issue to the same ordered engine.
    # Preserve issue order, and never reuse the event before observing it.
    wait, issue = (upper, lower) if isinstance(upper, AwaitOp) else (lower, upper)
    if isinstance(wait, AwaitOp) and isinstance(issue, AsyncOp):
        if wait.lock in issue.completion_locks:
            return False
        if cast(LockType, wait.lock.type).kind == cast(LockType, issue.lock.type).kind:
            ordered = ordered_async_roots(issue)
            peers = [peer for peer in wait.lock.users() if isinstance(peer, AsyncOp)
                     and execution_issuer_warp(peer) == execution_issuer_warp(issue)]
            ordered &= set().union(*(ordered_async_roots(peer) for peer in peers))
            upper_r, upper_w = upper_r - ordered, upper_w - ordered
            lower_r, lower_w = lower_r - ordered, lower_w - ordered
    conflicts = lower_w & upper_r | lower_r & upper_w | lower_w & upper_w
    if conflicts:
        upper_effects, lower_effects = effects(upper), effects(lower)
        assert upper_effects is not None and lower_effects is not None
        bulk_peers = (queue_stores and isinstance(wait, AwaitOp) and _bulk_issue(issue)
            and wait.lock.type.kind == 'bulk' and all(_bulk_issue(p)
                and execution_issuer_warp(p) == execution_issuer_warp(issue)
                for p in wait.lock.users() if isinstance(p, AsyncOp)))
        for a in upper_effects:
            for b in lower_effects:
                if ((a.kind == 'W' or b.kind == 'W') and a.roots & b.roots & conflicts
                        and memory_effects_may_alias(a, b)):
                    if bulk_peers and a.kind == b.kind == 'W' and _disjoint_store_views(a.value, b.value):
                        continue
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
    closure.update(dependency_slice(
        (v for seed in seeds for v in effective_operands(seed)),
        lambda value: value.definition in in_block
        and value.definition not in closure and is_side_effect_free(value.definition),
    ))
    return [op for op in ops if op in closure]


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


def _sink_awaits(ops: list[Op], *, reverse: bool = False) -> None:
    # Targets: the awaits; then sink each toward its first consumer.
    # Reverse visits can queue every independent MMA bank, but change K-loop
    # prefetch liveness. Enable only for schedules qualified with that order.
    targets = [op for op in (reversed(ops) if reverse else ops) if _is_await(op)]
    _bubble(ops, targets, step=+1)


def _queue_mma_completions(ops: list[Op], depth: int, *, carried: bool) -> None:
    """Keep ``depth`` MMA batches issued ahead of each observed completion.

    After awaits sink to their reuse and refills hoist behind them, each MMA
    completion wait follows the next batch: one batch stays queued while the
    issuer observes the previous one. Move each such wait, together with the
    refills issued right after it, forward until ``depth`` batches follow the
    batch it observes. In a loop body (``carried``), waits whose batch was
    issued by the previous iteration appear in issue order; the last of them
    observes the batch issued just before the block. Every move
    is an adjacent swap gated by `_can_swap`, so reuse, readiness and phase
    dependencies are unchanged.
    """
    roots = functools.cache(_effect_roots)

    def is_mma(op):
        return isinstance(op, AsyncOp) and cast(LockType, op.lock.type).kind == 'tcgen05'

    def is_wait(op):
        return isinstance(op, AwaitOp) and cast(LockType, op.lock.type).kind == 'tcgen05'

    issued_before = {}
    for wait in [op for op in ops if is_wait(op)]:
        index = ops.index(wait)
        if not any(is_mma(op) and wait.lock in op.completion_locks for op in ops[:index]):
            issued_before[wait] = None
    # Carried waits keep their issue order through sinking.
    leading = list(issued_before) if carried else []

    def queued(wait):
        index = ops.index(wait)
        if wait in leading:
            return len(leading) - 1 - leading.index(wait) + sum(map(is_mma, ops[:index]))
        if wait in issued_before:
            return None
        issue = max(i for i, op in enumerate(ops[:index]) if is_mma(op) and wait.lock in op.completion_locks)
        return sum(map(is_mma, ops[issue + 1:index]))

    def group(start):
        end = start + 1
        while end < len(ops) and not isinstance(ops[end], AwaitOp) and not is_mma(ops[end]):
            end += 1
        return end

    for wait in reversed([op for op in ops if is_wait(op)]):
        count = queued(wait)
        start = ops.index(wait)
        end = group(start)
        if count is None or not any(_is_async_move(op) for op in ops[start + 1:end]):
            continue
        while count < depth and end < len(ops):
            following = ops[end]
            if not all(_can_swap(member, following, roots) for member in ops[start:end]):
                break
            ops[start:end + 1] = [following, *ops[start:end]]
            start += 1
            end += 1
            count += is_mma(following)


def _queue_bulk_waits(ops: list[Op], maximum_pending: int) -> None:
    """Delay closed store observations, then count newer same-issuer groups.

    A block's queue is drained at its boundaries. A lock closed within this
    block may have other generations elsewhere; groups outside this block
    retain full waits. Counting issued groups (even if an intervening wait
    completed them) is conservative. The final wait drains to zero, and all
    original source-bank reuse dependencies are preserved.
    """
    closed = {}
    previous = {}
    for op in ops:
        lock = op.lock if isinstance(op, AsyncOp | AwaitOp) else None
        if lock is None or lock.type.kind != 'bulk':
            continue
        issue = previous.get(lock)
        if isinstance(op, AwaitOp) and issue is not None and _bulk_issue(issue):
            closed[op] = issue
        previous[lock] = op
    roots = functools.cache(_effect_roots)
    # Later waits move first, so an earlier one is not held behind them.
    for wait in reversed(closed):
        index = ops.index(wait)
        while index + 1 < len(ops) and _can_swap(
                ops[index], ops[index + 1], roots, queue_stores=True):
            ops[index], ops[index + 1] = ops[index + 1], ops[index]
            index += 1
    for wait, issue in closed.items():
        issuer = execution_issuer_warp(issue)
        newer = sum(_bulk_issue(op) and execution_issuer_warp(op) == issuer
                    for op in ops[ops.index(issue) + 1:ops.index(wait)])
        if newer:
            wait.attrs['pending'] = min(newer, maximum_pending)
        else:
            wait.attrs.pop('pending', None)


def _is_async_compute(op: Op) -> bool:
    return isinstance(op, AsyncOp) and bool(op.body.ops) and all(
        isinstance(inner, MatmulOp) for inner in op.body.ops)


def _group_async_computes(ops: list[Op], group_size: int) -> None:
    """Prefer consecutive MMA batches without weakening dependency proofs.

    Emit each group's compute dependencies first, then its remaining work in
    original order. Ready waits and required refills remain predecessors, so
    a bank wrap may shorten a group. Preserve engine issue order explicitly;
    completion events, phase observations and memory hazards use _can_swap.
    """
    computes = [i for i, op in enumerate(ops) if _is_async_compute(op)]
    if len(computes) < 2:
        return
    roots = functools.cache(_effect_roots)
    effects = functools.cache(memory_effects)
    predecessors = [set() for _ in ops]
    for j, later in enumerate(ops):
        for i, earlier in enumerate(ops[:j]):
            if not _can_swap(earlier, later, roots, effects):
                predecessors[j].add(i)
    moves = [i for i, op in enumerate(ops) if _is_async_move(op)]
    for sequence in (computes, moves):
        for a, b in zip(sequence, sequence[1:]):
            predecessors[b].add(a)
    emitted, order = set(), []

    def emit(index):
        stack = [(index, False)]
        while stack:
            current, ready = stack.pop()
            if current in emitted:
                continue
            if ready:
                emitted.add(current)
                order.append(current)
            else:
                stack.append((current, True))
                stack.extend((p, False) for p in sorted(predecessors[current], reverse=True)
                             if p not in emitted)

    cursor = 0
    for first in range(0, len(computes), group_size):
        group = computes[first:first + group_size]
        for index in group:
            emit(index)
        end = computes[first + group_size] if first + group_size < len(computes) else len(ops)
        for index in range(cursor, end):
            emit(index)
        cursor = end
    ops[:] = [ops[index] for index in order]


class ScheduleAsync(Pass):
    def __init__(self, *, pipeline_compute: bool = False,
                 store_pending_groups: int = 0, compute_issue_group_size: int = 1,
                 mma_queue_depth: int = 1) -> None:
        if type(mma_queue_depth) is not int or mma_queue_depth < 1:
            raise ValueError('MMA queue depth must be a positive integer')
        if mma_queue_depth > 1 and not pipeline_compute:
            raise ValueError('MMA queue depth requires compute pipelining')
        if type(compute_issue_group_size) is not int or compute_issue_group_size < 1:
            raise ValueError('compute issue group size must be a positive integer')
        if type(store_pending_groups) is not int or not 0 <= store_pending_groups <= 7:
            raise ValueError('store pending groups must be an integer in [0, 7]')
        super().__init__(pipeline_compute=pipeline_compute,
                         store_pending_groups=store_pending_groups,
                         compute_issue_group_size=compute_issue_group_size,
                         mma_queue_depth=mma_queue_depth)

    def run_block(self, block: Block) -> None:
        # Reorder a plain list: every write to ``block.ops`` re-syncs the whole
        # block, which makes bubbling through it cubic.
        ops = list(block.ops)
        parent = block.parent_op
        while parent is not None and not isinstance(parent,KernelOp):
            parent = parent.parent_block.parent_op if parent.parent_block is not None else None
        reverse = bool(parent is not None and parent.attrs.get('reverse_await_sinking'))
        queue_only = (self.store_pending_groups or self.compute_issue_group_size > 1) and not self.pipeline_compute
        if queue_only:
            pass  # Preserve the base schedule for an explicit queue preference.
        elif self.pipeline_compute:
            # Queue MMA by delaying completion, preserving issue/readiness
            # order. Then refill each released bank as early as possible.
            # Hoisting compute instead can postpone a refill behind a TMA
            # readiness wait and collapse the operand prefetch window.
            _sink_awaits(ops, reverse=reverse)
            _hoist_async_moves(ops)
            if self.mma_queue_depth > 1:
                _queue_mma_completions(ops, self.mma_queue_depth,
                                       carried=isinstance(block.parent_op, ForOp))
        else:
            _hoist_async_moves(ops)
            _sink_awaits(ops, reverse=reverse)
        if self.store_pending_groups:
            _queue_bulk_waits(ops, self.store_pending_groups)
        if self.compute_issue_group_size > 1 and isinstance(block.parent_op, ForOp):
            _group_async_computes(ops, self.compute_issue_group_size)
        block.ops = ops
        super().run_block(block)


def schedule_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return ScheduleAsync().run(module)
