"""Insert cross-thread synchronization barriers on shared-buffer hazards."""
from __future__ import annotations

import enum

from ..gpu_dialect import (
    AsyncOp,
    AwaitOp,
    BarrierOp,
    CollectiveScope,
    ForOp,
    KernelOp,
    ModuleOp,
    PTXTemplateOp,
    is_side_effect_free,
    memory_effects,
)
from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass


def _join_collective_scope(a: CollectiveScope, b: CollectiveScope) -> CollectiveScope:
    """The join (least upper bound) on the CollectiveScope lattice."""
    if a == CollectiveScope.GRID:
        return a
    if a == CollectiveScope.THREAD:
        return b
    if a == CollectiveScope.BLOCK:
        return b if b == CollectiveScope.GRID else a
    if a == CollectiveScope.BLOCK_LEADER:
        if b in {CollectiveScope.THREAD, CollectiveScope.BLOCK_LEADER}:
            return a
        if b in {CollectiveScope.BLOCK, CollectiveScope.GRID}:
            return b
        return CollectiveScope.BLOCK  # b is WARP or WARPGROUP
    if a == CollectiveScope.WARPGROUP:
        if b in {CollectiveScope.BLOCK, CollectiveScope.GRID}:
            return b
        if b == CollectiveScope.BLOCK_LEADER:
            return CollectiveScope.BLOCK
        return a  # b is THREAD, WARP, or WARPGROUP
    if a == CollectiveScope.WARP:
        if b in {CollectiveScope.WARP, CollectiveScope.THREAD}:
            return a
        if b == CollectiveScope.BLOCK_LEADER:
            return CollectiveScope.BLOCK
        return b  # b is WARPGROUP, BLOCK, or GRID
    raise NotImplementedError(f"_join_collective_scope: unhandled pair {a!r}, {b!r}")


def _leq_collective_scope(a: CollectiveScope, b: CollectiveScope) -> bool:
    """The partial order (<=) of the CollectiveScope lattice."""
    return _join_collective_scope(a, b) == b


class BarrierScope(enum.IntEnum):
    r"""The cross-thread barrier to emit for a hazard.

    Mirrors CollectiveScope, but where that names the granularity at which an
    access *executes*, this names the *sync* between those granularities. The two
    correspond one-to-one except for the asymmetric block leader syncs,
    BLOCK_TO_LEADER and LEADER_TO_BLOCK::

             GRID
              |
            BLOCK
            /   \
    WARPGROUP   BLOCK_TO_LEADER / LEADER_TO_BLOCK
           |     |
         WARP    |
            \   /
            THREAD
    """
    THREAD = 0
    WARP = 1
    WARPGROUP = 2
    BLOCK_TO_LEADER = 3
    LEADER_TO_BLOCK = 4
    BLOCK = 5
    GRID = 6


def _join_barrier_scope(a: BarrierScope, b: BarrierScope) -> BarrierScope:
    """The join (least upper bound) on the BarrierScope lattice."""
    if a == b:
        return a
    if a == BarrierScope.THREAD:
        return b
    if b == BarrierScope.THREAD:
        return a
    if a == BarrierScope.GRID or b == BarrierScope.GRID:
        return BarrierScope.GRID
    if a == BarrierScope.BLOCK or b == BarrierScope.BLOCK:
        return BarrierScope.BLOCK
    if {a, b} == {BarrierScope.WARP, BarrierScope.WARPGROUP}:
        return BarrierScope.WARPGROUP
    return BarrierScope.BLOCK


def _barrier_scope(prior: CollectiveScope, current: CollectiveScope) -> BarrierScope:
    """The barrier scope needed to order a r/w hazard from ``prior`` to ``current``."""
    if prior == current == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.THREAD
    if current == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.BLOCK_TO_LEADER
    if prior == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.LEADER_TO_BLOCK
    # Neither side is a leader: the symmetric barrier for the coarser scope.
    joined = _join_collective_scope(prior, current)
    if joined == CollectiveScope.THREAD:
        return BarrierScope.THREAD
    if joined == CollectiveScope.WARP:
        return BarrierScope.WARP
    if joined == CollectiveScope.WARPGROUP:
        return BarrierScope.WARPGROUP
    if joined == CollectiveScope.BLOCK:
        return BarrierScope.BLOCK
    if joined == CollectiveScope.GRID:
        return BarrierScope.GRID
    raise NotImplementedError(f"_barrier_scope: unhandled join {joined!r}")


def _barrier_covers(barrier: BarrierScope, scope: CollectiveScope) -> bool:
    """Whether ``barrier`` orders a prior effect at collective ``scope`` -- i.e.
    the effect is discharged from the live set once the barrier is placed."""
    if barrier == BarrierScope.THREAD:
        return scope == CollectiveScope.THREAD
    if barrier == BarrierScope.WARP:
        return scope in (CollectiveScope.THREAD, CollectiveScope.WARP)
    if barrier == BarrierScope.WARPGROUP:
        return scope in (
            CollectiveScope.THREAD, CollectiveScope.WARP, CollectiveScope.WARPGROUP
        )
    if barrier == BarrierScope.BLOCK_TO_LEADER:
        return scope == CollectiveScope.THREAD  # gather: nothing cross-thread
    if barrier == BarrierScope.LEADER_TO_BLOCK:
        # broadcast: t0's own (BLOCK_LEADER) work only, not all-thread BLOCK
        return scope in (CollectiveScope.THREAD, CollectiveScope.BLOCK_LEADER)
    if barrier == BarrierScope.BLOCK:
        return scope != CollectiveScope.GRID
    if barrier == BarrierScope.GRID:
        return True
    raise NotImplementedError(f"_barrier_covers: unhandled barrier {barrier!r}")


# The dataflow state: outstanding concurrent memory effects per root buffer (the
# dict key), each a (kind, scope, region) tuple. ``region`` is the lock of the
# open async completion region the effect belongs to (its ``await`` will close
# it), or ``None`` for synchronous effects.
ConcurrentMemoryEffectsType = dict[
    object, set[tuple[str, "CollectiveScope", object]]
]

# The analysis result: each effect-bearing op mapped to the barrier it needs
# before it (THREAD if none).
BarrierTargetsType = dict["Op", "BarrierScope"]


def _await_collective_scope(lock: Value) -> CollectiveScope:
    """The collective scope an ``await`` on ``lock`` synchronizes."""
    kind = lock.type.kind
    if kind == "wgmma":
        return CollectiveScope.WARPGROUP
    if kind == "mbar":
        # A TMA-load mbar await is a leader completion: t0 issues the copy, all
        # threads wait the mbarrier for it. So it closes BLOCK_LEADER (the leader
        # write) but not a full-block all-thread effect.
        return CollectiveScope.BLOCK_LEADER
    raise NotImplementedError(f"_await_collective_scope: unmodeled lock kind {kind!r}")


def _join(
    a: ConcurrentMemoryEffectsType, b: ConcurrentMemoryEffectsType,
) -> ConcurrentMemoryEffectsType:
    """The join of this (standard forward) dataflow analysis: the per-buffer
    union of the concurrent-effect-map state."""
    return {
        buf: a.get(buf, set()) | b.get(buf, set())
        for buf in a.keys() | b.keys()
    }


def _analyze_block(
    ops: list[Op], live_in: ConcurrentMemoryEffectsType, targets: BarrierTargetsType,
) -> ConcurrentMemoryEffectsType:
    live = _join(live_in, {})
    for op in ops:
        live = _analyze(op, live, targets)
    return live


def _analyze(
    op: Op, live: ConcurrentMemoryEffectsType, targets: BarrierTargetsType,
) -> ConcurrentMemoryEffectsType:
    """Transfer the concurrent-effect map across one ``op``, recording the barrier
    it needs in ``targets``; return the live-out. Container ops recurse into their
    body; a ``for`` is iterated to a fixpoint over its header."""
    if isinstance(op, (ModuleOp, KernelOp)):
        # A module/kernel body is a fresh scope -- no live effects flow in, and
        # its internal effects don't escape to the enclosing scope.
        _analyze_block(list(op.body.entry_block.ops), {}, targets)
        return live
    if isinstance(op, ForOp):
        # Analyze the body to a fixpoint over its header (the loop-carried state
        # at the top of the body); return the loop's live-out.
        header = live
        while True:
            live_out = _analyze_block(op.body.entry_block.ops, header, targets)
            joined = _join(header, live_out)  # fold the back edge
            if joined == header:
                return live_out
            header = joined
    if is_side_effect_free(op):
        return live
    if isinstance(op, PTXTemplateOp):
        # Raw PTX is opaque, so we can't compute its effects. Skip it. This is
        # unsound in general (a template may read/write memory), but it lets
        # hand-written PTX kernels (e.g. fastcu) through on the assumption they
        # self-manage their sync.
        # TODO: gate on a kernel attribute marking the IR hazard-free or
        # already synced instead.
        return live
    if isinstance(op, AwaitOp):
        # Close the async's region: clear the effects up to the await's collective
        # scope, settle (drop the region tag from) the rest.
        scope = _await_collective_scope(op.lock)
        for buf in live:
            live[buf] = {
                (k, s, None) if region == op.lock else (k, s, region)
                for (k, s, region) in live[buf]
                if not (region == op.lock and _leq_collective_scope(s, scope))
            }
        return live
    effects = memory_effects(op)
    if effects is None:
        raise NotImplementedError(
            f"unmodelled effect-bearing op: {type(op).__name__}"
        )
    # The barrier this op needs, as the join over its hazards. Each hazard
    # resolves per-pair (:func:`_barrier_scope`).
    barrier_scope: BarrierScope = BarrierScope.THREAD
    for kind_y, buf, scope_y in effects:
        for kind_x, scope_x, _region_x in live.get(buf, ()):
            # A hazard needs a write (RAR is safe).
            if kind_x == "W" or kind_y == "W":
                barrier_scope = _join_barrier_scope(
                    barrier_scope, _barrier_scope(scope_x, scope_y)
                )
    targets[op] = barrier_scope
    # A barrier orders settled accesses it covers; pending (in-region) effects
    # survive — only their await completes them.
    for buf in live:
        live[buf] = {
            (k, s, region) for (k, s, region) in live[buf]
            if region is not None or not _barrier_covers(barrier_scope, s)
        }
    # The async's effects join a completion region keyed by its lock; every other
    # effect-bearing op is synchronous (region = None).
    region = op.lock if isinstance(op, AsyncOp) else None
    for (k, b, s) in effects:
        live.setdefault(b, set()).add((k, s, region))
    return live


# BarrierScope -> the BarrierOp ``scope`` attr.
_BARRIER_OP_SCOPE: dict[BarrierScope, str] = {
    BarrierScope.BLOCK_TO_LEADER: "block_leader",
    BarrierScope.BLOCK: "block",
}


def _insert_barriers(block: Block, targets: BarrierTargetsType) -> None:
    """Insert a ``barrier`` before every op the analysis assigned one."""
    new_ops: list[Op] = []
    for op in block.ops:
        barrier = targets.get(op, BarrierScope.THREAD)
        if barrier != BarrierScope.THREAD:
            if barrier not in _BARRIER_OP_SCOPE:
                raise NotImplementedError(f"no lowering for barrier {barrier.name}")
            new_ops.append(BarrierOp.build(scope=_BARRIER_OP_SCOPE[barrier]))
        new_ops.append(op)
        for region in op.regions:
            for inner in region.blocks:
                _insert_barriers(inner, targets)
    block.ops = new_ops


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        targets: BarrierTargetsType = {}
        _analyze(module, {}, targets)  # analysis: where do barriers go
        _insert_barriers(module.body.entry_block, targets)  # transform: insert them
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
