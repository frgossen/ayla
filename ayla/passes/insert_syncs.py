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
from ..base.ir import Block, Op, Value, find_op
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


def _leq_barrier_scope(a: BarrierScope, b: BarrierScope) -> bool:
    """The partial order (<=) of the BarrierScope lattice."""
    return _join_barrier_scope(a, b) == b


def _barrier_scope(prior: CollectiveScope, current: CollectiveScope) -> BarrierScope:
    """The barrier ordering a single r/w hazard from ``prior`` to ``current``.

    Directional for the leader (t0): two leader accesses are program-ordered; an
    all-thread access before a leader op needs BLOCK_TO_LEADER, and a leader op
    before an all-thread access needs LEADER_TO_BLOCK. Otherwise the join of the
    two scopes names the (symmetric) barrier."""
    leader = CollectiveScope.BLOCK_LEADER
    if prior == leader and current == leader:
        return BarrierScope.THREAD
    if current == leader:
        return BarrierScope.BLOCK_TO_LEADER
    if prior == leader:
        return BarrierScope.LEADER_TO_BLOCK
    return BarrierScope[_join_collective_scope(prior, current).name]


def _barrier_covers(barrier: BarrierScope, scope: CollectiveScope) -> bool:
    """Whether ``barrier`` orders a prior effect at collective ``scope`` -- i.e.
    the effect is discharged from the live set once the barrier is placed."""
    if barrier in (BarrierScope.BLOCK_TO_LEADER, BarrierScope.LEADER_TO_BLOCK):
        return scope in (CollectiveScope.THREAD, CollectiveScope.BLOCK_LEADER)
    # A symmetric barrier names a CollectiveScope and covers it and everything
    # below it in the lattice.
    return _leq_collective_scope(scope, CollectiveScope[barrier.name])


# The dataflow state: outstanding concurrent memory effects per root buffer, each
# a (kind, root_buffer, scope, region) tuple. ``region`` is the lock of the open
# async completion region the effect belongs to (its ``await`` will close it), or
# ``None`` for synchronous effects.
ConcurrentMemoryEffectsType = dict[
    object, set[tuple[str, object, "CollectiveScope", object]]
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


def _transfer(
    ops: list[Op], live_in: ConcurrentMemoryEffectsType, targets: BarrierTargetsType,
) -> ConcurrentMemoryEffectsType:
    """Thread the concurrent-effect map through a straight-line ``ops`` sequence,
    recording into ``targets`` the barrier scope each op needs; return the
    live-out. Pure w.r.t. ``live_in``. A ``for`` op recurses via
    :func:`_transfer_loop`."""
    live = _join(live_in, {})
    for op in ops:
        if isinstance(op, ForOp):
            live = _transfer_loop(op, live, targets)
            continue
        if is_side_effect_free(op):
            continue
        if isinstance(op, PTXTemplateOp):
            # Raw PTX is opaque, so we can't compute its effects. Skip it. This is
            # unsound in general (a template may read/write memory), but it lets
            # hand-written PTX kernels (e.g. fastcu) through on the assumption they
            # self-manage their sync.
            # TODO: gate on a kernel attribute marking the IR hazard-free or
            # already synced instead.
            continue
        if isinstance(op, AwaitOp):
            # Close the async's region: clear the effects up to the await's
            # collective scope, settle (drop the region tag from) the rest.
            scope = _await_collective_scope(op.lock)
            for buf in live:
                live[buf] = {
                    (k, b, s, None) if region == op.lock else (k, b, s, region)
                    for (k, b, s, region) in live[buf]
                    if not (region == op.lock and _leq_collective_scope(s, scope))
                }
            continue
        effects = memory_effects(op)
        if effects is None:
            raise NotImplementedError(
                f"unmodelled effect-bearing op: {type(op).__name__}"
            )
        # The barrier this op needs, as the join over its hazards. Each hazard
        # resolves per-pair (:func:`_barrier_scope`) — e.g. an all-thread read vs
        # a t0 (BLOCK_LEADER) write needs only a named barrier. THREAD is the join
        # identity, so it stays THREAD iff no hazard was found.
        barrier: BarrierScope = BarrierScope.THREAD
        for kind_y, buf, scope_y in effects:
            for kind_x, _buf_x, scope_x, _region_x in live.get(buf, ()):
                # A hazard needs a write (RAR is safe).
                if kind_x == "W" or kind_y == "W":
                    barrier = _join_barrier_scope(
                        barrier, _barrier_scope(scope_x, scope_y)
                    )
        targets[op] = barrier
        # A barrier orders settled accesses it covers; pending (in-region) effects
        # survive — only their await completes them.
        for buf in live:
            live[buf] = {
                (k, b, s, region) for (k, b, s, region) in live[buf]
                if region is not None or not _barrier_covers(barrier, s)
            }
        # The async's effects join a completion region keyed by its lock; every
        # other effect-bearing op is synchronous (region = None).
        region = op.lock if isinstance(op, AsyncOp) else None
        for (k, b, s) in effects:
            live.setdefault(b, set()).add((k, b, s, region))
    return live


def _transfer_loop(
    loop: ForOp, live_in: ConcurrentMemoryEffectsType, targets: BarrierTargetsType,
) -> ConcurrentMemoryEffectsType:
    """Analyze a ``for`` body to a fixpoint over its header (the loop-carried
    state at the top of the body), then return the loop's live-out."""
    body = loop.body.entry_block.ops
    while True:
        live_out = _transfer(body, live_in, targets)  # state after one iteration
        joined = _join(live_in, live_out)             # fold the back edge back in
        if joined == live_in:
            return live_out
        live_in = joined


# BarrierScope -> the BarrierOp ``scope`` attr lower_to_ptx_template lowers. Only
# the barriers we actually emit are mapped; the rest (warp/warpgroup, the fan-out
# leader barrier, grid) have no lowering yet.
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


def _analyze(module: ModuleOp) -> BarrierTargetsType:
    """Analyze ``module`` and return the barrier scope each op needs before it
    (the analysis; the dataflow lives in :func:`_transfer`)."""
    kernel = find_op(module, lambda o: isinstance(o, KernelOp))
    targets: BarrierTargetsType = {}
    _transfer(list(kernel.body.entry_block.ops), {}, targets)
    return targets


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        kernel = find_op(module, lambda o: isinstance(o, KernelOp))
        targets = _analyze(module)      # analysis: where do barriers go
        _insert_barriers(kernel.body.entry_block, targets)  # transform: insert them
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
