"""Insert cross-thread synchronization barriers on shared-buffer hazards."""
from __future__ import annotations

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


def _barrier_scope(prior: CollectiveScope, current: CollectiveScope) -> CollectiveScope:
    """The barrier scope needed to order a r/w hazard from ``prior`` to ``current``.

    TODO: the result is really a *barrier* scope (which sync to emit), a separate
    but similar lattice from the collective scope (how the effect executes), even
    though both reuse CollectiveScope. Notably the barrier scopes are asymmetric --
    the named leader barrier orders only one direction -- which the collective
    lattice doesn't capture. Consider a distinct lattice for barrier scopes."""
    if prior == current == CollectiveScope.BLOCK_LEADER:
        return CollectiveScope.THREAD
    if prior == CollectiveScope.BLOCK and current == CollectiveScope.BLOCK_LEADER:
        return CollectiveScope.BLOCK_LEADER
    return _join_collective_scope(prior, current)


# The dataflow state: outstanding concurrent memory effects per root buffer, each
# a (kind, root_buffer, scope, region) tuple. ``region`` is the lock of the open
# async completion region the effect belongs to (its ``await`` will close it), or
# ``None`` for synchronous effects.
ConcurrentMemoryEffectsType = dict[
    object, set[tuple[str, object, "CollectiveScope", object]]
]

# The analysis result: each effect-bearing op mapped to the coarsest scope its
# accesses must be synced at.
BarrierTargetsType = dict["Op", "CollectiveScope"]


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
    if kind == "bulk":
        # Bulk async-groups are per issuing thread, not warp-group-wide. TMA
        # stores are issued and committed by t0, so t0's wait completes the
        # leader-owned read of shared memory. This is the completed memory
        # effect's collective scope, not the execution width of wait_group.
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
        # The barrier this op needs, as the lattice join over its hazards. Each
        # hazard resolves per-pair (:func:`_barrier_scope`) — e.g. an all-thread
        # read vs a t0 (BLOCK_LEADER) write needs only a named barrier. THREAD is
        # the join-identity, so it stays THREAD iff no hazard was found.
        barrier_scope: CollectiveScope = CollectiveScope.THREAD
        for kind_y, buf, scope_y in effects:
            for kind_x, _buf_x, scope_x, _region_x in live.get(buf, ()):
                # A hazard needs a write (RAR is safe).
                if kind_x == "W" or kind_y == "W":
                    barrier_scope = _join_collective_scope(
                        barrier_scope, _barrier_scope(scope_x, scope_y)
                    )
        targets[op] = barrier_scope
        # A barrier orders settled accesses it covers; pending (in-region) effects
        # survive — only their await completes them.
        for buf in live:
            live[buf] = {
                (k, b, s, region) for (k, b, s, region) in live[buf]
                if region is not None or not _leq_collective_scope(s, barrier_scope)
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


def _insert_barriers(block: Block, targets: BarrierTargetsType) -> None:
    """Insert a ``barrier`` before every op that needs a cross-thread sync
    (scope above THREAD)."""
    new_ops: list[Op] = []
    for op in block.ops:
        scope = targets.get(op, CollectiveScope.THREAD)
        if scope != CollectiveScope.THREAD:
            new_ops.append(BarrierOp.build(scope=scope.name.lower()))
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
