"""Insert cross-thread synchronization barriers on shared-buffer hazards."""
from __future__ import annotations

import dataclasses
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

    def join(self, other: BarrierScope) -> BarrierScope:
        """The join (least upper bound) on the BarrierScope lattice."""
        if self == other:
            return self
        if self == BarrierScope.THREAD:
            return other
        if other == BarrierScope.THREAD:
            return self
        if self == BarrierScope.GRID or other == BarrierScope.GRID:
            return BarrierScope.GRID
        if self == BarrierScope.BLOCK or other == BarrierScope.BLOCK:
            return BarrierScope.BLOCK
        if {self, other} == {BarrierScope.WARP, BarrierScope.WARPGROUP}:
            return BarrierScope.WARPGROUP
        return BarrierScope.BLOCK

    def leq(self, other: BarrierScope) -> bool:
        """The partial order (<=) of the BarrierScope lattice."""
        return self.join(other) == other


def _barrier_scope(prior: CollectiveScope, current: CollectiveScope) -> BarrierScope:
    """The barrier scope needed to order a r/w hazard from ``prior`` to ``current``."""
    if prior == current == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.THREAD
    if current == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.BLOCK_TO_LEADER
    if prior == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.LEADER_TO_BLOCK
    # Neither side is a leader: the symmetric barrier for the coarser scope.
    joined = prior.join(current)
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


@dataclasses.dataclass(frozen=True)
class ConcurrentMemoryEffect:
    """One outstanding concurrent memory effect on a buffer: an "R"/"W" ``kind``
    at collective ``scope``, with a ``pending_await`` on the lock that clears it
    (``None`` if it's already live for barriers to order)."""
    kind: str
    scope: CollectiveScope
    pending_await: Value | None


@dataclasses.dataclass
class DataflowState:
    """The forward analysis state: outstanding concurrent memory effects keyed by
    root buffer, named-barrier ids in use until a full-block sync, and ids in use
    until an await on a particular lock."""
    concurrent_memory_effects: dict[object, set[ConcurrentMemoryEffect]] = (
        dataclasses.field(default_factory=dict)
    )
    named_barrier_ids: set[int] = dataclasses.field(default_factory=set)
    named_barrier_ids_pending_await: dict[Value, set[int]] = (
        dataclasses.field(default_factory=dict)
    )

    def clone(self) -> DataflowState:
        """A copy with fresh sets, safe for in-place per-op mutation."""
        return self.join(DataflowState())  # union with empty -> fresh sets

    def join(self, other: DataflowState) -> DataflowState:
        """The dataflow join: per-buffer union of effects, union of barrier ids."""
        a, b = self.concurrent_memory_effects, other.concurrent_memory_effects
        p, q = (
            self.named_barrier_ids_pending_await,
            other.named_barrier_ids_pending_await,
        )
        return DataflowState(
            {buf: a.get(buf, set()) | b.get(buf, set()) for buf in a.keys() | b.keys()},
            self.named_barrier_ids | other.named_barrier_ids,
            {lock: p.get(lock, set()) | q.get(lock, set())
             for lock in p.keys() | q.keys()},
        )


@dataclasses.dataclass(frozen=True)
class BarrierTarget:
    """The barrier before an op: its ``scope`` and, for a block-leader barrier,
    the named-barrier ``named_id`` to lower to (``None`` otherwise)."""
    scope: BarrierScope
    named_id: int | None = None


# The analysis result: each effect-bearing op mapped to the barrier before it.
BarrierTargetsType = dict["Op", "BarrierTarget"]


# PTX exposes 16 barrier resources per CTA. ID 0 is reserved for the full-block
# ``bar.sync 0`` emitted by the lowering, leaving IDs 1..15 for named barriers.
_MAX_BARRIER_IDS = 16


def _async_collective_scope(lock: Value) -> CollectiveScope:
    """The collective scope an async op on ``lock`` synchronizes when awaited."""
    kind = lock.type.kind
    if kind == "wgmma":
        return CollectiveScope.WARPGROUP
    if kind == "tcgen05":
        # One elected thread issues the MMA, but its destination is CTA-owned
        # tensor memory and every participating thread waits on the commit.
        return CollectiveScope.BLOCK
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
    raise NotImplementedError(f"_async_collective_scope: unmodeled lock kind {kind!r}")


def _analyze_block(
    ops: list[Op], live_in: DataflowState, targets: BarrierTargetsType,
) -> DataflowState:
    live = live_in.clone()
    for op in ops:
        live = _analyze(op, live, targets)
    return live


def _analyze(
    op: Op, live: DataflowState, targets: BarrierTargetsType,
) -> DataflowState:
    """Transfer the concurrent-effect map across one ``op``, recording the barrier
    it needs in ``targets``; return the live-out. Container ops recurse into their
    body; a ``for`` is iterated to a fixpoint over its header."""
    if isinstance(op, (ModuleOp, KernelOp)):
        # A module/kernel body is a fresh scope -- no live effects flow in, and
        # its internal effects don't escape to the enclosing scope.
        _analyze_block(list(op.body.ops), DataflowState(), targets)
        return live
    if isinstance(op, ForOp):
        # Analyze the body to a fixpoint over its header (the loop-carried state
        # at the top of the body); return the loop's live-out.
        header = live
        while True:
            live_out = _analyze_block(op.body.ops, header, targets)
            joined = header.join(live_out)  # fold the back edge
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
        # Clear every effect pending on this lock -- only the ones this await fully
        # orders were made pending on it (see the async handling below). Effects it
        # only partially orders were never pending and stay live for a barrier.
        for buf, effs in live.concurrent_memory_effects.items():
            live.concurrent_memory_effects[buf] = {
                e for e in effs if e.pending_await != op.lock
            }
        # An mbar await also completes a block-leader handoff: all threads wait
        # for t0's async work, so no arriving warp can lap the named barrier.
        live.named_barrier_ids_pending_await.pop(op.lock, None)
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
        for e in live.concurrent_memory_effects.get(buf, ()):
            # A hazard needs a write (RAR is safe).
            if e.kind == "W" or kind_y == "W":
                barrier_scope = barrier_scope.join(_barrier_scope(e.scope, scope_y))
    # A block-leader barrier is named. Its id remains in use until either a
    # full-block sync, or (for an mbar async) the await where everyone waits for
    # t0. Reuse any id whose prior lifetime has ended.
    named_id = None
    if barrier_scope == BarrierScope.BLOCK_TO_LEADER:
        recorded = targets.get(op)
        pending_await_ids = {
            barrier_id
            for ids in live.named_barrier_ids_pending_await.values()
            for barrier_id in ids
        }
        used_ids = live.named_barrier_ids | pending_await_ids
        if recorded is not None and recorded.named_id is not None:
            named_id = recorded.named_id
            # TODO: Insert or upgrade to a full-block sync when a loop backedge
            # keeps this named barrier live, instead of rejecting the program.
            assert named_id not in used_ids, (
                f"named barrier {named_id} remains in use across a loop "
                f"backedge before {op}; named barriers are only released by a "
                "full-block sync or their mbar await, so add a full-block sync "
                "before the backedge; "
                f"pending awaits={live.named_barrier_ids_pending_await}"
            )
        if named_id is None:
            # A barrier can first become necessary on a later loop-fixpoint
            # iteration. Do not let it steal an id already assigned to another
            # static barrier site that has not been revisited yet.
            reserved_ids = (
                {
                    target.named_id
                    for target_op, target in targets.items()
                    if target_op is not op and target.named_id is not None
                }
                if recorded is not None else set()
            )
            available_ids = (
                set(range(1, _MAX_BARRIER_IDS)) - used_ids - reserved_ids
            )
            if not available_ids:
                raise ValueError(
                    "exhausted named barrier IDs: PTX provides 16 IDs and ID 0 "
                    "is reserved for full-block barriers"
                )
            named_id = min(available_ids)
        # BLOCK_TO_LEADER also applies to leader-issued bulk TMA stores. Only
        # an mbar await is an all-thread rendezvous that ends this handoff.
        if isinstance(op, AsyncOp) and op.lock.type.kind == "mbar":
            live.named_barrier_ids_pending_await.setdefault(
                op.lock, set(),
            ).add(named_id)
        else:
            live.named_barrier_ids.add(named_id)
    elif barrier_scope == BarrierScope.BLOCK:
        live.named_barrier_ids = set()
        live.named_barrier_ids_pending_await = {}
    targets[op] = BarrierTarget(barrier_scope, named_id)
    # A barrier orders settled accesses it covers; pending (in-block) effects
    # survive — only their await completes them.
    for buf, effs in live.concurrent_memory_effects.items():
        live.concurrent_memory_effects[buf] = {
            e for e in effs
            if e.pending_await is not None
            or not _barrier_covers(barrier_scope, e.scope)
        }
    # An async op's effects that its await fully orders wait pending on the lock
    # (the await will clear them). A coarser effect the await only partially
    # orders -- and every synchronous op's effect -- is live immediately, left for
    # a later barrier to order.
    async_scope = (
        _async_collective_scope(op.lock) if isinstance(op, AsyncOp) else None
    )
    for (k, b, s) in effects:
        pending = op.lock if async_scope is not None and s.leq(async_scope) else None
        effect = ConcurrentMemoryEffect(k, s, pending)
        live.concurrent_memory_effects.setdefault(b, set()).add(effect)
    return live


# BarrierScope -> the BarrierOp ``scope`` attr.
_BARRIER_OP_SCOPE: dict[BarrierScope, str] = {
    BarrierScope.BLOCK_TO_LEADER: "block_leader",
    BarrierScope.BLOCK: "block",
}


def _insert_barriers(
    block: Block,
    targets: BarrierTargetsType,
    num_threads: int,
) -> None:
    """Insert a ``barrier`` before every op the analysis assigned one."""
    new_ops: list[Op] = []
    for op in block.ops:
        target = targets.get(op)
        if target is not None and target.scope != BarrierScope.THREAD:
            if target.scope not in _BARRIER_OP_SCOPE:
                raise NotImplementedError(
                    f"no lowering for barrier {target.scope.name}"
                )
            attrs = {} if target.named_id is None else {"id": target.named_id}
            if target.scope == BarrierScope.BLOCK_TO_LEADER:
                if num_threads % 32 != 0:
                    raise ValueError(
                        "block-leader barriers require a whole number of "
                        f"warps; got {num_threads} threads"
                    )
                attrs["num_threads"] = num_threads
            new_ops.append(
                BarrierOp.build(scope=_BARRIER_OP_SCOPE[target.scope], attrs=attrs)
            )
        new_ops.append(op)
        for inner in op.blocks:
            _insert_barriers(inner, targets, num_threads)
    block.ops = new_ops


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        targets: BarrierTargetsType = {}
        _analyze(module, DataflowState(), targets)  # analysis: where do barriers go
        kernel = find_op(module, lambda op: isinstance(op, KernelOp))
        assert isinstance(kernel, KernelOp)
        # Transform: insert the barriers with the kernel-wide participant count.
        _insert_barriers(module.body, targets, kernel.num_threads)
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
