"""Analyze cross-thread synchronization hazards."""
from __future__ import annotations

import dataclasses
import enum
from typing import cast

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    ClusterBlockIdsOp,
    CollectiveScope,
    ElementwiseOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    LockType,
    MatmulOp,
    MoveOp,
    PTXTemplateOp,
    PtrAddOp,
    ReinterpretOp,
    ReshapeOp,
    ConstantOp,
    SelectOp,
    ThreadIdOp,
    TileOp,
    TmaMoveOp,
    ToPtrOp,
    WarpGroupIdOp,
    memory_effects,
)
from ..base.ir import Op, Value
from .dataflow import DataflowAnalysis, DataflowState, transfers


class BarrierScope(enum.IntEnum):
    r"""The cross-thread barrier to emit for a hazard.

    Mirrors CollectiveScope, but where that names the granularity at which an
    access *executes*, this names the *sync* between those granularities. The two
    correspond one-to-one except for the asymmetric block- and cluster-leader
    syncs::

    Direct cover edges, from wider to narrower scope::

        GRID -> CLUSTER
        CLUSTER -> {BLOCK, CLUSTER_TO_LEADER, LEADER_TO_CLUSTER}
        CLUSTER_TO_LEADER -> BLOCK_TO_LEADER
        BLOCK -> {WARPGROUP, BLOCK_TO_LEADER, LEADER_TO_BLOCK}
        WARPGROUP -> WARP
        {WARP, BLOCK_TO_LEADER, LEADER_TO_BLOCK, LEADER_TO_CLUSTER} -> THREAD

    A cluster-to-leader barrier gathers every CTA's threads at the leaders,
    so it includes a block-to-leader gather. The other directional scopes
    are pairwise independent.
    """
    THREAD = 0
    WARP = 1
    WARPGROUP = 2
    BLOCK_TO_LEADER = 3
    LEADER_TO_BLOCK = 4
    BLOCK = 5
    CLUSTER_TO_LEADER = 6
    LEADER_TO_CLUSTER = 7
    CLUSTER = 8
    GRID = 9

    def join(self, other: BarrierScope) -> BarrierScope:
        """The join (least upper bound) on the BarrierScope lattice."""
        common_upper_bounds = [
            scope for scope in BarrierScope
            if self.leq(scope) and other.leq(scope)
        ]
        [least] = [
            candidate for candidate in common_upper_bounds
            if all(candidate.leq(scope) for scope in common_upper_bounds)
        ]
        return least

    def leq(self, other: BarrierScope) -> bool:
        """The partial order (<=) of the BarrierScope lattice."""
        return self == other or any(
            upper.leq(other)
            for upper in _BARRIER_SCOPE_IMMEDIATE_UPPER_BOUNDS[self]
        )


# The docstring's cover edges, from each scope to the scopes directly above it.
_BARRIER_SCOPE_IMMEDIATE_UPPER_BOUNDS = {
    BarrierScope.THREAD: (
        BarrierScope.WARP,
        BarrierScope.BLOCK_TO_LEADER,
        BarrierScope.LEADER_TO_BLOCK,
        BarrierScope.LEADER_TO_CLUSTER,
    ),
    BarrierScope.WARP: (BarrierScope.WARPGROUP,),
    BarrierScope.WARPGROUP: (BarrierScope.BLOCK,),
    BarrierScope.BLOCK_TO_LEADER: (
        BarrierScope.BLOCK, BarrierScope.CLUSTER_TO_LEADER,
    ),
    BarrierScope.LEADER_TO_BLOCK: (BarrierScope.BLOCK,),
    BarrierScope.BLOCK: (BarrierScope.CLUSTER,),
    BarrierScope.CLUSTER_TO_LEADER: (BarrierScope.CLUSTER,),
    BarrierScope.LEADER_TO_CLUSTER: (BarrierScope.CLUSTER,),
    BarrierScope.CLUSTER: (BarrierScope.GRID,),
    BarrierScope.GRID: (),
}


def _barrier_scope(prior: CollectiveScope, current: CollectiveScope) -> BarrierScope:
    """The barrier scope needed to order a r/w hazard from ``prior`` to ``current``."""
    if prior == current and prior in (
        CollectiveScope.BLOCK_LEADER,
        CollectiveScope.CLUSTER_LEADER,
    ):
        return BarrierScope.THREAD
    if current == CollectiveScope.CLUSTER_LEADER:
        return (
            BarrierScope.GRID
            if prior == CollectiveScope.GRID
            else BarrierScope.CLUSTER_TO_LEADER
        )
    if prior == CollectiveScope.CLUSTER_LEADER:
        return (
            BarrierScope.GRID
            if current == CollectiveScope.GRID
            else BarrierScope.LEADER_TO_CLUSTER
        )
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
    if joined == CollectiveScope.CLUSTER:
        return BarrierScope.CLUSTER
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
    if barrier == BarrierScope.CLUSTER_TO_LEADER:
        return scope == CollectiveScope.THREAD  # gather: nothing cross-thread
    if barrier == BarrierScope.LEADER_TO_CLUSTER:
        # broadcast: the single cluster leader's work, not every block leader
        return scope in (
            CollectiveScope.THREAD,
            CollectiveScope.CLUSTER_LEADER,
        )
    if barrier == BarrierScope.BLOCK:
        return scope not in (
            CollectiveScope.CLUSTER_LEADER,
            CollectiveScope.CLUSTER,
            CollectiveScope.GRID,
        )
    if barrier == BarrierScope.CLUSTER:
        return scope != CollectiveScope.GRID
    if barrier == BarrierScope.GRID:
        return True
    raise NotImplementedError(f"_barrier_covers: unhandled barrier {barrier!r}")


@dataclasses.dataclass(frozen=True)
class ConcurrentMemoryEffect:
    """One outstanding concurrent memory effect on a buffer: an "R"/"W" ``kind``
    at collective ``scope``. Matching non-null ``tiling_scheme`` IDs prove two
    accesses disjoint. ``pending_await`` names the lock that clears the effect,
    or is ``None`` when it is already live for barriers to order."""
    kind: str
    scope: CollectiveScope
    tiling_scheme: int | None
    pending_await: Value | None


@dataclasses.dataclass
class SyncState(DataflowState):
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

    @classmethod
    def empty(cls) -> SyncState:
        return cls()

    def clone(self) -> SyncState:
        """A copy with fresh sets, safe for in-place per-op mutation."""
        return self.join(SyncState.empty())

    def join(self, other: SyncState) -> SyncState:
        """The dataflow join: per-buffer union of effects, union of barrier ids."""
        a, b = self.concurrent_memory_effects, other.concurrent_memory_effects
        p, q = (
            self.named_barrier_ids_pending_await,
            other.named_barrier_ids_pending_await,
        )
        return SyncState(
            {buf: a.get(buf, set()) | b.get(buf, set()) for buf in a.keys() | b.keys()},
            self.named_barrier_ids | other.named_barrier_ids,
            {lock: p.get(lock, set()) | q.get(lock, set())
             for lock in p.keys() | q.keys()},
        )


def _after_barrier(
    live: SyncState, barrier: BarrierScope,
) -> SyncState:
    """Discharge the settled effects ordered by ``barrier``."""
    live = live.clone()
    if barrier in (
        BarrierScope.BLOCK, BarrierScope.CLUSTER, BarrierScope.GRID,
    ):
        live.named_barrier_ids.clear()
        live.named_barrier_ids_pending_await.clear()
    for buf, effects in live.concurrent_memory_effects.items():
        live.concurrent_memory_effects[buf] = {
            effect for effect in effects
            if effect.pending_await is not None
            or not _barrier_covers(barrier, effect.scope)
        }
    return live


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


def _async_completion_scope(op: AsyncOp) -> CollectiveScope:
    """Largest memory-effect scope completed by awaiting ``op``."""
    assert isinstance(op.lock.type, LockType)
    kind = op.lock.type.kind
    if kind == "wgmma":
        return CollectiveScope.WARPGROUP
    if kind == "mbar":
        # A TMA-load mbar await completes the issuing leader's write. Multicast
        # uses one leader for the cluster; ordinary TMA uses one per block.
        return (
            CollectiveScope.CLUSTER_LEADER
            if op.is_tma_multicast
            else CollectiveScope.BLOCK_LEADER
        )
    if kind == "bulk":
        # Bulk async-groups are per issuing thread, not warp-group-wide. TMA
        # stores are issued and committed by t0, so t0's wait completes the
        # leader-owned read of shared memory. This is the completed memory
        # effect's collective scope, not the execution width of wait_group.
        return CollectiveScope.BLOCK_LEADER
    raise NotImplementedError(
        f"_async_completion_scope: unmodeled lock kind {kind!r}"
    )


def _transfer_memory_effects(
    op: Op,
    live: SyncState,
    targets: BarrierTargetsType,
    async_lock: Value | None = None,
) -> SyncState:
    """Transfer one synchronous or asynchronous memory-effect operation."""
    effects = memory_effects(op)
    if effects is None:
        raise NotImplementedError(
            f"unmodelled effect-bearing op: {type(op).__name__}"
        )
    # The barrier this op needs, as the join over its hazards. Each hazard
    # resolves per-pair (:func:`_barrier_scope`).
    barrier_scope: BarrierScope = BarrierScope.THREAD
    for current in effects:
        for root in current.roots:
            for prior in live.concurrent_memory_effects.get(root, ()):
                # Matching scheme IDs prove the runtime accesses pairwise disjoint.
                if (
                    current.tiling_scheme is not None
                    and current.tiling_scheme == prior.tiling_scheme
                ):
                    continue
                # A hazard needs a write (RAR is safe).
                if prior.kind == "W" or current.kind == "W":
                    barrier_scope = barrier_scope.join(
                        _barrier_scope(prior.scope, current.scope)
                    )
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
        assert async_lock is None or isinstance(async_lock.type, LockType)
        if (
            async_lock is not None
            and cast(LockType, async_lock.type).kind == "mbar"
        ):
            live.named_barrier_ids_pending_await.setdefault(
                async_lock, set(),
            ).add(named_id)
        else:
            live.named_barrier_ids.add(named_id)
    targets[op] = BarrierTarget(barrier_scope, named_id)
    # A barrier orders settled accesses it covers; pending async effects survive
    # until their await completes them.
    live = _after_barrier(live, barrier_scope)
    # An async op's effects that its await fully orders wait pending on the lock
    # (the await will clear them). A coarser effect the await only partially
    # orders -- and every synchronous op's effect -- is live immediately, left for
    # a later barrier to order.
    async_scope = (
        _async_completion_scope(op) if isinstance(op, AsyncOp) else None
    )
    for memory_effect in effects:
        pending = (
            async_lock
            if (
                async_scope is not None
                and memory_effect.scope.leq(async_scope)
            )
            else None
        )
        effect = ConcurrentMemoryEffect(
            memory_effect.kind, memory_effect.scope,
            memory_effect.tiling_scheme, pending,
        )
        for root in memory_effect.roots:
            live.concurrent_memory_effects.setdefault(root, set()).add(effect)
    return live


class SyncAnalysis(DataflowAnalysis[SyncState]):
    def __init__(self) -> None:
        super().__init__(SyncState, direction=self.Direction.FORWARD)
        self.targets: BarrierTargetsType = {}

    @transfers(
        BlockIdOp,
        ClusterBlockIdsOp,
        IndexAddOp,
        IndexConstantOp,
        IndexDivOp,
        IndexLtOp,
        IndexModOp,
        IndexMulOp,
        PtrAddOp,
        ReinterpretOp,
        ReshapeOp,
        ConstantOp,
        SelectOp,
        ThreadIdOp,
        TileOp,
        ToPtrOp,
        WarpGroupIdOp,
    )
    def transfer_no_effect(self, op: Op, state: SyncState) -> SyncState:
        return state

    @transfers(PTXTemplateOp)
    def transfer_ptx(self, op: PTXTemplateOp, state: SyncState) -> SyncState:
        # Raw PTX is opaque, so we can't compute its effects. Skip it. This is
        # unsound in general (a template may read/write memory), but it lets
        # hand-written PTX kernels (e.g. fastcu) through on the assumption they
        # self-manage their sync.
        # TODO: gate on a kernel attribute marking the IR hazard-free or
        # already synced instead.
        return state

    @transfers(BarrierOp)
    def transfer_barrier(
        self, op: BarrierOp, state: SyncState,
    ) -> SyncState:
        # Schedulers may author structural barriers before this pass, such as
        # block barriers between persistent-grid waves or grid barriers between
        # kernel phases. Account for their ordering so we do not insert a
        # redundant barrier after them.
        try:
            scope = BarrierScope[op.scope.upper()]
        except KeyError as error:
            raise ValueError(f"unknown barrier scope {op.scope!r}") from error
        return _after_barrier(state, scope)

    @transfers(AwaitOp)
    def transfer_await(self, op: AwaitOp, state: SyncState) -> SyncState:
        # Clear every effect pending on this lock -- only the ones this await
        # fully orders were made pending on it. Effects it only partially orders
        # were never pending and stay live for a barrier.
        for buffer, effects in state.concurrent_memory_effects.items():
            state.concurrent_memory_effects[buffer] = {
                effect for effect in effects
                if effect.pending_await != op.lock
            }
        # An mbar await also completes a block-leader handoff: all threads wait
        # for t0's async work, so no arriving warp can lap the named barrier.
        state.named_barrier_ids_pending_await.pop(op.lock, None)
        return state

    @transfers(AllocOp)
    def transfer_alloc(self, op: AllocOp, state: SyncState) -> SyncState:
        if op.init is None:
            return state
        return _transfer_memory_effects(op, state, self.targets)

    @transfers(AsyncOp)
    def transfer_async(self, op: AsyncOp, state: SyncState) -> SyncState:
        return _transfer_memory_effects(
            op, state, self.targets, async_lock=op.lock,
        )

    @transfers(ElementwiseOp, MatmulOp, MoveOp, TmaMoveOp)
    def transfer_memory_effects(self, op: Op, state: SyncState) -> SyncState:
        return _transfer_memory_effects(op, state, self.targets)
