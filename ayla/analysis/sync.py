"""Analyze cross-thread synchronization hazards."""

from __future__ import annotations

import dataclasses
import enum
from collections import Counter
from typing import cast

from ..gpu_dialect import (
    SmemTensorType,
    AllocOp,
    OperandPipelineOp,
    PersistentPipelineOp,
    DeallocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    ClusterBlockIdsOp,
    CollectiveScope,
    ElementwiseOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
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
    TmemTensorType,
    ToPtrOp,
    WarpGroupIdOp,
    WarpIdsOp,
    memory_effects,
    root_buffers,
    memory_regions_may_overlap,
    ordered_async_roots,
    releases_cluster_banks,
    lock_is_cluster_scoped,
    execution_issuer_warp,
    execution_warps,
)
from ..base.ir import Op, Value, find_ancestor_of_type, iter_ops
from .dataflow import DataflowAnalysis, DataflowState, transfers
from .index_proofs import uniform_guard_scope


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

    A cluster-to-leader barrier gathers every warp of the cluster at the
    leaders, so it includes a block-to-leader gather but no block barrier.

    ``CLUSTER_TO_LEADER -> BLOCK_TO_LEADER`` is a choice: we define the
    cluster gather as two-level, because one cluster mbarrier arrival per CTA
    is what the hardware offers. Each CTA first gathers its threads at its
    block leader, then the block leaders arrive at the cluster leader. That
    first level is a block-to-leader gather, so a group that needs both, such
    as fused local and multicast TMA loads, is covered by one cluster gather
    instead of a full cluster barrier. The broadcast is not the mirror image:
    ``LEADER_TO_CLUSTER`` publishes only the cluster leader's work, and block
    leaders release nothing in it, so it does not include ``LEADER_TO_BLOCK``.
    The other directional scopes are pairwise independent.
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
        def upper_bounds(scope: BarrierScope) -> set[BarrierScope]:
            bounds = {scope}
            for upper in _BARRIER_SCOPE_IMMEDIATE_UPPER_BOUNDS[scope]:
                bounds |= upper_bounds(upper)
            return bounds

        common = upper_bounds(self) & upper_bounds(other)
        [least] = [scope for scope in common if upper_bounds(scope) == common]
        return least

    def leq(self, other: BarrierScope) -> bool:
        """The partial order (<=) of the BarrierScope lattice."""
        return self == other or any(
            upper.leq(other) for upper in _BARRIER_SCOPE_IMMEDIATE_UPPER_BOUNDS[self]
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
        BarrierScope.BLOCK,
        BarrierScope.CLUSTER_TO_LEADER,
    ),
    BarrierScope.LEADER_TO_BLOCK: (BarrierScope.BLOCK,),
    BarrierScope.BLOCK: (BarrierScope.CLUSTER,),
    BarrierScope.CLUSTER_TO_LEADER: (BarrierScope.CLUSTER,),
    BarrierScope.LEADER_TO_CLUSTER: (BarrierScope.CLUSTER,),
    BarrierScope.CLUSTER: (BarrierScope.GRID,),
    BarrierScope.GRID: (),
}


def _barrier_scope(
    prior: CollectiveScope, current: CollectiveScope,
    prior_issuer: int = 0, current_issuer: int = 0,
) -> BarrierScope:
    """The barrier scope needed to order a r/w hazard from ``prior`` to ``current``."""
    if prior == current and prior in (
        CollectiveScope.BLOCK_LEADER,
        CollectiveScope.CLUSTER_LEADER,
    ):
        if prior_issuer == current_issuer:
            return BarrierScope.THREAD
        # Separate leaders do not have implicit program ordering. A full
        # rendezvous provides publication without a new hardware primitive.
        return (BarrierScope.BLOCK if prior == CollectiveScope.BLOCK_LEADER
                else BarrierScope.CLUSTER)
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
    if joined.name not in BarrierScope.__members__:
        raise NotImplementedError(f"_barrier_scope: unhandled join {joined!r}")
    return BarrierScope[joined.name]


def _barrier_covers(barrier: BarrierScope, scope: CollectiveScope) -> bool:
    """Whether ``barrier`` orders a prior effect at collective ``scope`` -- i.e.
    the effect is discharged from the live set once the barrier is placed."""
    if barrier == BarrierScope.THREAD:
        return scope == CollectiveScope.THREAD
    if barrier == BarrierScope.WARP:
        return scope in (CollectiveScope.THREAD, CollectiveScope.WARP)
    if barrier == BarrierScope.WARPGROUP:
        return scope in (
            CollectiveScope.THREAD,
            CollectiveScope.WARP,
            CollectiveScope.WARPGROUP,
        )
    if barrier == BarrierScope.BLOCK_TO_LEADER:
        return scope == CollectiveScope.THREAD  # gather: nothing cross-thread
    if barrier == BarrierScope.LEADER_TO_BLOCK:
        # broadcast: t0's own (BLOCK_LEADER) work only, not all-thread BLOCK
        return scope in (CollectiveScope.THREAD, CollectiveScope.BLOCK_LEADER)
    if barrier == BarrierScope.CLUSTER_TO_LEADER:
        # gather: every warp of the cluster arrives, and only the leaders wait
        return scope == CollectiveScope.THREAD
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


# A gather's barrier resource: its scope and its id among that scope's ids.
BarrierId = tuple[BarrierScope, int]


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
    # The async lock whose await completes the access (``None`` for a
    # synchronous access), and the op after which the access is complete,
    # once known.
    lock: Value | None = None
    completed_after: "Op | None" = None
    memory_region: tuple[int, int] | None = None
    issuer_warp: int | None = None
    # An asymmetric gather orders independent thread effects only at its
    # destination issuer. Keep the observation for any other producer.
    gathered_by: frozenset[int] = frozenset()
    # A split cluster gather publishes captured accesses to the issuing warp
    # in every CTA. Local warp visibility alone cannot prove that ordering.
    cluster_gathered_by: frozenset[int] = frozenset()
    # A completed effect may have been acquired by only these warps. None
    # retains the existing all-participant visibility for settled effects.
    visible_to: frozenset[int] | None = None
    # An explicit release arrival can relay acquired effects through another
    # completion event. A new issue starts with an empty publication set.
    published_by: frozenset[Value] = frozenset()
    observed_lock: Value | None = None
    # Paired transfer completion proves both CTA issuers passed these phase reads.
    phase_acked_by: frozenset[Value] = frozenset()
    participant_warps: frozenset[int] | None = None
    # A later same-issuer tcgen05 commit covers preceding MMA data accesses.
    # Event reads/parity and software handoff leases retain their own lifetimes.
    completed_by: frozenset[Value] = frozenset()
    path_guards: frozenset[tuple[Value, bool, CollectiveScope]] = frozenset()


@dataclasses.dataclass
class SyncState(DataflowState):
    """The forward analysis state: outstanding concurrent memory effects keyed by
    root buffer, gather barrier ids in use until a rendezvous that includes the
    gather, ids in use until an await on a particular lock, and where each
    released id's last handoff ended."""

    concurrent_memory_effects: dict[object, set[ConcurrentMemoryEffect]] = (
        dataclasses.field(default_factory=dict)
    )
    named_barrier_ids: set[BarrierId] = dataclasses.field(default_factory=set)
    named_barrier_ids_pending_await: dict[Value, set[BarrierId]] = dataclasses.field(
        default_factory=dict
    )
    named_barrier_ids_released_after: dict[BarrierId, set[Op]] = dataclasses.field(
        default_factory=dict
    )
    actor_tracking: bool = False
    gather_publications: dict[Value, set[BarrierId]] = dataclasses.field(default_factory=dict)

    path_guards: frozenset[tuple[Value, bool, CollectiveScope]] = frozenset()
    unreachable: bool = False

    @classmethod
    def empty(cls) -> SyncState:
        return cls()

    def clone(self) -> SyncState:
        """A copy with fresh sets, safe for in-place per-op mutation."""
        result = self.join(SyncState.empty())
        result.path_guards = self.path_guards
        result.unreachable = self.unreachable
        return result

    def join(self, other: SyncState) -> SyncState:
        """The dataflow join: per-buffer union of effects, union of barrier ids."""
        if self.unreachable:
            return other.clone() if not other.unreachable else dataclasses.replace(self)
        if other.unreachable:
            return self.clone()
        a, b = self.concurrent_memory_effects, other.concurrent_memory_effects
        p, q = (
            self.named_barrier_ids_pending_await,
            other.named_barrier_ids_pending_await,
        )
        r, t = (
            self.named_barrier_ids_released_after,
            other.named_barrier_ids_released_after,
        )
        def merge_effects(left, right):
            if not left or not right:
                return left | right
            effects = left | right
            # An effect with fewer completion witnesses subsumes an otherwise
            # identical effect with more witnesses: every wait that retires
            # the former also retires the latter. Keep the minimal frontier
            # alternatives. Optional MMA issues would otherwise enumerate
            # every subset of their completion events across branch joins.
            groups = {}
            for effect in effects:
                key = dataclasses.replace(effect, completed_by=frozenset())
                groups.setdefault(key, []).append(effect)
            result = set()
            for alternatives in groups.values():
                minimal = []
                for effect in sorted(alternatives, key=lambda e: len(e.completed_by)):
                    if not any(frontier <= effect.completed_by for frontier in minimal):
                        minimal.append(effect.completed_by)
                        result.add(effect)
            return result

        return SyncState(
            {buf: merge_effects(a.get(buf, set()), b.get(buf, set()))
             for buf in a.keys() | b.keys()},
            self.named_barrier_ids | other.named_barrier_ids,
            {
                lock: p.get(lock, set()) | q.get(lock, set())
                for lock in p.keys() | q.keys()
            },
            {key: r.get(key, set()) | t.get(key, set()) for key in r.keys() | t.keys()},
            self.actor_tracking or other.actor_tracking,
            {lock: self.gather_publications.get(lock, set()) | other.gather_publications.get(lock, set())
             for lock in self.gather_publications.keys() | other.gather_publications.keys()},
            self.path_guards & other.path_guards,
        )


# The barriers at which every participating thread waits for every other.
_RENDEZVOUS = (BarrierScope.BLOCK, BarrierScope.CLUSTER, BarrierScope.GRID)
# Lowered as barrier.cluster arrive/wait, whose arrive may omit the release.
_RELAXABLE_RENDEZVOUS = (BarrierScope.CLUSTER, BarrierScope.LEADER_TO_CLUSTER)


def _after_barrier(
    live: SyncState,
    barrier: BarrierScope,
    at: Op,
    issuer_warp: int = 0,
    participant_warps: frozenset[int] | None = None,
    reads_only: bool = False,
) -> SyncState:
    """Discharge the settled effects ordered by ``barrier`` before ``at``.

    A ``reads_only`` rendezvous has no release: it orders completed reads
    before later writes, but publishes no write."""
    live = live.clone()
    if barrier == BarrierScope.THREAD:
        # No cross-thread synchronization is emitted at this scope. Keep
        # settled accesses (including event parity observations) visible to
        # later producers and consumers, even across unrelated operations.
        return live
    if participant_warps is not None:
        # A subset rendezvous publishes only settled accesses owned by that
        # subset. It cannot retire an unrelated full-CTA handoff.
        for root, effects in live.concurrent_memory_effects.items():
            updated = set()
            for effect in effects:
                observers = effect.visible_to or effect.participant_warps
                if (effect.pending_await is None and observers is not None and observers <= participant_warps
                        and effect.scope != CollectiveScope.GRID):
                    visible = participant_warps | {issuer_warp} if barrier in (
                        BarrierScope.BLOCK_TO_LEADER, BarrierScope.CLUSTER_TO_LEADER) else participant_warps
                    effect = dataclasses.replace(effect, visible_to=visible)
                updated.add(effect)
            live.concurrent_memory_effects[root] = updated
        return live
    if barrier in _RENDEZVOUS:
        # A rendezvous ends every handoff through a gather it includes.
        ended = {
            key
            for key in live.named_barrier_ids.union(
                *live.named_barrier_ids_pending_await.values(),
            )
            if key[0].leq(barrier)
        }
        live.named_barrier_ids -= ended
        for ids in live.named_barrier_ids_pending_await.values():
            ids -= ended
        for ids in live.gather_publications.values():
            ids -= ended
        for key in ended:
            live.named_barrier_ids_released_after[key] = {at}
    for buf, effects in live.concurrent_memory_effects.items():
        if live.actor_tracking and barrier == BarrierScope.BLOCK_TO_LEADER:
            effects = {
                dataclasses.replace(effect, visible_to=effect.visible_to | {issuer_warp})
                if effect.pending_await is None and effect.visible_to is not None else effect
                for effect in effects
            }
            live.concurrent_memory_effects[buf] = {
                dataclasses.replace(effect, gathered_by=effect.gathered_by | {issuer_warp})
                if effect.pending_await is None and _barrier_covers(barrier, effect.scope)
                else effect
                for effect in effects
            }
            continue
        live.concurrent_memory_effects[buf] = {
            effect
            for effect in effects
            if effect.pending_await is not None
            or not _barrier_covers(barrier, effect.scope)
            or (reads_only and effect.kind == "W")
        }
    return live


@dataclasses.dataclass(frozen=True)
class BarrierTarget:
    """The barrier before an op: its ``scope`` and, for a gather, the
    ``named_id`` of the barrier resource to lower to (``None`` otherwise): a
    named barrier for a block-leader gather, a cluster mbarrier for a
    cluster-to-leader gather."""

    scope: BarrierScope
    named_id: int | None = None
    # For a cluster-to-leader barrier: where each access it orders is
    # complete (``None`` where that is unknown). Its arrival may move up to
    # the last of them; only its wait must stay before the op.
    release_after: frozenset["Op | None"] = frozenset()
    issuer_warp: int = 0
    participant_warps: frozenset[int] | None = None
    # Whether a hazard it orders follows a write. A cluster rendezvous that
    # only orders reads before writes needs no release.
    publishes: bool = True


# The analysis result: each effect-bearing op mapped to the barrier before it.
BarrierTargetsType = dict["Op", "BarrierTarget"]


# PTX exposes 16 barrier resources per CTA. ID 0 is reserved for the full-block
# ``bar.sync 0`` emitted by the lowering, leaving IDs 1..15 for named barriers.
_MAX_BARRIER_IDS = 16

# The ids each gather draws its barrier resource from. Cluster mbarriers are
# allocated in shared memory, so they have no fixed limit.
_GATHER_BARRIER_IDS: dict[BarrierScope, range | None] = {
    BarrierScope.BLOCK_TO_LEADER: range(1, _MAX_BARRIER_IDS),
    BarrierScope.CLUSTER_TO_LEADER: None,
}


def _lock_completion_scope(lock: Value) -> CollectiveScope:
    """Baseline completion scope for the work named by this hardware lock."""
    assert isinstance(lock.type, LockType)
    kind = lock.type.kind
    if kind == "tcgen05":
        return CollectiveScope.CLUSTER if lock_is_cluster_scoped(lock) else CollectiveScope.BLOCK
    if kind == "wgmma":
        return CollectiveScope.WARPGROUP
    if kind == "mbar":
        return CollectiveScope.CLUSTER_LEADER if lock_is_cluster_scoped(lock) else CollectiveScope.BLOCK_LEADER
    if kind == "bulk":
        return CollectiveScope.BLOCK_LEADER
    raise NotImplementedError(f"unmodeled completion mechanism: {kind!r}")


def _await_completion_scope(op: AwaitOp) -> CollectiveScope:
    """Completion proven at this wait, rather than predicted at its producer.

    A partial wait completes its listed lock at the mechanism's normal scope,
    leaving newer groups pending. A full-warpgroup wait covers block memory when the
    resolved block contains exactly that one group. This does not release
    resources that require a block rendezvous.
    """
    scope = _lock_completion_scope(op.lock)
    if not op.is_full_wait:
        return scope
    if scope != CollectiveScope.WARPGROUP:
        return scope
    kernel = find_ancestor_of_type(op, KernelOp)
    if kernel is None or kernel.is_clustered:
        return scope
    if kernel.resolved_num_warp_groups != 1:
        return scope
    return CollectiveScope.BLOCK


def _transfer_memory_effects(
    op: Op,
    live: SyncState,
    targets: BarrierTargetsType,
    async_lock: Value | None = None,
    ordered_peer_cache: dict[tuple[Value, int], bool] | None = None,
) -> SyncState:
    """Transfer one synchronous or asynchronous memory-effect operation."""
    if live.unreachable:
        return live
    effects = memory_effects(op)
    if effects is None:
        raise NotImplementedError(f"unmodelled effect-bearing op: {type(op).__name__}")
    # The barrier this op needs, as the join over its hazards. Each hazard
    # resolves per-pair (:func:`_barrier_scope`).
    barrier_scope: BarrierScope = BarrierScope.THREAD
    publishes = False
    release_after: set[Op | None] = set()
    hazard_actors: list[frozenset[int] | None] = []
    ordered = ordered_async_roots(op) if async_lock is not None else set()
    kernel = find_ancestor_of_type(op, KernelOp)
    ordering_lock = async_lock
    ordering_issuer = execution_issuer_warp(op)
    if isinstance(op, OperandPipelineOp) and releases_cluster_banks(op):
        from ..passes.pipeline_rolled_k import _parts
        _, batch = _parts(op)
        ordered = ordered_async_roots(batch)
        ordering_lock = batch.lock
        ordering_issuer = execution_issuer_warp(batch)
    if (ordered and async_lock is not None and async_lock.type.kind == 'tcgen05' and kernel is not None
            and kernel.attrs.get('ordered_mma_completion')):
        issuer = execution_issuer_warp(op)
        if ordered_peer_cache is None:
            ordered_peer_cache = {}
        for root, prior_effects in live.concurrent_memory_effects.items():
            updated = set()
            for prior in prior_effects:
                frontier = prior.completed_by - {async_lock}
                if (not isinstance(getattr(root, 'type', None), LockType)
                        and ('completion_cta_masks' not in async_lock.type.attrs or isinstance(root.type, TmemTensorType))
                        and prior.lock is not None and prior.lock.type.kind == 'tcgen05'
                        and prior.lock.type.cta_group == async_lock.type.cta_group):
                    key = (prior.lock, issuer)
                    if key not in ordered_peer_cache:
                        peers = [p for p in prior.lock.users() if isinstance(p, AsyncOp)]
                        ordered_peer_cache[key] = bool(peers) and all(
                            ordered_async_roots(p) and execution_issuer_warp(p) == issuer
                            for p in peers
                        )
                    if ordered_peer_cache[key]:
                        frontier |= {event for event in op.completion_locks
                            if event.type.attrs.get('completion_role') != 'result'
                            or isinstance(root.type, TmemTensorType)} if isinstance(op, AsyncOp) else {async_lock}
                updated.add(dataclasses.replace(prior, completed_by=frontier))
            live.concurrent_memory_effects[root] = updated
    for current in effects:
        for root in current.roots:
            for prior in live.concurrent_memory_effects.get(root, ()):
                if not memory_regions_may_overlap(current.memory_region, prior.memory_region):
                    continue
                if (
                    root in ordered
                    and ordering_lock is not None
                    and prior.lock is not None
                    and cast(LockType, prior.lock.type).kind == cast(LockType, ordering_lock.type).kind
                    and any(
                        root in ordered_async_roots(peer)
                        and ordering_issuer == execution_issuer_warp(peer)
                        for peer in prior.lock.users()
                        if isinstance(peer, AsyncOp)
                    )
                ):
                    continue  # Same engine orders these accumulator updates.
                # Matching scheme IDs prove the runtime accesses pairwise disjoint.
                if (
                    current.tiling_scheme is not None
                    and current.tiling_scheme == prior.tiling_scheme
                ):
                    continue
                # A hazard needs a write (RAR is safe).
                if prior.kind == "W" or current.kind == "W":
                    actors = current.participant_warps
                    observers = prior.visible_to
                    if (actors is not None and prior.pending_await is None
                            and observers is not None and actors <= observers
                            and current.scope != CollectiveScope.GRID and prior.scope != CollectiveScope.GRID):
                        continue
                    if (actors is not None and actors == prior.participant_warps
                            and prior.pending_await is None and prior.scope == CollectiveScope.THREAD
                            and current.scope == CollectiveScope.THREAD):
                        continue
                    before = prior.visible_to or prior.participant_warps
                    hazard_actors.append(before | actors if before is not None and actors is not None
                        and prior.pending_await is None
                        and prior.scope not in (CollectiveScope.CLUSTER, CollectiveScope.CLUSTER_LEADER, CollectiveScope.GRID)
                        and current.scope not in (CollectiveScope.CLUSTER, CollectiveScope.CLUSTER_LEADER, CollectiveScope.GRID)
                        else None)
                    leader_async = (async_lock is not None
                                    and cast(LockType, async_lock.type).kind in ("mbar", "tcgen05", "bulk"))
                    current_issuer = execution_issuer_warp(op) if leader_async else current.issuer_warp or 0
                    if prior.pending_await is None and prior.visible_to is not None:
                        if (leader_async and 'completion_cta_masks' in async_lock.type.attrs and root is async_lock
                                and prior.observed_lock is root and prior.visible_to == frozenset({0})
                                and current_issuer == 0):
                            # A counted new phase cannot complete until every
                            # contributing issuer has observed this old phase.
                            # PipelineClusterBanks verifies the operand-bank
                            # wait/refill/MMA dependency for all contributors.
                            continue
                        paired_leader_issue = (leader_async and kernel is not None
                                              and kernel.attrs.get('correlate_uniform_guards')
                                              and async_lock.type.cta_group == 2 and async_lock.type.is_event)
                        if (paired_leader_issue and root is async_lock and root.type.kind=='mbar'
                                and prior.visible_to==frozenset({current_issuer})
                                and (prior.lock is root or prior.observed_lock is root)
                                and all(execution_issuer_warp(peer)==current_issuer for peer in root.users() if isinstance(peer,AsyncOp))
                                and all(peer.waiter_warps==(current_issuer,) for peer in root.users() if isinstance(peer,AwaitOp))):
                            continue
                        if (paired_leader_issue and prior.scope == CollectiveScope.CLUSTER_LEADER
                                and prior.issuer_warp == current_issuer and current_issuer in prior.visible_to):
                            continue
                        leader_access = current.scope == CollectiveScope.BLOCK_LEADER or leader_async
                        cluster_access = current.scope in (CollectiveScope.CLUSTER_LEADER, CollectiveScope.CLUSTER)
                        if (cluster_access and leader_async
                                and current_issuer in prior.cluster_gathered_by):
                            continue
                        if not cluster_access and leader_access and current_issuer in prior.visible_to:
                            continue
                        # Ordering starts at the actual observers, not the
                        # asynchronous engine's original issuing warp. Engine
                        # effects retain their physical scopes; an elected
                        # issuer only needs publication before starting them.
                        scope = (BarrierScope.CLUSTER_TO_LEADER if current.scope == CollectiveScope.CLUSTER_LEADER
                                 else BarrierScope.CLUSTER if cluster_access
                                 else BarrierScope.BLOCK_TO_LEADER if leader_access else BarrierScope.BLOCK)
                        barrier_scope = barrier_scope.join(scope)
                        publishes |= prior.kind == "W"
                        if scope == BarrierScope.CLUSTER_TO_LEADER:
                            release_after.add(prior.completed_after)
                        continue
                    if (live.actor_tracking and prior.pending_await is None
                        and prior.scope == CollectiveScope.THREAD
                        and current.scope == CollectiveScope.BLOCK_LEADER
                        and current_issuer in prior.gathered_by):
                        continue
                    scope = _barrier_scope(prior.scope, current.scope,
                        prior.issuer_warp or 0, current_issuer)
                    barrier_scope = barrier_scope.join(scope)
                    if scope != BarrierScope.THREAD:
                        publishes |= prior.kind == "W"
                    if scope == BarrierScope.CLUSTER_TO_LEADER:
                        release_after.add(prior.completed_after)
    recorded = targets.get(op)
    if recorded is not None:
        barrier_scope = barrier_scope.join(recorded.scope)
        publishes |= recorded.publishes
    participants = None
    if (hazard_actors and all(a is not None for a in hazard_actors)
            and BarrierScope.THREAD < barrier_scope <= BarrierScope.BLOCK):
        participants = frozenset().union(*hazard_actors)
        if recorded is not None and recorded.scope != BarrierScope.THREAD:
            participants = None if recorded.participant_warps is None else participants | recorded.participant_warps
        if participants is not None:
            barrier_scope = BarrierScope.BLOCK
    # A gather's barrier resource remains in use until either a rendezvous
    # that includes the gather, or (for an mbar async) the await where everyone
    # waits for the leaders' work. Reuse any id whose prior lifetime has ended.
    named_id = None
    if barrier_scope in _GATHER_BARRIER_IDS:
        used_ids = {
            barrier_id
            for scope, barrier_id in live.named_barrier_ids.union(
                *live.named_barrier_ids_pending_await.values(),
            )
            if scope == barrier_scope
        }
        if (
            recorded is not None
            and recorded.named_id is not None
            and recorded.scope == barrier_scope
        ):
            named_id = recorded.named_id
            if named_id in used_ids:
                # The same static site has reached us through a loop backedge
                # before its barrier was released. Upgrade the site to the
                # least rendezvous including the gather, which both performs
                # the gather and releases its barriers for the next iteration.
                barrier_scope = next(
                    scope for scope in _RENDEZVOUS if barrier_scope.leq(scope)
                )
                named_id = None
        if barrier_scope in _GATHER_BARRIER_IDS and named_id is None:
            # A barrier can first become necessary on a later loop-fixpoint
            # iteration. Do not let it steal an id already assigned to another
            # static barrier site that has not been revisited yet.
            reserved_ids = (
                {
                    target.named_id
                    for target_op, target in targets.items()
                    if target_op is not op
                    and target.named_id is not None
                    and target.scope == barrier_scope
                }
                if recorded is not None
                else set()
            )
            ids = _GATHER_BARRIER_IDS[barrier_scope] or range(
                len(used_ids) + len(reserved_ids) + 1,
            )
            available_ids = set(ids) - used_ids - reserved_ids
            if not available_ids:
                # A full rendezvous performs the gather and retires its live
                # named handoffs. It also works when role/stage choices need
                # more hardware counters than a CTA provides.
                barrier_scope = next(
                    scope for scope in _RENDEZVOUS if barrier_scope.leq(scope)
                )
            else:
                named_id = min(available_ids)
        # BLOCK_TO_LEADER also applies to leader-issued bulk TMA stores. Only
        # an mbar await is an all-thread rendezvous that ends this handoff.
        if named_id is not None:
            key = (barrier_scope, named_id)
            assert async_lock is None or isinstance(async_lock.type, LockType)
            if (
                async_lock is not None
                and cast(LockType, async_lock.type).is_event
            ):
                live.named_barrier_ids_pending_await.setdefault(
                    async_lock,
                    set(),
                ).add(key)
            else:
                live.named_barrier_ids.add(key)
            # A cluster release arrives early: never before the end of the
            # id's previous handoff.
            release_after |= live.named_barrier_ids_released_after.pop(
                key,
                set(),
            )
    if barrier_scope == BarrierScope.CLUSTER_TO_LEADER:
        recorded = targets.get(op)
        if recorded is not None:
            release_after |= recorded.release_after
    else:
        release_after = set()
    targets[op] = BarrierTarget(
        barrier_scope,
        named_id,
        frozenset(release_after),
        execution_issuer_warp(op) if barrier_scope in (BarrierScope.BLOCK_TO_LEADER, BarrierScope.CLUSTER_TO_LEADER) else 0,
        participants,
        publishes,
    )
    # A barrier orders settled accesses it covers; pending async effects survive
    # until their await completes them. A cluster rendezvous lowers with a
    # release only when it publishes a write.
    live = _after_barrier(live, barrier_scope, op, execution_issuer_warp(op), participants,
        reads_only=not publishes and barrier_scope in _RELAXABLE_RENDEZVOUS)
    if async_lock is not None and op.attrs.get("release_prior"):
        issuer = execution_issuer_warp(op)
        publications = live.gather_publications.setdefault(async_lock, set())
        for root, prior_effects in live.concurrent_memory_effects.items():
            updated = set()
            for prior in prior_effects:
                if (prior.pending_await is None and prior.visible_to is not None
                        and issuer in prior.visible_to):
                    prior = dataclasses.replace(prior, published_by=prior.published_by | {async_lock})
                    dependency = prior.observed_lock or prior.lock
                    if dependency is not None:
                        publications |= live.named_barrier_ids_pending_await.get(dependency, set())
                updated.add(prior)
            live.concurrent_memory_effects[root] = updated
    # Defer accesses covered by the mechanism's baseline scope until its await.
    # Wider effects remain hazards until the actual wait proves completion at
    # their scope, or a barrier orders them.
    async_scope = _lock_completion_scope(async_lock) if async_lock is not None else None
    for memory_effect in effects:
        event = async_lock
        additional = frozenset()
        if isinstance(op, AsyncOp) and len(op.completion_locks) > 1:
            if isinstance(memory_effect.value.type, LockType):
                # Completion of another event is not a phase observation of
                # this one. Each event retains its own reuse dependency.
                event = memory_effect.value
            else:
                additional = frozenset(event for event in op.completion_locks[1:]
                    if event.type.attrs.get('completion_role') != 'result'
                    or isinstance(memory_effect.value.type, TmemTensorType))
        pending = (
            event
            if (async_scope is not None and memory_effect.scope.leq(async_scope))
            else None
        )
        effect = ConcurrentMemoryEffect(
            memory_effect.kind,
            memory_effect.scope,
            memory_effect.tiling_scheme,
            pending,
            lock=event,
            completed_by=additional,
            completed_after=None if async_lock is not None else op,
            issuer_warp=memory_effect.issuer_warp,
            memory_region=memory_effect.memory_region,
            participant_warps=memory_effect.participant_warps,
            path_guards=frozenset(fact for fact in live.path_guards if memory_effect.scope.leq(fact[2])),
        )
        for root in memory_effect.roots:
            live.concurrent_memory_effects.setdefault(root, set()).add(dataclasses.replace(
                effect, path_guards=frozenset(fact for fact in effect.path_guards
                if _guard_effect_scope(root, effect.scope).leq(fact[2]))))
    return live


def _guard_effect_scope(root, scope):
    if not isinstance(root.type, LockType):
        return scope
    cluster = root.type.cta_group == 2 or root.type.kind == 'cluster_mbar'
    return scope.join(CollectiveScope.CLUSTER if cluster else CollectiveScope.BLOCK)


class SyncAnalysis(DataflowAnalysis[SyncState]):
    def __init__(self) -> None:
        super().__init__(SyncState, direction=self.Direction.FORWARD)
        self.targets: BarrierTargetsType = {}
        self._guard_scopes = {}
        # The IR remains fixed during one analysis. Cache the structural peer
        # proof across effects and loop iterations; this changes no ordering.
        self._ordered_peer_cache: dict[tuple[Value, int], bool] = {}
        self._bank_kernels: dict[KernelOp, bool] = {}

    def boundary_state(self, op: Op) -> SyncState:
        state = SyncState.empty()
        if isinstance(op, KernelOp):
            self._guard_scopes = {}
            if op.attrs.get('correlate_uniform_guards'):
                counts = Counter(region.condition for region in iter_ops(op)
                                 if isinstance(region,IfOp))
                for condition,count in counts.items():
                    if count < 2:
                        continue
                    scope = uniform_guard_scope(condition,op)
                    if scope is not None:
                        self._guard_scopes[condition] = scope
            state.actor_tracking = any(o.attrs.get('issuer_warp', 0) != 0
                                       or isinstance(o, AwaitOp) and o.waiter_warps is not None
                                       for o in iter_ops(op))
        return state

    def branch_state(self, op, state, taken):
        scope=self._guard_scopes.get(op.condition)
        if scope is None:
            return state
        state=state.clone()
        facts={v:(truth,s) for v,truth,s in state.path_guards}
        if op.condition in facts and facts[op.condition][0]!=taken:
            state.unreachable=True
            return state
        facts[op.condition]=(taken,scope)
        state.path_guards=frozenset((v,t,s) for v,(t,s) in facts.items())
        for root,effects in state.concurrent_memory_effects.items():
            state.concurrent_memory_effects[root]={dataclasses.replace(effect,path_guards=effect.path_guards|
                frozenset(fact for fact in state.path_guards if _guard_effect_scope(root, effect.scope).leq(fact[2])))
                for effect in effects if all(v not in facts or facts[v][0]==t for v,t,s in effect.path_guards)}
        return state

    def _forget_loop_guards(self,op,state):
        state=state.clone()
        local=set(iter_ops(op))
        def outside(facts):return frozenset((v,t,s) for v,t,s in facts if v.definition not in local)
        state.path_guards=outside(state.path_guards)
        for root,effects in state.concurrent_memory_effects.items():
            state.concurrent_memory_effects[root]={dataclasses.replace(effect,path_guards=outside(effect.path_guards)) for effect in effects}
        return state

    loop_backedge=_forget_loop_guards
    loop_exit=_forget_loop_guards

    @transfers(
        AllocOp,
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
        WarpIdsOp,
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
        self,
        op: BarrierOp,
        state: SyncState,
    ) -> SyncState:
        if state.unreachable:
            return state
        # Schedulers may author structural barriers before this pass, such as
        # block barriers between persistent-grid waves or grid barriers between
        # kernel phases. Account for their ordering so we do not insert a
        # redundant barrier after them.
        try:
            scope = (BarrierScope.BLOCK_TO_LEADER if op.scope == 'block_leader'
                     else BarrierScope[op.scope.upper()])
        except KeyError as error:
            raise ValueError(f"unknown barrier scope {op.scope!r}") from error
        if op.attrs.get("phase") == "arrive":
            kernel = find_ancestor_of_type(op, KernelOp)
            participants = execution_warps(op)
            all_warps = participants is None or (
                kernel is not None and participants == frozenset(range(kernel.num_threads // 32))
            )
            if op.scope in ('block_leader', 'cluster_to_leader') and all_warps:
                # Capture settled thread accesses/phase observations and
                # completed collective TMEM reads preceding this arrival.
                # The later wait cannot publish accesses after arrival.
                for root, effects in state.concurrent_memory_effects.items():
                    state.concurrent_memory_effects[root] = {
                        dataclasses.replace(effect, published_by=effect.published_by | {op.sync})
                        if (effect.pending_await is None and (
                            _barrier_covers(scope, effect.scope)
                            or (isinstance(root.type, TmemTensorType) and effect.kind == 'R'
                                and isinstance(effect.completed_after, MoveOp)
                                and isinstance(effect.completed_after.src.type, TmemTensorType)
                                and execution_warps(effect.completed_after) is None)))
                        else effect for effect in effects
                    }
            return state  # Publication is acquired at the matching wait.
        if op.scope in ('block_leader', 'cluster_to_leader') and op.attrs.get('phase') == 'wait':
            issuer = execution_issuer_warp(op)
            unpublished = {}
            for root, effects in state.concurrent_memory_effects.items():
                unpublished[root] = {effect for effect in effects if op.sync not in effect.published_by}
                state.concurrent_memory_effects[root] = {
                    dataclasses.replace(effect,
                        visible_to=((effect.visible_to or frozenset()) | {issuer}
                                    if isinstance(root.type, TmemTensorType) else effect.visible_to),
                        cluster_gathered_by=(effect.cluster_gathered_by | {issuer}
                                            if op.scope == 'cluster_to_leader'
                                            else effect.cluster_gathered_by),
                        published_by=effect.published_by - {op.sync})
                    for effect in effects if op.sync in effect.published_by
                }
            state = _after_barrier(state, scope, op, issuer)
            for root, effects in unpublished.items():
                state.concurrent_memory_effects[root] |= effects
            return state
        participants = execution_warps(op)
        if scope == BarrierScope.CLUSTER_TO_LEADER and op.sync is not None:
            raw = op.sync.type.attrs.get("release_warps")
            if raw is not None:
                participants = frozenset(raw)
        return _after_barrier(state, scope, op, execution_issuer_warp(op), participants)

    @transfers(AwaitOp)
    def transfer_await(self, op: AwaitOp, state: SyncState) -> SyncState:
        if state.unreachable:
            return state
        scope = _await_completion_scope(op)
        waiters = op.waiter_warps
        kernel = find_ancestor_of_type(op, KernelOp)
        local_completion = (waiters is None and kernel is not None and kernel.is_clustered
                            and scope.leq(CollectiveScope.BLOCK))
        local_observers = frozenset(range(kernel.num_threads // 32)) if local_completion else frozenset()
        settle = cast(LockType, op.lock.type).kind == "bulk"
        if local_completion and settle:
            # A bulk group completes only in its issuing thread's queue.
            local_observers = frozenset(execution_issuer_warp(peer) for peer in op.lock.users()
                                        if isinstance(peer, AsyncOp))
        bank_release = 'completion_cta_masks' in op.lock.type.attrs and waiters == (0,)
        if kernel is not None and kernel not in self._bank_kernels:
            self._bank_kernels[kernel] = releases_cluster_banks(kernel)
        bank_ready = bool(kernel is not None and self._bank_kernels[kernel]
                          and op.lock.type.kind == 'mbar' and waiters == (0,))
        paired_acquire = bool(kernel is not None and kernel.cluster_dim == (2, 1, 1)
            and kernel.attrs.get('paired_completion_handoffs') and waiters is not None
            and not op.attrs.get('waiter_cta_ranks') and op.lock.type.cta_group == 2 and op.lock.type.is_event)
        released_banks = set()
        if bank_release:
            for peer in op.lock.users():
                if isinstance(peer, AsyncOp):
                    for mma in peer.body.ops:
                        released_banks |= root_buffers(mma.lhs) | root_buffers(mma.rhs)
        for buffer, effects in state.concurrent_memory_effects.items():
            remaining = set()
            for effect in effects:
                if (bank_release and isinstance(buffer.type, LockType) and buffer.type.kind == 'mbar'
                        and effect.pending_await is None):
                    writers = [peer for peer in buffer.users() if isinstance(peer, AsyncOp)]
                    written_banks = set().union(*(root_buffers(move.dst) for peer in writers for move in peer.body.ops))
                    if written_banks and written_banks <= released_banks:
                        # MMA completion proves every consumer observed this
                        # bank's ready phase before issuing its MMA.
                        continue
                if (kernel is not None and kernel.attrs.get('paired_phase_relay')
                        and op.lock in effect.phase_acked_by
                        and op.lock.type.kind == 'mbar' and op.lock.type.cta_group == 2
                        and (effect.observed_lock is buffer
                            or paired_acquire and effect.lock is buffer)
                        and isinstance(buffer.type, LockType) and buffer.type.kind == 'tcgen05'
                        and effect.visible_to == frozenset(execution_issuer_warp(peer)
                            for peer in op.lock.users() if isinstance(peer, AsyncOp))
                        and (frozenset(waiters) == frozenset(execution_issuer_warp(peer)
                            for peer in buffer.users() if isinstance(peer, AsyncOp))
                            if paired_acquire else waiters == (0,))
                        and effect.pending_await is None):
                    # Both CTA leaders issue this paired transfer only after
                    # their bank-completion phase observations. Transaction
                    # completion proves both issuers passed those observations.
                    continue
                if effect.pending_await is None and op.lock in effect.published_by:
                    if waiters is None:
                        continue  # This event includes an explicit release arrival.
                    # The release publishes preceding issuer observations to
                    # every acquiring consumer, including a bank-refill warp.
                    # Other warps still owe their own completion observation.
                    remaining.add(dataclasses.replace(effect,
                        visible_to=(effect.visible_to or frozenset()) | frozenset(waiters)))
                    continue
                if effect.lock != op.lock and op.lock not in effect.completed_by:
                    remaining.add(effect)
                    continue
                if (waiters is not None or local_completion) and effect.scope.leq(scope):
                    visibility = (effect.visible_to or frozenset()) | (frozenset(waiters) if waiters is not None else local_observers)
                    remaining.add(dataclasses.replace(effect, pending_await=None,
                        completed_after=effect.completed_after or op, visible_to=visibility,
                        # A paired engine completion acquired in both CTAs
                        # already covers the engine effects at cluster scope.
                        # It publishes only effects named by this generation.
                        cluster_gathered_by=effect.cluster_gathered_by | (
                            frozenset(waiters) if paired_acquire
                            else {0} if bank_release or bank_ready else set())))
                    continue
                if not settle and effect.scope.leq(scope):
                    continue
                remaining.add(dataclasses.replace(effect,
                    pending_await=None if settle else effect.pending_await,
                    completed_after=effect.completed_after or op))
            state.concurrent_memory_effects[buffer] = remaining
        if cast(LockType, op.lock.type).is_event:
            observations = [None] if waiters is None else [frozenset({warp}) for warp in waiters]
            for visibility in observations:
                state.concurrent_memory_effects.setdefault(op.lock, set()).add(
                    ConcurrentMemoryEffect("R", CollectiveScope.THREAD, None, None,
                        completed_after=op, visible_to=visibility,
                        observed_lock=op.lock if waiters is not None else None,
                        participant_warps=visibility, path_guards=frozenset(
                            fact for fact in state.path_guards
                            if _guard_effect_scope(op.lock, CollectiveScope.THREAD).leq(fact[2]))))
        if waiters is None:
            released = state.named_barrier_ids_pending_await.pop(op.lock, set())
            publications = state.gather_publications.pop(op.lock, set())
            released |= publications
            # Only an explicit relay can end another event's handoff. Keep
            # legacy lease ownership unchanged for ordinary completion waits.
            for ids in state.named_barrier_ids_pending_await.values():
                ids -= publications
            state.named_barrier_ids -= publications
            # Captured IDs are leases, not permanent dependencies. Once a
            # lease ends, another captured completion cannot retire its reuse.
            for ids in state.gather_publications.values():
                ids -= released
            for key in released:
                state.named_barrier_ids_released_after[key] = {op}
        return state

    @transfers(AsyncOp)
    def transfer_async(self, op: AsyncOp, state: SyncState) -> SyncState:
        kernel = find_ancestor_of_type(op, KernelOp)
        if kernel is not None and kernel.attrs.get('paired_phase_relay'):
            state = self._relay_paired_phases(op, state, kernel)
        return _transfer_memory_effects(op, state, self.targets,
            async_lock=op.lock, ordered_peer_cache=self._ordered_peer_cache)

    def _relay_paired_phases(self, op: AsyncOp, state: SyncState, kernel: KernelOp) -> SyncState:
        relay = (op.lock.type.kind == 'mbar' and op.lock.type.cta_group == 2
            and not op.is_tma_multicast and bool(op.body.ops)
            and all(isinstance(m, TmaMoveOp) and isinstance(m.dst.type, SmemTensorType)
                    for m in op.body.ops))
        for root, effects in state.concurrent_memory_effects.items():
            updated = set()
            for effect in effects:
                acknowledgments = effect.phase_acked_by - {op.lock}
                if (relay and isinstance(root.type, LockType) and root.type.kind == 'tcgen05'
                        and root.type.cta_group == 2
                        and (effect.observed_lock is root
                            or kernel.attrs.get('paired_completion_handoffs') and effect.lock is root)
                        and effect.pending_await is None
                        and effect.visible_to == frozenset({execution_issuer_warp(op)})
                        and effect.completed_after is not None and not effect.completed_after.attrs.get("waiter_cta_ranks")
                        and isinstance(effect.completed_after, AwaitOp)
                        and effect.completed_after.waiter_warps == (execution_issuer_warp(op),)):
                    acknowledgments |= {op.lock}
                updated.add(dataclasses.replace(effect, phase_acked_by=acknowledgments))
            state.concurrent_memory_effects[root] = updated
        if (kernel.attrs.get('paired_completion_handoffs')
                and op.lock.type.kind == 'tcgen05' and op.lock.type.cta_group == 2
                and op.body.ops and all(isinstance(m, MatmulOp) for m in op.body.ops)):
            issuer = execution_issuer_warp(op)
            read_banks = set().union(*(root_buffers(m.lhs) | root_buffers(m.rhs)
                for m in op.body.ops))
            for root, effects in state.concurrent_memory_effects.items():
                covered = root in read_banks
                if isinstance(root.type, LockType) and root.type.kind == 'mbar':
                    peers = [p for p in root.users() if isinstance(p, AsyncOp)]
                    written = set().union(*(root_buffers(m.dst) for p in peers
                        for m in p.body.ops if isinstance(m, TmaMoveOp)))
                    covered = bool(written) and written <= read_banks
                if not covered:
                    continue
                # The issuer acquired these input writes/ready observations
                # before the MMA. Its completion is a transitive witness for
                # those settled effects, including at the refill issuer.
                # Pending loads and unrelated banks retain their dependencies.
                state.concurrent_memory_effects[root] = {
                    dataclasses.replace(e, completed_by=e.completed_by | {op.lock})
                    if (e.pending_await is None and e.visible_to is not None
                        and issuer in e.visible_to and (e.observed_lock is root
                            or e.lock is not None and e.lock.type.kind == 'mbar'))
                    else e for e in effects}
        return state

    @transfers(OperandPipelineOp)
    def transfer_operand_pipeline(self, op: OperandPipelineOp, state: SyncState) -> SyncState:
        state = _transfer_memory_effects(op, state, self.targets)
        if releases_cluster_banks(op):
            # LowerRolledK drains its hardware events and executes a full
            # cluster rendezvous at this completed-region boundary. Model
            # that actual rendezvous instead of inserting another outside it.
            state = _after_barrier(state, BarrierScope.CLUSTER, op)
        return state

    @transfers(PersistentPipelineOp)
    def transfer_persistent_pipeline(self, op, state):
        state = _transfer_memory_effects(op, state, self.targets)
        kernel = find_ancestor_of_type(op, KernelOp)
        # The implementation drains stores and all live operand epochs, then
        # rendezvouses before returning the arena to its caller.
        scope = BarrierScope.CLUSTER if kernel.is_clustered else BarrierScope.BLOCK
        return _after_barrier(state, scope, op)

    @transfers(DeallocOp, ElementwiseOp, MatmulOp, MoveOp, TmaMoveOp)
    def transfer_memory_effects(self, op: Op, state: SyncState) -> SyncState:
        return _transfer_memory_effects(op, state, self.targets)
