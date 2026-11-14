"""Insert synchronization barriers identified by synchronization analysis."""

from __future__ import annotations

from typing import cast

from ..analysis.sync import (
    _RELAXABLE_RENDEZVOUS,
    BarrierScope,
    BarrierTargetsType,
    SyncAnalysis,
)
from ..base.ir import Block, Op, Value, find_op, find_ops_of_type, find_ancestor_of_type, find_ancestor
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    DeallocOp,
    ForOp,
    BarrierOp,
    KernelOp,
    LockType,
    ModuleOp,
    TmemTensorType,
    effective_operands,
    execution_issuer_warp,
    lock_is_cluster_scoped,
)


# BarrierScope -> the BarrierOp ``scope`` attr.
_BARRIER_OP_SCOPE: dict[BarrierScope, str] = {
    BarrierScope.BLOCK_TO_LEADER: "block_leader",
    BarrierScope.LEADER_TO_BLOCK: "block",
    BarrierScope.BLOCK: "block",
    BarrierScope.CLUSTER_TO_LEADER: "cluster_to_leader",
    BarrierScope.LEADER_TO_CLUSTER: "leader_to_cluster",
    BarrierScope.CLUSTER: "cluster",
}


def _insert_barriers(
    block: Block,
    targets: BarrierTargetsType,
    num_threads: int | None,
    barrier_syncs: dict[Op, Value] | None = None,
    participant_ids: dict[frozenset[int], int] | None = None,
) -> None:
    """Insert a ``barrier`` before every op the analysis assigned one."""
    new_ops: list[Op] = []
    # Locks initialized above whose publication is still owed, and its scope.
    # Publish once something uses one of them, or a barrier is due anyway.
    pending_locks: set[Value] = set()
    pending_init = BarrierScope.THREAD
    # Arrivals may move up, but never above a publication of their lock.
    published = 0
    for op in block.ops:
        target = targets.get(op)
        if pending_locks and (
            (target is not None and target.scope != BarrierScope.THREAD)
            or not pending_locks.isdisjoint(effective_operands(op))
        ):
            new_ops.extend(_publication_barriers(pending_init))
            published = len(new_ops)
            if (
                target is not None
                and target.scope.leq(pending_init)
                and op not in (barrier_syncs or {})
            ):
                target = None  # The publication already orders this hazard.
            pending_locks.clear()
            pending_init = BarrierScope.THREAD
        if target is not None and target.scope != BarrierScope.THREAD:
            if target.scope not in _BARRIER_OP_SCOPE:
                raise NotImplementedError(
                    f"no lowering for barrier {target.scope.name}"
                )
            sync = (
                barrier_syncs.get(op) if barrier_syncs is not None else None
            )
            # A cluster gather's id selects its mbarrier, the sync operand.
            attrs: dict[str, object] = (
                {} if target.named_id is None or sync is not None
                else {"id": target.named_id}
            )
            if not target.publishes and target.scope in _RELAXABLE_RENDEZVOUS:
                attrs['publishes'] = False
            if target.participant_warps is not None:
                attrs['participant_warps'] = sorted(target.participant_warps)
                attrs['id'] = (participant_ids or {}).get(target.participant_warps, 15)
            if target.scope == BarrierScope.BLOCK_TO_LEADER:
                if target.issuer_warp:
                    attrs['issuer_warp'] = target.issuer_warp
                assert num_threads is not None, (
                    "block-leader barriers require resolved block dimensions"
                )
                if num_threads % 32 != 0:
                    raise ValueError(
                        "block-leader barriers require a whole number of "
                        f"warps; got {num_threads} threads"
                    )
                attrs["num_threads"] = num_threads
            if target.scope == BarrierScope.CLUSTER_TO_LEADER:
                assert sync is not None
                if target.issuer_warp:
                    attrs['issuer_warp'] = target.issuer_warp
                # Split the release: every warp arrives as soon as the
                # accesses it orders are complete, and only the leader waits,
                # just before the op.
                arrival = BarrierOp.build(
                    scope="cluster_to_leader", sync=sync,
                    attrs={**attrs, "phase": "arrive"},
                )
                new_ops.insert(
                    _arrival_index(new_ops, target.release_after, published),
                    arrival,
                )
                attrs = {**attrs, "phase": "wait"}
            new_ops.append(BarrierOp.build(
                scope=_BARRIER_OP_SCOPE[target.scope],
                sync=sync,
                attrs=attrs,
            ))
        new_ops.append(op)
        if (init := _lock_init_barrier(op)) != BarrierScope.THREAD:
            pending_locks.add(op.result())
            pending_init = pending_init.join(init)
        for inner in op.blocks:
            _insert_barriers(inner, targets, num_threads, barrier_syncs, participant_ids)
    new_ops.extend(_publication_barriers(pending_init))
    block.ops = new_ops


def _arrival_index(
    new_ops: list[Op], release_after: frozenset[Op | None], published: int,
) -> int:
    """Where a release may arrive: after the last access it orders completes.

    That access must be earlier in this block, so the arrival and its wait
    run once each per execution of the block; otherwise the arrival stays
    beside its wait.
    """
    positions = [
        next((i for i, op in enumerate(new_ops) if op is after), None)
        for after in release_after
    ]
    if not positions or None in positions:
        return len(new_ops)
    return max(published, max(cast(list[int], positions)) + 1)


def _materialize_gather_mbars(
    kernel: KernelOp,
    targets: BarrierTargetsType,
) -> dict[Op, Value]:
    """Materialize shared counters using the analysis's gather lifetimes.

    Cluster gathers always use mbarriers; block gathers may opt in. Sites
    share a counter only after the analysis proves its prior handoff ended.
    """
    barrier_syncs: dict[Op, Value] = {}
    allocs: dict[object, AllocOp] = {}
    mbar_block_gathers = kernel.attrs.get("block_gather_backend", "named") == "mbar"
    for op, target in targets.items():
        is_block = target.scope == BarrierScope.BLOCK_TO_LEADER and mbar_block_gathers
        if target.scope != BarrierScope.CLUSTER_TO_LEADER and not is_block:
            continue
        # A site without an id keeps an mbarrier of its own.
        masks = None
        if not is_block and isinstance(op, AsyncOp) and op.lock.type.kind == 'tcgen05' and op.lock.type.cta_group == 2:
            from ..cluster_layout import cluster_size, pair_mask
            masks = tuple(pair_mask(r) for r in range(cluster_size(kernel)))
        if not is_block and isinstance(op, AsyncOp) and op.lock.type.kind == 'mbar':
            from ..cluster_layout import cluster_size
            from ..gpu_dialect import TmaMoveOp, MoveOp
            size = cluster_size(kernel)
            moves = [move for move in op.body.ops if isinstance(move, (TmaMoveOp, MoveOp))]
            if moves and any('multicast_cta_masks' in move.attrs for move in moves):
                masks = tuple(0 for _ in range(size))
                for move in moves:
                    sharing = move.attrs.get('multicast_cta_masks',
                        [(1 << size)-1]*size if move.multicast else [1 << r for r in range(size)])
                    masks = tuple(a | b for a, b in zip(masks, sharing, strict=True))
        key = op if target.named_id is None else (target.scope, target.named_id, masks)
        if key not in allocs:
            allocs[key] = AllocOp.build_with_fresh_id(
                LockType.build(kind="mbar", attrs={"arrivals": kernel.num_threads // 32})
                if is_block else LockType.build(kind="cluster_mbar",
                    attrs={'release_cta_masks': list(masks)} if masks is not None else None),
            )
        barrier_syncs[op] = allocs[key].result()
    kernel.body.ops[0:0] = list(allocs.values())
    return barrier_syncs


def _lock_init_barrier(op: Op) -> BarrierScope:
    """Publication barrier required by an implicitly initialized lock."""
    lock = op.result() if isinstance(op, AllocOp) else None
    if lock is None or not isinstance(lock.type, LockType):
        return BarrierScope.THREAD
    if lock_is_cluster_scoped(lock):
        return BarrierScope.CLUSTER
    return BarrierScope.BLOCK if lock.type.is_event else BarrierScope.THREAD


def _publication_barriers(scope: BarrierScope) -> list[Op]:
    """A CTA barrier, followed by a cluster rendezvous for cluster scope."""
    scopes = {
        BarrierScope.THREAD: [],
        BarrierScope.BLOCK: [BarrierScope.BLOCK],
        BarrierScope.CLUSTER: [BarrierScope.BLOCK, BarrierScope.CLUSTER],
    }[scope]
    return [BarrierOp.build(_BARRIER_OP_SCOPE[s], attrs={"initialization_publication": True}) for s in scopes]


def drop_vacuous_publications(block: Block) -> None:
    """Remove initialization publications that no longer publish a lock.

    Lowerings after InsertSyncs may replace the events whose initialization
    a publication was placed for (publishing their own). Once the originals
    are dead, such a barrier still rendezvouses the CTA, or the cluster,
    before work that does not need it.
    """
    owed = BarrierScope.THREAD
    kept: list[Op] = []
    for op in block.ops:
        if isinstance(op, BarrierOp) and op.attrs.get("initialization_publication"):
            scope = BarrierScope.CLUSTER if op.scope == "cluster" else BarrierScope.BLOCK
            if owed == BarrierScope.THREAD:
                continue
            if not owed.leq(scope):
                kept.append(op)  # The CTA half of a cluster publication.
                continue
            owed = BarrierScope.THREAD
        elif (init := _lock_init_barrier(op)) != BarrierScope.THREAD:
            owed = owed.join(init)
        kept.append(op)
    block.ops = kept


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        # Initialize each completion event at its issuing warp. Publication
        # still uses the ordinary CTA barrier before observers access it.
        for alloc in find_ops_of_type(module, AllocOp):
            lock = alloc.result()
            if not isinstance(lock.type, LockType) or not lock.type.is_event:
                continue
            issuers = {execution_issuer_warp(user) for user in lock.users() if isinstance(user, AsyncOp)}
            if len(issuers) > 1:
                raise ValueError('a completion lock requires one issuing warp')
            if issuers and (issuer := next(iter(issuers))):
                alloc.attrs['issuer_warp'] = issuer
        kernel = find_op(module, lambda op: isinstance(op, KernelOp))
        assert isinstance(kernel, KernelOp)
        repeated_tmem = not kernel.is_clustered and kernel.has_repeated_tmem_allocation
        backend = kernel.attrs.get('block_gather_backend') or ('mbar' if repeated_tmem else 'named')
        if backend == 'named' and repeated_tmem:
            raise ValueError('named gathers require a reusable TMEM allocation')
        if backend == 'mbar':
            kernel.attrs['block_gather_backend'] = backend
        analysis = SyncAnalysis()
        analysis.analyze(module)
        barrier_syncs = _materialize_gather_mbars(kernel, analysis.targets)
        # Subset rendezvous must not reuse an ID held by an asynchronous
        # gather or by a different participant set.
        reserved = {t.named_id for t in analysis.targets.values() if t.named_id is not None}
        reserved |= {b.attrs.get('id', 0) for b in find_ops_of_type(kernel, BarrierOp) if b.scope == 'block'}
        groups = sorted({t.participant_warps for t in analysis.targets.values() if t.participant_warps is not None}, key=lambda ws: tuple(sorted(ws)))
        available = sorted(set(range(1, 16)) - reserved, reverse=True)
        if len(groups) > len(available):
            raise ValueError('participant rendezvous exceeds available named barriers')
        participant_ids = dict(zip(groups, available))
        # Transform: insert the barriers with their proven participant counts.
        _insert_barriers(
            module.body, analysis.targets, kernel.num_threads, barrier_syncs, participant_ids,
        )
        allocs = find_ops_of_type(kernel, AllocOp)
        if (not kernel.is_clustered and any(isinstance(a.result().type, TmemTensorType) for a in allocs)
                or kernel.attrs.get('block_gather_backend') == 'mbar' or kernel.attrs.get('reuse_completion_locks')
                or any(a.attrs.get('issuer_warp', 0) for a in allocs)):
            retirements: dict[Block, list[DeallocOp]] = {}
            for alloc in allocs:
                lock = alloc.result()
                users = lock.users()
                if (not isinstance(lock.type, LockType) or not lock.type.is_event
                        or find_ancestor(alloc, lambda parent: isinstance(parent, ForOp)
                                         and parent.trip_count > 1) is None
                        or not any(isinstance(user, AsyncOp) for user in users)
                        or any(isinstance(user, DeallocOp) for user in users)):
                    continue
                block = alloc.parent_block
                assert block is not None and block.parent_op is not None
                if any(find_ancestor(user, lambda parent: parent is block.parent_op) is None for user in users):
                    raise ValueError('a repeated completion lock cannot escape its allocation scope')
                release = DeallocOp.build(lock)
                if issuer := execution_issuer_warp(alloc):
                    release.attrs['issuer_warp'] = issuer
                retirements.setdefault(block, []).append(release)
            for block, releases in retirements.items():
                # All observers and asynchronous producers must retire before
                # an owner invalidates an event that the next iteration resets.
                if not isinstance(block.ops[-1], BarrierOp) or block.ops[-1].scope != 'block':
                    block.ops.append(BarrierOp.build(attrs={'completion_lifetime_release': True}))
                block.ops.extend(releases)
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
