"""Insert synchronization barriers identified by synchronization analysis."""

from __future__ import annotations

from ..analysis.sync import (
    BarrierScope,
    BarrierTargetsType,
    SyncAnalysis,
)
from ..base.ir import Block, Op, Value, find_op
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    BarrierOp,
    KernelOp,
    LockType,
    ModuleOp,
    effective_operands,
    lock_has_tma_multicast,
)


# BarrierScope -> the BarrierOp ``scope`` attr.
_BARRIER_OP_SCOPE: dict[BarrierScope, str] = {
    BarrierScope.BLOCK_TO_LEADER: "block_leader",
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
) -> None:
    """Insert a ``barrier`` before every op the analysis assigned one."""
    new_ops: list[Op] = []
    # Locks initialized above whose publication is still owed, and its scope.
    # Publish once something uses one of them, or a barrier is due anyway.
    pending_locks: set[Value] = set()
    pending_init = BarrierScope.THREAD
    for op in block.ops:
        target = targets.get(op)
        if pending_locks and (
            (target is not None and target.scope != BarrierScope.THREAD)
            or not pending_locks.isdisjoint(effective_operands(op))
        ):
            new_ops.extend(_publication_barriers(pending_init))
            # The CTA barrier of a publication orders memory as well. Its
            # cluster rendezvous orders only the initialization.
            if (
                target is not None
                and target.scope.leq(pending_init)
                and target.scope.leq(BarrierScope.BLOCK)
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
            attrs: dict[str, object] = (
                {} if target.named_id is None else {"id": target.named_id}
            )
            if target.scope == BarrierScope.BLOCK_TO_LEADER:
                assert num_threads is not None, (
                    "block-leader barriers require resolved block dimensions"
                )
                if num_threads % 32 != 0:
                    raise ValueError(
                        "block-leader barriers require a whole number of "
                        f"warps; got {num_threads} threads"
                    )
                attrs["num_threads"] = num_threads
            sync = (
                barrier_syncs.get(op) if barrier_syncs is not None else None
            )
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
            _insert_barriers(inner, targets, num_threads, barrier_syncs)
    new_ops.extend(_publication_barriers(pending_init))
    block.ops = new_ops


def _materialize_cluster_mbars(
    kernel: KernelOp,
    targets: BarrierTargetsType,
) -> dict[Op, Value]:
    """Allocate one cluster mbarrier per cluster-to-leader barrier site."""
    barrier_syncs: dict[Op, Value] = {}
    allocs: list[AllocOp] = []
    # TODO: Track cluster-mbarrier lifetimes and reuse an allocation after a
    # cluster/grid rendezvous proves that every CTA completed its prior use.
    # Until then, allocate one mbarrier per static barrier site.
    for op, target in targets.items():
        if target.scope != BarrierScope.CLUSTER_TO_LEADER:
            continue
        alloc = AllocOp.build_with_fresh_id(
            LockType.build(kind="cluster_mbar"),
        )
        allocs.append(alloc)
        barrier_syncs[op] = alloc.result()
    kernel.body.ops[0:0] = allocs
    return barrier_syncs


def _lock_init_barrier(op: Op) -> BarrierScope:
    """Publication barrier required by an implicitly initialized lock."""
    lock = op.result() if isinstance(op, AllocOp) else None
    if lock is None or not isinstance(lock.type, LockType):
        return BarrierScope.THREAD
    if lock.type.kind == "cluster_mbar" or lock_has_tma_multicast(lock):
        return BarrierScope.CLUSTER
    return BarrierScope.BLOCK if lock.type.kind == "mbar" else BarrierScope.THREAD


def _publication_barriers(scope: BarrierScope) -> list[Op]:
    """A CTA barrier, followed by a cluster rendezvous for cluster scope."""
    scopes = {
        BarrierScope.THREAD: [],
        BarrierScope.BLOCK: [BarrierScope.BLOCK],
        BarrierScope.CLUSTER: [BarrierScope.BLOCK, BarrierScope.CLUSTER],
    }[scope]
    # Each initializing thread already releases its initialization to the
    # cluster with fence.mbarrier_init, so the rendezvous is a publication:
    # it orders no other memory and lowers without a fence.
    return [
        BarrierOp.build(
            _BARRIER_OP_SCOPE[s],
            attrs={"publication": True} if s == BarrierScope.CLUSTER else None,
        )
        for s in scopes
    ]


def _hoist_lock_allocations(kernel: KernelOp) -> None:
    """Move the kernel's top-level published lock allocations to its entry.

    A lock allocation has no operands, so it may run before any op. Grouping
    the allocations lets one publication barrier cover every initialization,
    however an earlier scheduler interleaved them with their first uses.
    """
    locks = [
        op for op in kernel.body.ops
        if _lock_init_barrier(op) != BarrierScope.THREAD
    ]
    rest = [op for op in kernel.body.ops if op not in set(locks)]
    kernel.body.ops = locks + rest


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        kernel = find_op(module, lambda op: isinstance(op, KernelOp))
        assert isinstance(kernel, KernelOp)
        _hoist_lock_allocations(kernel)
        analysis = SyncAnalysis()
        analysis.analyze(module)
        barrier_syncs = _materialize_cluster_mbars(kernel, analysis.targets)
        # Transform: insert the barriers with the kernel-wide participant count.
        _insert_barriers(
            module.body, analysis.targets, kernel.num_threads, barrier_syncs,
        )
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
