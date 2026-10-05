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
    AsyncOp,
    BarrierOp,
    KernelOp,
    LockType,
    ModuleOp,
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
    pending_init: set[BarrierScope] = set()
    for op in block.ops:
        if pending_init and not isinstance(op, AllocOp):
            new_ops.extend(
                BarrierOp.build(_BARRIER_OP_SCOPE[scope])
                for scope in (BarrierScope.BLOCK, BarrierScope.CLUSTER)
                if scope in pending_init
            )
            pending_init.clear()
        target = targets.get(op)
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
        pending_init.update(_lock_init_barriers(op))
        for inner in op.blocks:
            _insert_barriers(inner, targets, num_threads, barrier_syncs)
    new_ops.extend(
        BarrierOp.build(_BARRIER_OP_SCOPE[scope])
        for scope in (BarrierScope.BLOCK, BarrierScope.CLUSTER)
        if scope in pending_init
    )
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


def _lock_init_barriers(op: Op) -> tuple[BarrierScope, ...]:
    """Publication barriers required by an implicitly initialized lock."""
    if not isinstance(op, AllocOp):
        return ()
    lock_type = op.result().type
    if not isinstance(lock_type, LockType):
        return ()
    if lock_type.kind == "cluster_mbar":
        return (BarrierScope.BLOCK, BarrierScope.CLUSTER)
    if lock_type.kind != "mbar":
        return ()
    multicast = any(
        isinstance(user, AsyncOp) and user.is_tma_multicast
        for user in op.result().users()
    )
    return (
        (BarrierScope.BLOCK, BarrierScope.CLUSTER)
        if multicast
        else (BarrierScope.BLOCK,)
    )


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        analysis = SyncAnalysis()
        analysis.analyze(module)
        kernel = find_op(module, lambda op: isinstance(op, KernelOp))
        assert isinstance(kernel, KernelOp)
        barrier_syncs = _materialize_cluster_mbars(kernel, analysis.targets)
        # Transform: insert the barriers with the kernel-wide participant count.
        _insert_barriers(
            module.body, analysis.targets, kernel.num_threads, barrier_syncs,
        )
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
