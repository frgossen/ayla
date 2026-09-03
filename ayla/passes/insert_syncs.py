"""Insert synchronization barriers identified by synchronization analysis."""

from __future__ import annotations

from ..analysis.sync import (
    BarrierScope,
    BarrierTargetsType,
    SyncAnalysis,
)
from ..base.ir import Block, Op, find_op, find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import BarrierOp, KernelOp, ModuleOp


# BarrierScope -> the BarrierOp ``scope`` attr.
_BARRIER_OP_SCOPE: dict[BarrierScope, str] = {
    BarrierScope.BLOCK_TO_LEADER: "block_leader",
    BarrierScope.BLOCK: "block",
    BarrierScope.CLUSTER: "cluster",
}


def _insert_barriers(
    block: Block,
    targets: BarrierTargetsType,
    num_threads: int | None,
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
                assert num_threads is not None, (
                    "block-leader barriers require resolved block dimensions"
                )
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
        analysis = SyncAnalysis()
        analysis.analyze(module)
        kernel = find_op(module, lambda op: isinstance(op, KernelOp))
        assert isinstance(kernel, KernelOp)
        if any(
            op.scope == "grid"
            for op in find_ops_of_type(kernel, BarrierOp)
        ):
            kernel.attrs["requires_resident_grid"] = True
        # Transform: insert the barriers with the kernel-wide participant count.
        _insert_barriers(module.body, analysis.targets, kernel.num_threads)
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
