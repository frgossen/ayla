"""Infer full-cluster TMA multicast from SSA index dependence.

A clustered kernel may multicast a gmem→smem TMA load when its source tile is
invariant along every clustered block dimension.  For example, with
``cluster_dim=[2, 1, 1]`` a B tile indexed only by ``block.y`` is shared by both
``block.x`` CTAs.  The pass records the cluster participant count on the move;
later async/synchronization/lowering passes derive locks and barriers from that
semantic fact.
"""
from __future__ import annotations

from ..base.ir import Block, Op, Value, find_ops
from ..base.pipeline import Pass
from ..gpu_dialect import (
    BlockIdsOp,
    KernelOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TmaMoveOp,
)


def _depends_on(value: Value, targets: set[Value], seen: set[Value]) -> bool:
    if value in targets:
        return True
    if value in seen:
        return False
    seen.add(value)
    definition = value.definition
    if isinstance(definition, Block):
        return False
    return any(_depends_on(v, targets, seen) for v in definition.operands)


def _is_tma_load(op: Op) -> bool:
    return (
        isinstance(op, MoveOp)
        and bool(op.attrs.get("tma"))
        and isinstance(op.dst.type, SmemTensorType)
    ) or (
        isinstance(op, TmaMoveOp)
        and isinstance(op.dst.type, SmemTensorType)
    )


class InferTmaMulticast(Pass):
    """Annotate cluster-invariant TMA loads."""

    def run_op(self, op: Op) -> None:
        if not isinstance(op, KernelOp):
            super().run_op(op)
            return

        cluster_size = 1
        clustered_ids: set[Value] = set()
        for dim, extent in enumerate(op.cluster_dim):
            cluster_size *= extent
            if extent > 1:
                for ids in find_ops(op, lambda x: isinstance(x, BlockIdsOp)):
                    clustered_ids.add(ids.results[dim])

        if cluster_size == 1:
            return

        # Mark the transfer itself.  MoveOp uses its tiled gmem source; a
        # pre-lowered TmaMoveOp uses the materialized descriptor coordinates.
        for candidate in find_ops(op, _is_tma_load):
            values = (
                [candidate.src]
                if isinstance(candidate, MoveOp)
                else list(candidate.indices)
            )
            invariant = not any(
                _depends_on(v, clustered_ids, set()) for v in values
            )
            if invariant:
                candidate.attrs["multicast_count"] = cluster_size
            else:
                candidate.attrs.pop("multicast_count", None)

def infer_tma_multicast(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return InferTmaMulticast().run(module)
