"""Infer full-cluster TMA multicast from SSA index dependence."""
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
    return any(_depends_on(operand, targets, seen) for operand in definition.operands)


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
        for dimension, extent in enumerate(op.cluster_dim):
            cluster_size *= extent
            if extent > 1:
                for ids in find_ops(op, lambda candidate: isinstance(candidate, BlockIdsOp)):
                    clustered_ids.add(ids.results[dimension])
        if cluster_size == 1:
            return

        for candidate in find_ops(op, _is_tma_load):
            values = (
                [candidate.src]
                if isinstance(candidate, MoveOp)
                else list(candidate.indices)
            )
            invariant = not any(
                _depends_on(value, clustered_ids, set()) for value in values
            )
            if invariant:
                candidate.attrs["multicast_count"] = cluster_size
            else:
                candidate.attrs.pop("multicast_count", None)


def infer_tma_multicast(module: ModuleOp) -> ModuleOp:
    return InferTmaMulticast().run(module)
