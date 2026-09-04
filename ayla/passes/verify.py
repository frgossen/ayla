"""Verify IR invariants required at successive compilation stages."""
from __future__ import annotations

from enum import Enum, auto

from ..base.ir import Type, iter_ops
from ..base.pipeline import Pass
from ..gpu_dialect import MatmulOp, MemTensorType, ModuleOp, TmaDescType


class VerificationStage(Enum):
    """The invariants that must hold when verification runs."""

    DEFAULT = auto()
    PTX_LOWERING = auto()


def _verify_static_shape(typ: Type) -> None:
    if isinstance(typ, MemTensorType) and typ.is_dynamic:
        raise ValueError(f"unresolved dynamic shape {typ}")
    if isinstance(typ, TmaDescType):
        _verify_static_shape(typ.gmem_type)


def _verify_matmul(op: MatmulOp) -> None:
    lhs, rhs, dst = op.lhs.type, op.rhs.type, op.dst.type
    if not all(isinstance(typ, MemTensorType) for typ in (lhs, rhs, dst)):
        raise ValueError("matmul operands must be memory tensors")
    if any(typ.rank != 2 for typ in (lhs, rhs, dst)):
        raise ValueError(
            "matmul operands must be rank 2, got "
            f"{lhs.sizes}, {rhs.sizes}, {dst.sizes}"
        )

    m, k = lhs.sizes
    n, rhs_k = rhs.sizes
    dst_m, dst_n = dst.sizes
    constraints = ((k, rhs_k), (m, dst_m), (n, dst_n))
    if any(a is not None and b is not None and a != b for a, b in constraints):
        raise ValueError(
            f"inconsistent matmul shapes: {lhs.sizes}, {rhs.sizes}, "
            f"{dst.sizes}"
        )


class Verify(Pass):
    """Check general IR invariants and stage-specific lowering requirements."""

    def __init__(
        self, stage: VerificationStage = VerificationStage.DEFAULT,
    ) -> None:
        self.stage = stage

    def run(self, module: ModuleOp) -> ModuleOp:
        # TODO: Register verification functions per op and stage as the set of
        # checked invariants grows.
        for op in iter_ops(module):
            if isinstance(op, MatmulOp):
                _verify_matmul(op)
            if self.stage is VerificationStage.PTX_LOWERING:
                for block in op.blocks:
                    for argument in block.arguments:
                        _verify_static_shape(argument.type)
                for value in (*op.operands, *op.results):
                    _verify_static_shape(value.type)
        return module


def verify(
    module: ModuleOp,
    stage: VerificationStage = VerificationStage.DEFAULT,
) -> ModuleOp:
    """Convenience entry point."""
    return Verify(stage).run(module)
