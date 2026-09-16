"""Verify GPU dialect invariants at successive compilation stages."""
from __future__ import annotations

from enum import Enum, auto

from ..base.ir import Op, Type
from ..base.verification_pass import VerificationPass, verifies
from ..gpu_dialect import (
    AddrMemTensorType,
    CollectiveAxis,
    ElementwiseOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    ReshapeOp,
    TmaDescType,
)


_ELEMENTWISE_FUNCTIONS = {"convert", "sigmoid"}


class VerificationStage(Enum):
    """The invariants that must hold when verification runs."""

    DEFAULT = auto()
    PTX_LOWERING = auto()


def _verify_static_shape(typ: Type) -> None:
    if isinstance(typ, MemTensorType) and typ.is_dynamic:
        raise ValueError(f"unresolved dynamic shape {typ}")
    if isinstance(typ, TmaDescType):
        _verify_static_shape(typ.gmem_type)


class VerifyGpuDialect(VerificationPass):
    """Check GPU dialect invariants and lowering requirements."""

    def __init__(
        self, stage: VerificationStage = VerificationStage.DEFAULT,
    ) -> None:
        super().__init__(stage)

    @verifies(MatmulOp)
    def _verify_matmul(self, op: MatmulOp) -> None:
        collective = op.attrs.get("collective")
        if (
            not isinstance(collective, dict)
            or set(collective) != set(CollectiveAxis)
        ):
            raise ValueError(
                "matmul collective attribute must list x, y, and z"
            )

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
        if any(
            a is not None and b is not None and a != b
            for a, b in constraints
        ):
            raise ValueError(
                f"inconsistent matmul shapes: {lhs.sizes}, {rhs.sizes}, "
                f"{dst.sizes}"
            )

    @verifies(ElementwiseOp)
    def _verify_elementwise(self, op: ElementwiseOp) -> None:
        if op.fn not in _ELEMENTWISE_FUNCTIONS:
            raise ValueError(f"unknown elementwise function {op.fn!r}")

    @verifies(ReshapeOp)
    def _verify_reshape(self, op: ReshapeOp) -> None:
        arg = op.arg.type
        result = op.result().type
        if not isinstance(arg, MemTensorType) or not isinstance(
            result, MemTensorType,
        ):
            raise ValueError("reshape operands must be memory tensors")
        if type(arg) is not type(result):
            raise ValueError("reshape must preserve the memory space")
        if arg.dtype != result.dtype:
            raise ValueError("reshape must preserve the element type")
        # TODO: Accept non-canonical strides when the reshape preserves the
        # underlying linear element order.
        if isinstance(arg, AddrMemTensorType) and (
            not arg.is_canonical_strides
            or not result.is_canonical_strides
        ):
            raise ValueError("reshape operands must have canonical strides")
        if (
            arg.num_elements is not None
            and result.num_elements is not None
            and arg.num_elements != result.num_elements
        ):
            raise ValueError("reshape must preserve the number of elements")

    @verifies(Op, stage=VerificationStage.PTX_LOWERING)
    def _verify_static_shapes(self, op: Op) -> None:
        for block in op.blocks:
            for argument in block.arguments:
                _verify_static_shape(argument.type)
        for value in (*op.operands, *op.results):
            _verify_static_shape(value.type)


def verify_gpu_dialect(
    module: ModuleOp,
    stage: VerificationStage = VerificationStage.DEFAULT,
) -> ModuleOp:
    """Verify a GPU dialect module."""
    return VerifyGpuDialect(stage).run(module)
