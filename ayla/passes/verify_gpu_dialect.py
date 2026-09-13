"""Verify GPU dialect invariants at successive compilation stages."""
from __future__ import annotations

from enum import Enum, auto

from ..base.ir import Op, Type
from ..base.verification_pass import VerificationPass, verifies
from ..gpu_dialect import (
    AddrMemTensorType,
    BarrierOp,
    ClusterBlockIdsOp,
    CollectiveAxis,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    ReshapeOp,
    TmaDescType,
)


_ELEMENTWISE_ARITIES = {
    "convert": 1,
    "silu": 1,
    "mul": 2,
    "sigmoid": 1,
}


class VerificationStage(Enum):
    """The invariants that must hold when verification runs."""

    DEFAULT = auto()
    PTX_LOWERING = auto()


def _verify_static_shape(typ: Type) -> None:
    if isinstance(typ, MemTensorType) and typ.is_dynamic:
        raise ValueError(f"unresolved dynamic shape {typ}")
    if isinstance(typ, TmaDescType):
        _verify_static_shape(typ.gmem_type)


def _ancestor_kernel(op: Op) -> KernelOp | None:
    block = op.parent_block
    while block is not None:
        parent = block.parent_op
        if isinstance(parent, KernelOp):
            return parent
        block = None if parent is None else parent.parent_block
    return None


class VerifyGpuDialect(VerificationPass):
    """Check GPU dialect invariants and lowering requirements."""

    def __init__(
        self, stage: VerificationStage = VerificationStage.DEFAULT,
    ) -> None:
        super().__init__(stage)

    @verifies(KernelOp)
    def _verify_kernel(self, op: KernelOp) -> None:
        cluster_dim = op.cluster_dim
        if len(cluster_dim) != 3 or any(
            dim is not None and (not isinstance(dim, int) or dim <= 0)
            for dim in cluster_dim
        ):
            raise ValueError(
                "cluster_dim must contain three positive dimensions"
            )
        if all(dim is None for dim in cluster_dim):
            return
        if self.stage == VerificationStage.PTX_LOWERING:
            cluster_dim = tuple(
                dim if dim is not None else 1 for dim in cluster_dim
            )
        grid_dim = (
            tuple(dim if dim is not None else 1 for dim in op.grid_dim)
            if self.stage == VerificationStage.PTX_LOWERING
            else op.grid_dim
        )
        if any(
            grid is not None and cluster is not None and grid % cluster
            for grid, cluster in zip(grid_dim, cluster_dim, strict=True)
        ):
            raise ValueError("cluster_dim must divide grid_dim")

    @verifies(ClusterBlockIdsOp)
    def _verify_cluster_block_ids(self, op: ClusterBlockIdsOp) -> None:
        kernel = _ancestor_kernel(op)
        if kernel is None or not any(
            dim is not None for dim in kernel.cluster_dim
        ):
            raise ValueError(
                "cluster_block_ids requires an enclosing clustered kernel"
            )

    @verifies(BarrierOp)
    def _verify_barrier(self, op: BarrierOp) -> None:
        scope = op.scope
        expected_operands = 1 if scope == "grid" else 0
        if len(op.operands) != expected_operands:
            requirement = "one sync operand" if expected_operands else "no operands"
            raise ValueError(f"BarrierOp: {scope} scope requires {requirement}")
        if scope == "grid":
            sync_type = op.operands[0].type
            if not isinstance(sync_type, GmemTensorType):
                raise ValueError("BarrierOp: grid sync must be global memory")
            if (
                sync_type.num_elements is not None
                and sync_type.num_elements * sync_type.dtype.byte_size < 8
            ):
                raise ValueError(
                    "BarrierOp: grid sync must contain at least 8 bytes"
                )
        if scope == "cluster":
            kernel = _ancestor_kernel(op)
            if kernel is None or not any(
                dim is not None for dim in kernel.cluster_dim
            ):
                raise ValueError(
                    "cluster barrier requires an enclosing clustered kernel"
                )

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
        expected = _ELEMENTWISE_ARITIES.get(op.fn)
        if expected is None:
            raise ValueError(f"unknown elementwise function {op.fn!r}")
        if op.num_srcs != expected:
            raise ValueError(
                f"elementwise {op.fn} expects {expected} sources, "
                f"got {op.num_srcs}"
            )

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
