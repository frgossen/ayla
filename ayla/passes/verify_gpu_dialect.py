"""Verify GPU dialect invariants at successive compilation stages."""
from __future__ import annotations

from enum import Enum, auto
import math

from ..base.ir import Op, Type, find_ancestor_of_type
from ..base.verification_pass import VerificationPass, verifies
from ..gpu_dialect import (
    AddrMemTensorType,
    AsyncOp,
    BF16Type,
    BarrierOp,
    ClusterBlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F16Type,
    F32Type,
    F64Type,
    ForOp,
    GmemTensorType,
    KernelOp,
    LockType,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    ReshapeOp,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    WrmemTensorType,
)


_ELEMENTWISE_ARITIES = {
    "convert": 1,
    "mul": 2,
    "sigmoid": 1,
    "silu": 1,
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


def _verify_collective_attr(op: Op) -> None:
    collective = op.attrs.get("collective")
    if (
        not isinstance(collective, dict)
        or set(collective) != set(CollectiveAxis)
        or not all(
            isinstance(level, CollectiveLevel)
            for level in collective.values()
        )
    ):
        raise ValueError(
            f"{op.op_id} collective attribute must list x, y, and z"
        )


def _verify_tma_multicast(op: MoveOp | TmaMoveOp) -> None:
    multicast = op.multicast
    if type(multicast) is not bool:
        raise ValueError("multicast must be a bool")
    if not multicast:
        return
    if isinstance(op, MoveOp) and not op.attrs.get("tma"):
        raise ValueError("multicast requires a TMA move")
    if not isinstance(op.dst.type, SmemTensorType):
        raise ValueError("TMA multicast is only supported for loads into smem")
    if min(op.collective.values()) < CollectiveLevel.CLUSTER:
        raise ValueError("TMA multicast must be cluster-collective")
    kernel = find_ancestor_of_type(op, KernelOp)
    if kernel is None or not any(dim is not None for dim in kernel.cluster_dim):
        raise ValueError("TMA multicast requires an enclosing clustered kernel")
    cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
    if cluster_size < 2:
        raise ValueError("TMA multicast requires at least two CTAs per cluster")
    if cluster_size > 16:
        raise ValueError("TMA multicast supports at most 16 CTAs per cluster")


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

    @verifies(ForOp)
    def _verify_for(self, op: ForOp) -> None:
        if type(op.step) is not int or op.step <= 0:
            raise ValueError(
                f"for step must be a positive integer, got {op.step!r}"
            )

    @verifies(TileOp)
    def _verify_tile(self, op: TileOp) -> None:
        if len(op.results) != 1:
            raise ValueError(
                f"tile must produce exactly one result, got {len(op.results)}"
            )
        arg_type = op.arg.type
        result_type = op.result().type
        if type(arg_type) is not type(result_type):
            raise ValueError("tile must preserve its operand's memory space")
        if not isinstance(arg_type, (AddrMemTensorType, WrmemTensorType)):
            raise ValueError(
                "tile operand and result must be address-backed memory "
                "tensors or wrmem"
            )
        assert isinstance(result_type, (AddrMemTensorType, WrmemTensorType))
        if not (
            arg_type.rank == result_type.rank == len(op.indices)
        ):
            raise ValueError(
                "tile operand rank, result rank, and number of indices must "
                f"match; got {arg_type.rank}, {result_type.rank}, and "
                f"{len(op.indices)}"
            )
        if isinstance(arg_type, AddrMemTensorType):
            assert isinstance(result_type, AddrMemTensorType)
            if arg_type.strides != result_type.strides:
                raise ValueError(
                    f"tile result strides {result_type.strides} must match "
                    f"operand strides {arg_type.strides}"
                )

    @verifies(ClusterBlockIdsOp)
    def _verify_cluster_block_ids(self, op: ClusterBlockIdsOp) -> None:
        kernel = find_ancestor_of_type(op, KernelOp)
        if kernel is None or not any(
            dim is not None for dim in kernel.cluster_dim
        ):
            raise ValueError(
                "cluster_block_ids requires an enclosing clustered kernel"
            )

    @verifies(BarrierOp)
    def _verify_barrier(self, op: BarrierOp) -> None:
        if op.scope == "cluster_to_leader":
            if len(op.operands) > 1:
                raise ValueError(
                    "BarrierOp: cluster_to_leader scope accepts at most one "
                    "sync operand"
                )
        else:
            expected_operands = 1 if op.scope == "grid" else 0
            if len(op.operands) != expected_operands:
                requirement = (
                    "one sync operand" if expected_operands else "no operands"
                )
                raise ValueError(
                    f"BarrierOp: {op.scope} scope requires {requirement}"
                )
        if op.scope == "grid":
            sync_type = op.operands[0].type
            if not isinstance(sync_type, GmemTensorType):
                raise ValueError("BarrierOp: grid sync must be global memory")
            dtype = sync_type.dtype
            assert isinstance(dtype, (F16Type, BF16Type, F32Type, F64Type))
            if (
                sync_type.num_elements is not None
                and sync_type.num_elements * dtype.byte_size < 8
            ):
                raise ValueError(
                    "BarrierOp: grid sync must contain at least 8 bytes"
                )
        elif op.scope == "cluster_to_leader" and op.sync is not None:
            if not isinstance(op.operands[0].type, LockType) or (
                op.operands[0].type.kind != "cluster_mbar"
            ):
                raise ValueError(
                    "cluster_to_leader barrier sync must be a cluster_mbar lock"
                )
        if op.scope in (
            "cluster_to_leader", "leader_to_cluster", "cluster",
        ):
            kernel = find_ancestor_of_type(op, KernelOp)
            if kernel is None or not any(
                dim is not None for dim in kernel.cluster_dim
            ):
                raise ValueError(
                    "cluster barrier requires an enclosing clustered kernel"
                )

    @verifies(AsyncOp)
    def _verify_async(self, op: AsyncOp) -> None:
        _verify_collective_attr(op)
        for inner in op.body.ops:
            collective = inner.attrs.get("collective")
            if collective is not None and collective != op.collective:
                raise ValueError(
                    "async collective must match the collective of every "
                    f"contained operation; got {op.collective} and "
                    f"{collective}"
                )
        multicast_modes = [
            inner.multicast
            for user in op.lock.users()
            if isinstance(user, AsyncOp)
            for inner in user.body.ops
            if (
                isinstance(inner, TmaMoveOp)
                or (
                    isinstance(inner, MoveOp)
                    and inner.attrs.get("tma") is True
                )
            )
        ]
        if multicast_modes and any(
            mode != multicast_modes[0] for mode in multicast_modes[1:]
        ):
            raise ValueError(
                "TMA operations sharing a lock must use the same multicast mode"
            )
        if (
            op.is_tma_multicast
            and min(op.collective.values()) < CollectiveLevel.CLUSTER
        ):
            raise ValueError(
                "TMA multicast async must be cluster-collective"
            )
        if op.lock.type.kind == "cluster_mbar":
            raise ValueError(
                "cluster_mbar is a barrier resource, not an async lock"
            )

    @verifies(MoveOp)
    def _verify_move(self, op: MoveOp) -> None:
        _verify_collective_attr(op)
        _verify_tma_multicast(op)

    @verifies(TmaMoveOp)
    def _verify_tma_move(self, op: TmaMoveOp) -> None:
        if len(op.operands) < 2:
            raise ValueError(
                "tma_move expects dst, src, and one index per dimension; "
                f"got {len(op.operands)} operands"
            )
        dst_is_tma = isinstance(op.dst.type, TmaDescType)
        src_is_tma = isinstance(op.src.type, TmaDescType)
        dst_is_smem = isinstance(op.dst.type, SmemTensorType)
        src_is_smem = isinstance(op.src.type, SmemTensorType)
        if not (
            (dst_is_tma and src_is_smem)
            or (dst_is_smem and src_is_tma)
        ):
            raise ValueError(
                "tma_move expects one TMA descriptor and one smem operand; "
                f"got dst={op.dst.type}, src={op.src.type}"
            )
        descriptor_type = op.dst.type if dst_is_tma else op.src.type
        assert isinstance(descriptor_type, TmaDescType)
        if len(op.indices) != descriptor_type.gmem_type.rank:
            raise ValueError(
                "tma_move expects one index per TMA dimension; "
                f"got {len(op.indices)} indices for rank "
                f"{descriptor_type.gmem_type.rank}"
            )
        _verify_collective_attr(op)
        _verify_tma_multicast(op)

    @verifies(MatmulOp)
    def _verify_matmul(self, op: MatmulOp) -> None:
        _verify_collective_attr(op)
        for operand_name in ("lhs", "rhs", "dst"):
            attr_name = f"{operand_name}_tiling_scheme"
            scheme_id = op.attrs.get(attr_name)
            if scheme_id is not None and (
                type(scheme_id) is not int or scheme_id < 0
            ):
                raise ValueError(
                    f"matmul {attr_name} must be a non-negative integer"
                )

        lhs, rhs, dst = op.lhs.type, op.rhs.type, op.dst.type
        if (
            not isinstance(lhs, MemTensorType)
            or not isinstance(rhs, MemTensorType)
            or not isinstance(dst, MemTensorType)
        ):
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
        if isinstance(arg, AddrMemTensorType):
            assert isinstance(result, AddrMemTensorType)
            if not arg.is_canonical_strides or not result.is_canonical_strides:
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
