"""Verify GPU dialect invariants at successive compilation stages."""
from __future__ import annotations

from enum import Enum, auto
from functools import cache
import itertools
import math

from ..base.ir import Block, Op, Type, Value, find_ancestor_of_type, iter_ancestors
from ..base.verification_pass import VerificationPass, verifies
from ..gpu_dialect import (
    AddrMemTensorType,
    AsyncOp,
    AwaitOp,
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
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LockType,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    ReduceOp,
    ReshapeOp,
    RmemTensorType,
    ConstantOp,
    SelectOp,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    WgrmemTensorType,
    is_column_major,
    BlockIdsOp,
    join_collectives,
    physical_layout,
    root_buffer,
)


_ELEMENTWISE_ARITIES = {
    "convert": 1,
    "mul": 2,
    "sigmoid": 1,
    "silu": 1,
}

# The target is the device description currently available to compilation.
# Grid-barrier schedules conservatively assume one CTA per SM.
_RESIDENT_GRID_CTAS_BY_TARGET = {
    "sm_90a": 132,  # H100/H200 SXM
    "sm_100a": 148,  # B200
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


def _non_unit_layout(
    typ: AddrMemTensorType,
) -> list[tuple[int | None, int | None]]:
    """The (size, stride) pairs of non-singleton dimensions."""
    return [
        (size, stride)
        for size, stride in zip(typ.sizes, typ.strides, strict=True)
        if size != 1
    ]


def _verify_transposed_fragment_store(op: MoveOp) -> None:
    """A WGMMA fragment reaches column-major smem only through STMatrix.trans."""
    src = op.src.type
    if not (
        isinstance(src, WgrmemTensorType) and src.layout == "wgmma_layout_0"
    ):
        return
    dst_root = root_buffer(op.dst)
    if dst_root is None or not is_column_major(dst_root.type):
        return
    dst = dst_root.type
    if not (
        isinstance(dst, SmemTensorType)
        and isinstance(src.dtype, (BF16Type, F16Type))
        and dst.swizzle == "128B"
        and physical_layout(dst) == (64, "block_major")
    ):
        raise ValueError(
            "a transposed WGMMA fragment store needs a 16-bit fragment and a "
            f"128B-swizzled, 64-wide block-major smem destination; got {dst}"
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
    if set(op.collective.values()) != {CollectiveLevel.CLUSTER}:
        raise ValueError("TMA multicast must be exactly cluster-collective")
    kernel = find_ancestor_of_type(op, KernelOp)
    if kernel is None or not any(dim is not None for dim in kernel.cluster_dim):
        raise ValueError("TMA multicast requires an enclosing clustered kernel")
    cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
    if cluster_size < 2:
        raise ValueError("TMA multicast requires at least two CTAs per cluster")
    if cluster_size > 16:
        raise ValueError("TMA multicast supports at most 16 CTAs per cluster")
    if any(dim is not None for dim in kernel.grid_dim):
        _verify_cluster_shares_multicast_tile(kernel, op)


# Enumerating every CTA and loop iteration is exact but grows with the grid.
# Beyond this many evaluations the multicast-tile check is skipped.
_MAX_MULTICAST_TILE_EVALUATIONS = 1 << 20


def _static_index(value: Value | int, env: dict[Value, int]) -> int | None:
    """Evaluate static index arithmetic, or ``None`` if it is not static."""
    if isinstance(value, int):
        return value
    if value in env:
        return env[value]
    op = value.definition
    if not isinstance(op, Op):
        return None
    if isinstance(op, IndexConstantOp):
        return op.attrs["value"]  # type: ignore[return-value]
    binary = {
        IndexAddOp: lambda a, b: a + b,
        IndexMulOp: lambda a, b: a * b,
        IndexDivOp: lambda a, b: a // b,
        IndexModOp: lambda a, b: a % b,
        IndexLtOp: lambda a, b: int(a < b),
    }
    if type(op) in binary:
        lhs, rhs = (_static_index(operand, env) for operand in op.operands)
        if lhs is None or rhs is None:
            return None
        return binary[type(op)](lhs, rhs)
    if isinstance(op, SelectOp):
        index = _static_index(op.index, env)
        if index is None or not 0 <= index < len(op.cases):
            return None
        return _static_index(op.cases[index], env)
    return None


def _index_leaves(values: list[Value | int]) -> set[Value]:
    """The values with no defining op that ``values`` are computed from."""
    leaves: set[Value] = set()
    stack = [value for value in values if isinstance(value, Value)]
    seen: set[Value] = set()
    while stack:
        value = stack.pop()
        if value in seen:
            continue
        seen.add(value)
        op = value.definition
        if isinstance(op, (BlockIdsOp, ClusterBlockIdsOp)) or isinstance(op, Block):
            leaves.add(value)
        elif isinstance(op, Op):
            stack.extend(op.operands)
    return leaves


def _multicast_tile_indices(op: MoveOp | TmaMoveOp) -> list[Value | int]:
    """The indices that select a multicast load's global tile."""
    if isinstance(op, TmaMoveOp):
        return list(op.indices)
    indices: list[Value | int] = []
    source = op.src
    while isinstance(source.definition, TileOp):
        indices.extend(source.definition.indices)
        source = source.definition.arg
    return indices


def _verify_cluster_shares_multicast_tile(
    kernel: KernelOp, op: MoveOp | TmaMoveOp,
) -> None:
    """Reject a multicast load whose cluster CTAs select different tiles.

    Every CTA of a cluster receives the issuer's tile, so each must ask for
    the same one. The check evaluates the static index arithmetic for every
    CTA and loop iteration, which also covers linearized output grids.
    """
    grid = tuple(dim or 1 for dim in kernel.grid_dim)
    cluster = tuple(dim or 1 for dim in kernel.cluster_dim)
    loops: list[ForOp] = []
    guards: list[Value] = []
    for ancestor in iter_ancestors(op):
        if isinstance(ancestor, ForOp):
            loops.append(ancestor)
        elif isinstance(ancestor, IfOp):
            guards.append(ancestor.condition)
    indices = _multicast_tile_indices(op)
    loop_ivs = {loop.body.arguments[0] for loop in loops}

    @cache
    def cluster_invariant(value: Value | int) -> bool:
        if isinstance(value, int) or value in loop_ivs:
            return True
        definition = value.definition
        if isinstance(definition, (BlockIdsOp, ClusterBlockIdsOp)):
            return cluster[value.result_index] == 1
        if isinstance(definition, IndexConstantOp):
            return True
        if isinstance(definition, (
            IndexAddOp, IndexMulOp, IndexDivOp, IndexModOp, IndexLtOp, SelectOp,
        )):
            return all(cluster_invariant(operand) for operand in definition.operands)
        return False

    # A pure expression whose inputs cannot vary within the cluster needs no
    # grid enumeration. Include control predicates so inactive CTAs cannot be
    # mistaken for participants. Unknown expressions retain the exact check.
    if all(cluster_invariant(value) for value in [*indices, *guards]):
        return
    leaves = _index_leaves([*indices, *guards])
    loops = [loop for loop in loops if loop.body.arguments[0] in leaves]
    ranges = [range(loop.lb, loop.ub, loop.step) for loop in loops]
    evaluations = math.prod(grid) * math.prod(len(r) for r in ranges)
    if evaluations > _MAX_MULTICAST_TILE_EVALUATIONS:
        return
    block_ids = [
        leaf for leaf in leaves if isinstance(leaf.definition, BlockIdsOp)
    ]
    cluster_ids = [
        leaf for leaf in leaves
        if isinstance(leaf.definition, ClusterBlockIdsOp)
    ]

    def tile(block: tuple[int, ...], ivs: tuple[int, ...]):
        env: dict[Value, int] = {}
        for value in block_ids:
            env[value] = block[value.result_index]
        for value in cluster_ids:
            env[value] = block[value.result_index] % cluster[value.result_index]
        for loop, iv in zip(loops, ivs, strict=True):
            env[loop.body.arguments[0]] = iv
        if not all(_static_index(guard, env) for guard in guards):
            return ()
        return tuple(_static_index(index, env) for index in indices)

    origins = itertools.product(
        *(range(0, g, c) for g, c in zip(grid, cluster, strict=True))
    )
    for origin in origins:
        members = [
            tuple(o + d for o, d in zip(origin, offset, strict=True))
            for offset in itertools.product(*(range(c) for c in cluster))
        ]
        for ivs in itertools.product(*ranges):
            tiles = {member: tile(member, ivs) for member in members}
            if any(None in t for t in tiles.values()):
                return  # Not static: nothing to prove.
            if len(set(tiles.values())) > 1:
                (a, ta), (b, tb) = [
                    (member, t) for member, t in tiles.items()
                    if t != tiles[members[0]]
                ][:1] + [(members[0], tiles[members[0]])]
                raise ValueError(
                    "multicast load selects different tiles within a "
                    f"cluster: block {a} loads {ta or 'nothing'} but block "
                    f"{b} loads {tb or 'nothing'}; the cluster must run "
                    "along an axis on which the multicast tile is shared"
                )


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

    @verifies(ConstantOp)
    def _verify_constant(self, op: ConstantOp) -> None:
        result_type = op.result().type
        if not isinstance(result_type, MemTensorType):
            raise ValueError("constant result must be a memory tensor")
        if not isinstance(
            result_type.dtype, (F16Type, BF16Type, F32Type, F64Type),
        ):
            raise ValueError("constant result must have a floating-point dtype")
        value = op.attrs.get("value")
        if type(value) in (int, float):
            return
        if not isinstance(value, list) or any(
            type(element) not in (int, float) for element in value
        ):
            raise ValueError(
                "constant value must be an int, float, or flat numerical list"
            )
        if result_type.num_elements != len(value):
            raise ValueError(
                f"constant has {len(value)} values for tensor shape "
                f"{result_type.sizes}"
            )

    @verifies(ConstantOp, stage=VerificationStage.PTX_LOWERING)
    def _verify_lowerable_constant(self, op: ConstantOp) -> None:
        result_type = op.result().type
        if not isinstance(result_type, RmemTensorType):
            raise ValueError(
                "PTX lowering supports constant materialization only in rmem; "
                f"got {result_type}"
            )
        if result_type.num_elements is None:
            raise ValueError(
                f"PTX lowering requires a statically shaped constant; "
                f"got {result_type}"
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
        if not isinstance(arg_type, (AddrMemTensorType, WgrmemTensorType)):
            raise ValueError(
                "tile operand and result must be address-backed memory "
                "tensors or wgrmem"
            )
        assert isinstance(result_type, (AddrMemTensorType, WgrmemTensorType))
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
        expected_operands = (
            1 if op.scope in ("cluster_to_leader", "grid") else 0
        )
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
            kernel = find_ancestor_of_type(op, KernelOp)
            module = find_ancestor_of_type(op, ModuleOp)
            assert kernel is not None and module is not None
            max_ctas = _RESIDENT_GRID_CTAS_BY_TARGET.get(module.target)
            if max_ctas is None:
                raise ValueError(
                    "software grid barrier has no residency limit for target "
                    f"{module.target!r}"
                )
            grid_size = math.prod(dim or 1 for dim in kernel.grid_dim)
            if grid_size > max_ctas:
                raise ValueError(
                    f"kernel {kernel.name!r} uses a software grid barrier and "
                    f"launches {grid_size} physical CTAs, but target "
                    f"{module.target!r} permits at most {max_ctas}"
                )
        elif op.scope == "cluster_to_leader":
            if not isinstance(op.operands[0].type, LockType) or (
                op.operands[0].type.kind != "cluster_mbar"
            ):
                raise ValueError(
                    "cluster_to_leader barrier sync must be a cluster_mbar lock"
                )
    @verifies(AsyncOp)
    def _verify_async(self, op: AsyncOp) -> None:
        _verify_collective_attr(op)
        joined_collective = {
            axis: CollectiveLevel.THREAD for axis in CollectiveAxis
        }
        for inner in op.body.ops:
            collective = inner.attrs.get("collective")
            if collective is None:
                continue
            assert isinstance(collective, dict)
            joined_collective = join_collectives(joined_collective, collective)
        if joined_collective != op.collective:
            raise ValueError(
                "async collective must match the join of its contained "
                f"operations; got {op.collective}, expected "
                f"{joined_collective}"
            )
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        if lock_type.kind == "cluster_mbar":
            raise ValueError(
                "cluster_mbar is a barrier resource, not an async lock"
            )

    @verifies(AwaitOp)
    def _verify_await(self, op: AwaitOp) -> None:
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        pending = op.attrs.get("pending", 0)
        limit = 7 if lock_type.supports_partial_await else 0
        if type(pending) is not int or not 0 <= pending <= limit:
            raise ValueError(
                f"await pending count must be an integer in [0, {limit}] for "
                f"a {lock_type.kind} lock; got {pending!r}"
            )

    @verifies(MoveOp)
    def _verify_move(self, op: MoveOp) -> None:
        _verify_collective_attr(op)
        _verify_tma_multicast(op)
        _verify_transposed_fragment_store(op)
        dst_type, src_type = op.dst.type, op.src.type
        if (
            isinstance(dst_type, WgrmemTensorType)
            and isinstance(src_type, AddrMemTensorType)
            and (
                dst_type.rank != 2
                or dst_type.sizes != src_type.sizes
                or dst_type.dtype != src_type.dtype
                or not (
                    (
                        src_type.strides[0] is not None
                        and src_type.strides[0] % 2 == 0
                        and src_type.strides[1] == 1
                    )
                    or (
                        src_type.strides[1] == 0
                        and isinstance(src_type.dtype, (BF16Type, F16Type, F32Type))
                    )
                )
            )
        ):
            raise ValueError(
                "address-to-WGRMEM move requires equal rank-2 shapes and dtypes "
                "with aligned contiguous element pairs or broadcast columns"
            )

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
        if op.init is not None and op.init_value is not None:
            raise ValueError("matmul accepts a scalar or tensor init, not both")
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
        init = op.init_value
        if init is not None and (
            not isinstance(init.type, MemTensorType) or init.type.sizes != dst.sizes
        ):
            raise ValueError("matmul tensor init must match the destination shape")

    @verifies(ReduceOp)
    def _verify_reduce(self, op: ReduceOp) -> None:
        _verify_collective_attr(op)
        dst, src = op.dst.type, op.src.type
        if not isinstance(dst, MemTensorType) or not isinstance(
            src, MemTensorType,
        ):
            raise ValueError("reduce operands must be memory tensors")
        if src.rank == 0 or dst.rank != src.rank:
            raise ValueError(
                "reduce source and destination must have the same nonzero "
                f"rank; got {src.sizes} and {dst.sizes}"
            )
        if any(
            src_size != dst_size
            for src_size, dst_size in zip(
                src.sizes[:-1], dst.sizes[:-1], strict=True,
            )
        ):
            raise ValueError(
                "reduce source and destination parallel dimensions must "
                f"match; got {src.sizes} and {dst.sizes}"
            )
        src_reduction_size = src.sizes[-1]
        dst_reduction_size = dst.sizes[-1]
        if (
            src_reduction_size is not None
            and dst_reduction_size is not None
            and dst_reduction_size > src_reduction_size
        ):
            raise ValueError(
                "reduce destination reduction dimension cannot exceed the "
                f"source; got {src.sizes} and {dst.sizes}"
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
        # TODO: Accept other non-canonical strides when the reshape preserves
        # the underlying linear element order.
        if isinstance(arg, AddrMemTensorType):
            assert isinstance(result, AddrMemTensorType)
            if not (
                arg.is_canonical_strides and result.is_canonical_strides
            ) and _non_unit_layout(arg) != _non_unit_layout(result):
                raise ValueError(
                    "reshape operands must have canonical strides or differ "
                    "only by singleton dimensions"
                )
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
