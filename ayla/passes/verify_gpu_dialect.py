"""Verify GPU dialect invariants at successive compilation stages."""

from __future__ import annotations

from ..cluster_layout import cluster_size, validate_masks
from ..base.ir import find_ops_of_type
from ..gpu_dialect import PTXTemplateOp

from enum import Enum, auto
from functools import cache
import itertools
import math

from ..base.ir import Block, Op, Type, Value, find_ancestor_of_type, iter_ancestors, iter_ops
from ..base.verification_pass import VerificationPass, verifies
from ..target import CUDA_LAUNCH_LIMITS
from ..gpu_dialect import (
    WarpIdsOp,
    HW_ID_LEVELS,
    AddrMemTensorType,
    AllocOp,
    AsyncOp,
    AwaitOp,
    BF16Type,
    BarrierOp,
    ClusterBlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    CollectiveScope,
    ElementwiseOp,
    F16Type,
    F32Type,
    F64Type,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexType,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LockType,
    OperandPipelineOp,
    PersistentPipelineOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    ReduceOp,
    ReshapeOp,
    RmemTensorType,
    ConstantOp,
    DeallocOp,
    SelectOp,
    SmemTensorType,
    TmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    execution_issuer_warp,
    releases_cluster_banks,
    is_warp_list,
    execution_warps,
    ThreadIdOp,
    WgrmemTensorType,
    BlockIdOp,
    join_collectives,
)


# The target is the device description currently available to compilation.
# Grid-barrier schedules conservatively assume one CTA per SM.
from ..gpu_dialect import root_buffer, is_column_major, physical_layout

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
        or not all(isinstance(level, CollectiveLevel) for level in collective.values())
    ):
        raise ValueError(f"{op.op_id} collective attribute must list x, y, and z")


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
    """An m16n8 fragment reaches column-major smem only through STMatrix.trans."""
    src = op.src.type
    if not (
        isinstance(src, WgrmemTensorType)
        and src.layout in ("wgmma_layout_0", "tcgen05_16x256b")
    ):
        return
    dst_root = root_buffer(op.dst)
    if dst_root is None or not is_column_major(dst_root.type):
        return
    dst = dst_root.type
    if isinstance(dst, GmemTensorType) and isinstance(src.dtype, F32Type):
        # FP32 column-major global partials use two scalar stores per
        # fragment column pair; no STMatrix staging is needed.
        if dst.dtype != src.dtype or list(dst.strides) != [1, dst.sizes[0]]:
            raise ValueError("FP32 transposed fragment stores need dense column-major memory")
        return
    if not (
        isinstance(dst, SmemTensorType)
        and isinstance(src.dtype, (BF16Type, F16Type))
        and dst.swizzle == "128B"
        and list(dst.strides) == [1, dst.sizes[0]]
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
        if 'multicast_cta_masks' in op.attrs or 'multicast_slices' in op.attrs:
            raise ValueError('recipient masks and slices require multicast movement')
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
    if 'multicast_cta_masks' in op.attrs:
        if not op.multicast:
            raise ValueError('recipient masks require multicast movement')
        validate_masks(op.attrs['multicast_cta_masks'],cluster_size,groups=True)
    if 'multicast_slices' in op.attrs:
        slices = op.attrs['multicast_slices']
        masks = op.attrs.get('multicast_cta_masks', [(1 << cluster_size)-1]*cluster_size)
        if type(slices) is not int or not 1 <= slices <= min(mask.bit_count() for mask in masks):
            raise ValueError('multicast slices must fit the recipient group')
    if any(dim is not None for dim in kernel.grid_dim):
        _verify_cluster_shares_multicast_tile(kernel, op)


# Enumerating every CTA and loop iteration is exact but grows with the grid.
# Beyond this many evaluations the multicast-tile check is skipped.
_MAX_MULTICAST_TILE_EVALUATIONS = 1 << 20


def _static_index(
    value: Value | int,
    env: dict[Value, int],
    memo: dict[Value, int | None] | None = None,
) -> int | None:
    """Evaluate static index arithmetic, or ``None`` if it is not static."""
    if isinstance(value, int):
        return value
    if value in env:
        return env[value]
    if memo is None:
        memo = {}
    if value in memo:
        return memo[value]
    result = _evaluate_static_index(value, env, memo)
    memo[value] = result
    return result


def _evaluate_static_index(
    value: Value,
    env: dict[Value, int],
    memo: dict[Value, int | None],
) -> int | None:
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
        lhs, rhs = (_static_index(operand, env, memo) for operand in op.operands)
        if lhs is None or rhs is None:
            return None
        return binary[type(op)](lhs, rhs)
    if isinstance(op, SelectOp):
        index = _static_index(op.index, env, memo)
        if index is None or not 0 <= index < len(op.cases):
            return None
        return _static_index(op.cases[index], env, memo)
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
        if isinstance(op, (BlockIdOp, ClusterBlockIdsOp)) or isinstance(op, Block):
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
    kernel: KernelOp,
    op: MoveOp | TmaMoveOp,
) -> None:
    """Reject a multicast load whose cluster CTAs select different tiles.

    CTAs in each recipient group must select the same tile and participate
    under the same guards. The check evaluates static index arithmetic for
    every CTA and loop iteration, including linearized output grids.
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
        if isinstance(definition, (BlockIdOp, ClusterBlockIdsOp)):
            return (
                cluster[
                    definition.axis
                    if isinstance(definition, BlockIdOp)
                    else value.result_index
                ]
                == 1
            )
        if isinstance(definition, IndexConstantOp):
            return True
        if isinstance(
            definition,
            (
                IndexAddOp,
                IndexMulOp,
                IndexDivOp,
                IndexModOp,
                IndexLtOp,
                SelectOp,
            ),
        ):
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
    block_ids = [leaf for leaf in leaves if isinstance(leaf.definition, BlockIdOp)]
    cluster_ids = [
        leaf for leaf in leaves if isinstance(leaf.definition, ClusterBlockIdsOp)
    ]

    def tile(block: tuple[int, ...], ivs: tuple[int, ...]):
        env: dict[Value, int] = {}
        for value in block_ids:
            definition = value.definition
            assert isinstance(definition, BlockIdOp)
            env[value] = block[definition.axis]
        for value in cluster_ids:
            env[value] = block[value.result_index] % cluster[value.result_index]
        for loop, iv in zip(loops, ivs, strict=True):
            env[loop.body.arguments[0]] = iv
        # Each CTA/iteration has a fresh environment. Shared expression DAGs
        # are evaluated once in that environment; enumeration stays exact.
        memo: dict[Value, int | None] = {}
        if not all(_static_index(guard, env, memo) for guard in guards):
            return ()
        return tuple(_static_index(index, env, memo) for index in indices)

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
            size = math.prod(cluster)
            masks = op.attrs.get('multicast_cta_masks', [(1 << size) - 1] * size)
            ordered = sorted(members, key=lambda b: (b[0] - origin[0]) + cluster[0] * (
                (b[1] - origin[1]) + cluster[1] * (b[2] - origin[2])))
            for mask in set(masks):
                recipients = [ordered[q] for q in range(size) if mask & (1 << q)]
                a = recipients[0]
                for b in recipients[1:]:
                    if tiles[a] == tiles[b]:
                        continue
                    raise ValueError(
                        "multicast load selects different tiles within a "
                        f"cluster: block {a} loads {tiles[a] or 'nothing'} but block "
                        f"{b} loads {tiles[b] or 'nothing'}; recipients must select "
                        "the same multicast tile"
                    )


def _verify_issuer_warp(op: Op) -> None:
    issuer = execution_issuer_warp(op)
    if not issuer:
        return
    kernel = find_ancestor_of_type(op, KernelOp)
    module = find_ancestor_of_type(op, ModuleOp)
    if kernel is None or module is None or module.target != 'sm_100a':
        raise ValueError('nonzero issuer_warp requires a Blackwell kernel')
    if kernel.block_dim[1:] != (1,1) or kernel.num_threads is None or issuer >= kernel.num_threads//32:
        raise ValueError('issuer_warp must name a launched warp in a one-dimensional block')
    if isinstance(op, MoveOp) and not op.attrs.get('tma'):
        raise ValueError('issuer_warp on a move requires TMA')
    if isinstance(op, MatmulOp) and not isinstance(op.dst.type, TmemTensorType):
        raise ValueError('issuer_warp on a matmul requires tcgen05')


def _verify_completion_group(op: AsyncOp | AwaitOp) -> None:
    typ = op.lock.type
    group = typ.cta_group
    if typ.attrs.get('completion_role') not in (None,'result') or typ.attrs.get('completion_role') == 'result' and typ.kind != 'tcgen05':
        raise ValueError('result completion requires a TCGEN event')
    if type(group) is not int or group not in (1, 2):
        raise ValueError("completion cta_group must be one or two")
    for peer in op.lock.users():
        for inner in peer.body.ops if isinstance(peer, AsyncOp) else ():
            # A paired event completes the matching paired MMA or SMEM load.
            mma = isinstance(inner, MatmulOp) and isinstance(inner.dst.type, TmemTensorType)
            load = isinstance(inner, (MoveOp, TmaMoveOp)) and isinstance(inner.dst.type, SmemTensorType)
            if ((inner.dst.type.cta_group if mma else inner.attrs.get('cta_group', 1)) != group
                    or group == 2 and typ.is_event and not (mma if typ.kind == 'tcgen05' else load)):
                raise ValueError("completion group must match the paired operation")
    kernel = find_ancestor_of_type(op, KernelOp)
    if 'completion_cta_masks' in typ.attrs:
        if kernel is None or typ.kind!='tcgen05':
            raise ValueError('CTA completion masks require a clustered TCGEN event')
        masks=validate_masks(typ.attrs['completion_cta_masks'],cluster_size(kernel))
        if group == 2 and len(masks) % 2:
            raise ValueError('paired completion requires complete adjacent rank pairs')
        if group==2 and any(masks[r]!=masks[r^1] for r in range(len(masks))):
            raise ValueError('paired completion masks must agree within each adjacent rank pair')
    if group == 2:
        if kernel is not None and kernel.attrs.get('paired_phase_relay') and kernel.cluster_dim != (2,1,1):
            raise ValueError('paired completion relay requires its original two-X-CTA mapping')
        if not typ.is_event or kernel is None or cluster_size(kernel) % 2 or cluster_size(kernel) < 2:
            raise ValueError("paired completion requires a TMA or tcgen05 lock in complete adjacent rank pairs")


class VerifyGpuDialect(VerificationPass):
    """Check GPU dialect invariants and lowering requirements."""

    def __init__(
        self,
        stage: VerificationStage = VerificationStage.DEFAULT,
    ) -> None:
        super().__init__(stage)

    def run_op(self, op: Op) -> None:
        self._verify_tmem_types(op)
        VerificationPass.run_op(self, op)

    def _verify_tmem_types(self, op: Op) -> None:
        for block in op.blocks:
            for argument in block.arguments:
                if isinstance(argument.type, TmemTensorType):
                    argument.type.validate()
        for value in (*op.operands, *op.results):
            if isinstance(value.type, TmemTensorType):
                value.type.validate()

    @verifies(Op)
    def _verify_participants(self, op: Op) -> None:
        actors = execution_warps(op)
        if actors is None:
            return
        module = find_ancestor_of_type(op, ModuleOp)
        if module is None or module.target != 'sm_100a':
            raise ValueError('operation participants require SM100')
        fragments = [v.type for v in op.operands if isinstance(v.type, WgrmemTensorType)]
        if isinstance(op, IfOp):
            # A bound may select a prefix/suffix, or narrow a verified
            # enclosing participant region to a contiguous interior range.
            condition = op.condition.definition
            kernel = find_ancestor_of_type(op, KernelOp)
            if not isinstance(condition, IndexLtOp):
                raise ValueError('participant region requires a thread-index bound')
            low, high = [v.definition for v in condition.operands]
            nwarps = kernel.num_threads // 32
            selected = None
            if (isinstance(low, ThreadIdOp) and low.axis == CollectiveAxis.X
                    and isinstance(high, IndexConstantOp) and high.value % 32 == 0):
                selected = frozenset(range(max(0, min(nwarps, high.value // 32))))
            elif (isinstance(low, IndexConstantOp) and isinstance(high, ThreadIdOp)
                    and high.axis == CollectiveAxis.X and (low.value + 1) % 32 == 0):
                selected = frozenset(range(max(0, (low.value + 1) // 32), nwarps))
            elif (isinstance(low, WarpIdsOp) and condition.lhs is low.x
                    and kernel.block_dim[1:] == (1, 1) and isinstance(high, IndexConstantOp)):
                selected = frozenset(range(max(0, min(nwarps, high.value))))
            elif (isinstance(low, IndexConstantOp) and isinstance(high, WarpIdsOp)
                    and condition.rhs is high.x and kernel.block_dim[1:] == (1, 1)):
                selected = frozenset(range(max(0, low.value + 1), nwarps))
            for enclosing in iter_ancestors(op):
                if isinstance(enclosing, IfOp) and execution_warps(enclosing) is not None and selected is not None:
                    selected &= execution_warps(enclosing)
            if selected != actors:
                raise ValueError('participant region predicate must match its declared warps')
        elif isinstance(op, (MoveOp, TmaMoveOp, ElementwiseOp)) and fragments:
            if actors not in (frozenset(range(4)), frozenset(range(4, 8))) or any(t.warps != 4 for t in fragments):
                raise ValueError('participant fragments require an aligned group of four warps')
        elif isinstance(op, (AsyncOp, MatmulOp, MoveOp, TmaMoveOp, ElementwiseOp)):
            if actors != frozenset({execution_issuer_warp(op)}):
                raise ValueError('async and engine participants must be the issuing warp')
        elif not isinstance(op, BarrierOp):
            raise ValueError(f'unsupported participant operation {op.op_id}')
        elif (op.scope == 'block' or op.scope == 'block_leader' and op.sync is None) and (
                type(op.attrs.get('id')) is not int or not 1 <= op.attrs['id'] <= 15):
            raise ValueError('subset rendezvous requires a dedicated named barrier')

    @verifies(KernelOp)
    def _verify_kernel(self, op: KernelOp) -> None:
        if type(op.attrs.get("rolled_refill_before_compute", False)) is not bool:
            raise ValueError("rolled_refill_before_compute must be a boolean")
        if op.attrs.get("rolled_refill_before_compute"):
            module = find_ancestor_of_type(op, ModuleOp)
            if module is None or module.target != "sm_100a":
                raise ValueError("rolled_refill_before_compute requires SM100")
        handoffs = op.attrs.get('paired_completion_handoffs', False)
        banks = releases_cluster_banks(op)
        if type(handoffs) is not bool or handoffs and (
                op.cluster_dim != (2, 1, 1) or not op.attrs.get('paired_phase_relay') or banks):
            raise ValueError('paired completion handoffs require the native two-CTA phase relay')
        if banks and not find_ops_of_type(op, PTXTemplateOp):
            from .pipeline_cluster_banks import verify_cluster_bank_pipeline
            verify_cluster_bank_pipeline(op)
        if op.attrs.get("block_gather_backend", "named") not in ("named", "mbar"):
            raise ValueError("block_gather_backend must be named or mbar")
        CUDA_LAUNCH_LIMITS.validate(op.grid_dim, op.block_dim)
        cluster_dim = op.cluster_dim
        if len(cluster_dim) != 3 or any(
            dim is not None and (not isinstance(dim, int) or dim <= 0)
            for dim in cluster_dim
        ):
            raise ValueError("cluster_dim must contain three positive dimensions")
        if all(dim is None for dim in cluster_dim):
            return
        if self.stage == VerificationStage.PTX_LOWERING:
            cluster_dim = tuple(dim if dim is not None else 1 for dim in cluster_dim)
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
        module = find_ancestor_of_type(op, ModuleOp)
        if (module is not None and module.target == "sm_100a"
                and (cluster_dim[0] or 1) % 2
                and any(isinstance(value.type, TmemTensorType) and value.type.cta_group == 2
                        for inner in iter_ops(op)
                        for value in (*inner.operands, *inner.results))):
            # Paired TMEM's physical CTA pairing requires adjacent X ranks.
            # Odd X widths compile but fail the SM100 launch configuration.
            raise ValueError("paired TMEM on sm_100a requires an even physical cluster X dimension")


    @verifies(PersistentPipelineOp)
    def _verify_persistent_pipeline(self, op):
        from .pipeline_rolled_shared_ring import _source

        loop, reduction, _, _, _ = _source(op)
        kernel = find_ancestor_of_type(op, KernelOp)
        frames = op.attrs.get("frames", 2)
        job_regions = op.attrs.get("job_regions", False)
        if (kernel is None or not kernel.attrs.get("rolled_shared_ring")
                or loop.lb != 0 or loop.step != 1
                or type(op.attrs.get("depth")) is not int
                or op.attrs["depth"] < 2
                or op.attrs["depth"] != reduction.attrs["depth"]
                or type(op.attrs.get("mma_queue_depth", 1)) is not int
                or not 1 <= op.attrs.get("mma_queue_depth", 1) < op.attrs["depth"]
                or len(op.operands) != 2
                or not isinstance(op.operands[0].type, TmemTensorType)
                or type(frames) is not int or frames not in (1, 2)
                or frames == 1 and (op.attrs.get("overlap") is not False or op.attrs.get("job_regions", False))
                or kernel.attrs.get("persistent_output_slots") != frames
                or op.operands[0].type.sizes[1] != frames * op.attrs["slot_columns"]):
            raise ValueError("invalid compact persistent pipeline")
        if type(job_regions) is not bool or job_regions and (
                kernel.block_dim != (256, 1, 1) or frames != 2
                or not op.attrs.get("overlap")):
            raise ValueError("job participant regions require a one-dimensional eight-warp overlapping ring")

    @verifies(OperandPipelineOp)
    def _verify_operand_pipeline(self, op):
        from .pipeline_rolled_k import _parts

        _, mma = _parts(op)
        depth = op.attrs.get("depth")
        queue = op.attrs.get("mma_queue_depth", 1)
        lb, ub = op.attrs.get("lb"), op.attrs.get("ub")
        carried = op.attrs.get("carried_events", False)
        if (
            type(depth) is not int
            or depth < 2
            or type(queue) is not int
            or not 1 <= queue < depth
            or type(lb) is not int
            or type(ub) is not int
            # A persistent ring carries banks across output jobs.
            or (ub - lb < (1 if carried else depth - 1)
                and find_ancestor_of_type(op, PersistentPipelineOp) is None)
            or op.attrs.get("step") != 1
            or len(op.body.arguments) != 1
        ):
            raise ValueError("invalid rolled operand pipeline bounds or depth")
        banks = op.operands[:-3] if carried else op.operands
        if carried:
            if len(op.operands) != 5:
                raise ValueError("carried operand pipeline requires job and bank events")
            job, ready, free = op.operands[-3:]
            if (not isinstance(job.type, IndexType)
                    or any(not isinstance(value.type, LockType)
                           or value.type.kind != kind
                           or value.type.attrs.get("count") != depth
                           or value.type.attrs.get("cta_group", 1) != mma.lock.type.attrs.get("cta_group", 1)
                           for value, kind in ((ready, "mbar"), (free, "tcgen05")))):
                raise ValueError("invalid carried operand event storage")
        if len(banks) != 2 or any(
            not isinstance(v.type, SmemTensorType) for v in banks
        ):
            raise ValueError(
                "rolled operand pipeline requires two shared operand views"
            )

    @verifies(ForOp)
    def _verify_for(self, op: ForOp) -> None:
        if type(op.step) is not int or op.step <= 0:
            raise ValueError(f"for step must be a positive integer, got {op.step!r}")

    @verifies(ConstantOp)
    def _verify_constant(self, op: ConstantOp) -> None:
        result_type = op.result().type
        if not isinstance(result_type, MemTensorType):
            raise ValueError("constant result must be a memory tensor")
        if not isinstance(
            result_type.dtype,
            (F16Type, BF16Type, F32Type, F64Type),
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
                f"constant has {len(value)} values for tensor shape {result_type.sizes}"
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
                f"PTX lowering requires a statically shaped constant; got {result_type}"
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
        if not isinstance(arg_type, (AddrMemTensorType, WgrmemTensorType, TmemTensorType)):
            raise ValueError(
                "tile operand and result must be address-backed memory "
                "tensors, wgrmem, or tmem"
            )
        assert isinstance(result_type, (AddrMemTensorType, WgrmemTensorType, TmemTensorType))
        if isinstance(arg_type, WgrmemTensorType):
            from .lower_to_ptx_template import _M16N8_FRAGMENT_ROWS, _wgmma_view_pairs
            if arg_type.layout in _M16N8_FRAGMENT_ROWS:
                _wgmma_view_pairs(op.result())
        native_view = op.attrs.get('native_tmem_view', False)
        if type(native_view) is not bool:
            raise ValueError('native_tmem_view must be a boolean')
        if native_view and (
            not isinstance(arg_type, TmemTensorType)
            or not isinstance(result_type, TmemTensorType)
            or arg_type.sizes[0] != 128 or result_type.sizes[0] not in (64,128)
            or arg_type.cta_group != result_type.cta_group
            or len(op.static_indices) != 2
            or op.static_indices[0] != 0 or op.static_indices[1] is None
        ):
            raise ValueError('native TMEM views require a static column region of a 128-row arena with matching CTA group')
        if isinstance(arg_type, TmemTensorType) and arg_type.sizes[0] != result_type.sizes[0] and not native_view:
            raise ValueError('TMEM views must preserve logical row ownership')
        if isinstance(arg_type, TmemTensorType):
            row, column = op.static_indices
            if row is not None and row != 0:
                raise ValueError('TMEM views require a zero row origin')
            if column is not None and (column < 0
                    or (column + 1) * result_type.sizes[1] > arg_type.sizes[1]):
                raise ValueError('TMEM column view exceeds its logical parent extent')
        if (isinstance(arg_type, TmemTensorType) and arg_type.cta_group == 2
                and arg_type.sizes[0] == 64 and arg_type.sizes != result_type.sizes):
            raise ValueError("paired M64 TMEM requires whole accumulator views")
        if not (arg_type.rank == result_type.rank == len(op.indices)):
            raise ValueError(
                "tile operand rank, result rank, and number of indices must "
                f"match; got {arg_type.rank}, {result_type.rank}, and "
                f"{len(op.indices)}"
            )
        if isinstance(arg_type, AddrMemTensorType):
            assert isinstance(result_type, AddrMemTensorType)
            # A complete K-bank view of block-major SMEM has its own
            # canonical logical stride. Its physical block stride is fixed
            # by the unchanged outer extent and block width, not that logical
            # stride. Ordinary address-backed views still preserve strides.
            blocked_bank = (
                isinstance(arg_type, SmemTensorType)
                and isinstance(result_type, SmemTensorType)
                and (layout := physical_layout(arg_type)) is not None
                and layout[1] == "block_major"
                and physical_layout(result_type) == layout
                and arg_type.dtype == result_type.dtype
                and arg_type.swizzle == result_type.swizzle
                and arg_type.rank == 2
                and all(isinstance(size, int) for size in (*arg_type.sizes, *result_type.sizes))
                and arg_type.strides == (arg_type.sizes[1], 1)
                and result_type.strides == (result_type.sizes[1], 1)
                and arg_type.sizes[0] == result_type.sizes[0]
                and arg_type.sizes[1] % layout[0] == 0
                and result_type.sizes[1] % layout[0] == 0
            )
            if arg_type.strides != result_type.strides and not blocked_bank:
                raise ValueError(
                    f"tile result strides {result_type.strides} must match "
                    f"operand strides {arg_type.strides}"
                )

    @verifies(*HW_ID_LEVELS)
    def _verify_hardware_ids(self, op: Op) -> None:
        if find_ancestor_of_type(op, KernelOp) is None:
            raise ValueError(f"{op.op_id} requires an enclosing kernel")

    @verifies(ClusterBlockIdsOp)
    def _verify_cluster_block_ids(self, op: ClusterBlockIdsOp) -> None:
        kernel = find_ancestor_of_type(op, KernelOp)
        if kernel is None or not any(dim is not None for dim in kernel.cluster_dim):
            raise ValueError("cluster_block_ids requires an enclosing clustered kernel")

    @verifies(BarrierOp)
    def _verify_barrier(self, op: BarrierOp) -> None:
        _verify_issuer_warp(op)
        if 'phase' in op.attrs and (op.attrs['phase'] not in ('arrive', 'wait')
                or op.scope not in ('block_leader', 'cluster_to_leader')
                or op.sync is None):
            raise ValueError('split barrier phases require a gather resource')
        if op.scope in (
            "cluster_to_leader",
            "leader_to_cluster",
            "cluster",
        ):
            kernel = find_ancestor_of_type(op, KernelOp)
            if kernel is None or not any(dim is not None for dim in kernel.cluster_dim):
                raise ValueError(
                    "cluster barrier requires an enclosing clustered kernel"
                )
        expected_operands = 1 if op.scope in ("cluster_to_leader", "grid") or (
            op.scope == "block_leader" and op.sync is not None) else 0
        if len(op.operands) != expected_operands:
            requirement = "one sync operand" if expected_operands else "no operands"
            raise ValueError(f"BarrierOp: {op.scope} scope requires {requirement}")
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
                raise ValueError("BarrierOp: grid sync must contain at least 8 bytes")
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
        elif op.scope == "block_leader" and op.sync is not None:
            typ = op.sync.type
            kernel = find_ancestor_of_type(op, KernelOp)
            assert kernel is not None
            # One arrival per arriving warp: every warp, or the op's participants.
            warps = execution_warps(op)
            arrivals = kernel.num_threads // 32 if warps is None else len(warps)
            if (not isinstance(typ, LockType) or typ.kind != "mbar"
                    or type(typ.attrs.get("arrivals")) is not int or typ.attrs["arrivals"] <= 0
                    or op.attrs.get("phase") != "wait" and typ.attrs["arrivals"] != arrivals):
                raise ValueError("block-leader mbarrier requires one arrival per arriving warp")
            if (op.attrs.get("phase") != "arrive" and warps is not None
                    and execution_issuer_warp(op) not in warps):
                raise ValueError("block-leader wait participants must include its issuing warp")
        elif op.scope == "cluster_to_leader":
            if not isinstance(op.operands[0].type, LockType) or (
                op.operands[0].type.kind != "cluster_mbar"
            ):
                raise ValueError(
                    "cluster_to_leader barrier sync must be a cluster_mbar lock"
                )

    @verifies(AsyncOp)
    def _verify_async(self, op: AsyncOp) -> None:
        _verify_issuer_warp(op)
        _verify_collective_attr(op)
        joined_collective = {axis: CollectiveLevel.THREAD for axis in CollectiveAxis}
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
        _verify_completion_group(op)
        if len(op.completion_locks) > 1:
            module = find_ancestor_of_type(op, ModuleOp)
            kernel = find_ancestor_of_type(op, KernelOp)
            if (module is None or module.target != 'sm_100a' or kernel is None
                    or any(not isinstance(lock.type, LockType)
                        or lock.type.kind != 'tcgen05'
                        or lock.type.cta_group != op.lock.type.cta_group
                        or 'arrivals' in lock.type.attrs for lock in op.completion_locks)
                    or len(set(op.completion_locks)) != len(op.completion_locks)
                    or not op.body.ops or any(not isinstance(inner, MatmulOp) for inner in op.body.ops)):
                raise ValueError('additional completions require matching tcgen05 CTA groups')
            for lock in op.completion_locks[1:]:
                if not any(isinstance(user, AwaitOp) for user in lock.users()):
                    raise ValueError('every completion event requires an await')
                if any(isinstance(user, AsyncOp) and user.lock is lock and user.attrs.get('release_prior')
                       for user in lock.users()):
                    raise ValueError('additional completion peers cannot use release_prior')
        if "arrivals" in lock_type.attrs:
            raise ValueError("a gather mbarrier is not an async completion lock")
        if lock_type.kind == "cluster_mbar":
            raise ValueError("cluster_mbar is a barrier resource, not an async lock")
        release = op.attrs.get("release_prior", False)
        if type(release) is not bool:
            raise ValueError("async release_prior must be a bool")
        peers = [u for u in op.lock.users() if isinstance(u, AsyncOp) and u.lock is op.lock]
        if any(u.attrs.get("release_prior", False) != release for u in peers):
            raise ValueError("completion lock peers must agree on release_prior")
        if release:
            kernel = find_ancestor_of_type(op, KernelOp)
            module = find_ancestor_of_type(op, ModuleOp)
            if (lock_type.kind != "tcgen05" or kernel is None or (kernel.is_clustered and lock_type.cta_group != 2)
                    or module is None or module.attrs.get("target") != "sm_100a"):
                raise ValueError("release_prior requires an unclustered SM100 tcgen05 completion")
        if not any(isinstance(user, AwaitOp) for user in op.lock.users()):
            raise ValueError("async lock must have a matching await")

    @verifies(AwaitOp)
    def _verify_await(self, op: AwaitOp) -> None:
        lock_type = op.lock.type
        assert isinstance(lock_type, LockType)
        if "bulk_read_only" in op.attrs and (
            lock_type.kind != "bulk" or type(op.attrs["bulk_read_only"]) is not bool
        ):
            raise ValueError("bulk_read_only must be a boolean on a bulk await")
        _verify_completion_group(op)
        if "arrivals" in lock_type.attrs:
            raise ValueError("a gather mbarrier must be consumed by a barrier")
        raw_waiters = op.attrs.get("waiter_warps")
        if raw_waiters is not None:
            kernel = find_ancestor_of_type(op, KernelOp)
            module = find_ancestor_of_type(op, ModuleOp)
            paired_engine = (kernel is not None and cluster_size(kernel)>=2 and cluster_size(kernel)%2==0
                and lock_type.is_event and lock_type.cta_group == 2)
            if (kernel is None or kernel.is_clustered and not (paired_engine or releases_cluster_banks(kernel) or kernel.cluster_dim == (2, 1, 1) and kernel.attrs.get('persistent_output_slots') == 2) or kernel.block_dim[1:] != (1, 1)
                    or module is None or module.attrs.get("target") != "sm_100a"
                    or not lock_type.is_event):
                raise ValueError("waiter_warps requires a one-dimensional SM100 block and a supported completion event")
            if not is_warp_list(raw_waiters, kernel):
                raise ValueError("waiter_warps must list distinct valid warp indices")
        pending = op.pending_groups
        limit = 7 if lock_type.supports_partial_await else 0
        if type(pending) is not int or not 0 <= pending <= limit:
            raise ValueError(
                f"await pending count must be an integer in [0, {limit}] for "
                f"a {lock_type.kind} lock; got {pending!r}"
            )

    @verifies(AllocOp)
    def _verify_alloc(self, op: AllocOp) -> None:
        _verify_issuer_warp(op)
        if "init" in op.attrs:
            raise ValueError("alloc does not initialize; use elementwise fill")
        typ = op.result().type
        if isinstance(typ, TmemTensorType):
            kernel = find_ancestor_of_type(op, KernelOp)
            if typ.cta_group == 2 and (typ.sizes[0] not in (64,128) or kernel is None
                    or cluster_size(kernel) < 2 or cluster_size(kernel) % 2):
                raise ValueError("paired TMEM requires two clustered CTAs and 64/128 local rows")
        if isinstance(typ, LockType) and 'release_warps' in typ.attrs:
            kernel = find_ancestor_of_type(op, KernelOp)
            warps = typ.attrs['release_warps']
            if typ.kind != 'cluster_mbar' or kernel is None or not is_warp_list(warps, kernel):
                raise ValueError('slot release must name distinct launched warp arrivals')
            for user in op.result().users():
                expected = frozenset(warps) if user.attrs.get('phase') == 'arrive' else frozenset({0})
                if not isinstance(user, BarrierOp) or execution_warps(user) != expected:
                    raise ValueError('slot release participants must match its arrival count and issuer')
        if isinstance(typ, LockType) and "arrivals" in typ.attrs:
            count = typ.attrs['arrivals']
            if typ.kind != 'mbar' or type(count) is not int or count <= 0:
                raise ValueError('gather arrivals must be a positive integer on an mbar lock')

    @verifies(MoveOp)
    def _verify_move(self, op: MoveOp) -> None:
        _verify_issuer_warp(op)
        _verify_collective_attr(op)
        _verify_tma_multicast(op)
        _verify_transposed_fragment_store(op)
        dst_type, src_type = op.dst.type, op.src.type
        for memory, fragment in ((dst_type, src_type), (src_type, dst_type)):
            if (isinstance(memory, TmemTensorType) and isinstance(fragment, WgrmemTensorType)
                    and memory.sizes[0] == 64
                    and (memory.cta_group == 2) != (fragment.attrs.get("tmem_cta_group", 1) == 2)):
                raise ValueError("paired M64 TMEM moves require matching fragment cta_group metadata")
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
                        and (src_type.strides[0] % 2 == 0
                             or dst_type.layout in ("tcgen05_16x256b", "wgmma_layout_0"))
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
                "with supported contiguous element pairs or broadcast columns"
            )

        from .lower_to_ptx_template import _M16N8_FRAGMENT_ROWS, _m16n8_move_pairs
        for memory, fragment in ((op.dst, op.src), (op.src, op.dst)):
            if (isinstance(memory.type, AddrMemTensorType)
                    and isinstance(fragment.type, WgrmemTensorType)
                    and fragment.type.layout in _M16N8_FRAGMENT_ROWS):
                _m16n8_move_pairs(op)

    @verifies(TmaMoveOp)
    def _verify_tma_move(self, op: TmaMoveOp) -> None:
        _verify_issuer_warp(op)
        if len(op.operands) < 2:
            raise ValueError(
                "tma_move expects dst, src, and one index per dimension; "
                f"got {len(op.operands)} operands"
            )
        dst_is_tma = isinstance(op.dst.type, TmaDescType)
        src_is_tma = isinstance(op.src.type, TmaDescType)
        dst_is_smem = isinstance(op.dst.type, SmemTensorType)
        src_is_smem = isinstance(op.src.type, SmemTensorType)
        if not ((dst_is_tma and src_is_smem) or (dst_is_smem and src_is_tma)):
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
        _verify_issuer_warp(op)

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
        if isinstance(dst, TmemTensorType) and op.attrs.get('cta_group', dst.cta_group) != dst.cta_group:
            raise ValueError("matmul cta_group must match its TMEM accumulator")
        if isinstance(dst, TmemTensorType) and dst.cta_group == 2 and isinstance(rhs, SmemTensorType):
            n *= 2  # Each CTA supplies half of B to the pair's MMA.
        constraints = ((k, rhs_k), (m, dst_m), (n, dst_n))
        if any(a is not None and b is not None and a != b for a, b in constraints):
            raise ValueError(
                f"inconsistent matmul shapes: {lhs.sizes}, {rhs.sizes}, {dst.sizes}"
            )
        init = op.init
        if init is not None and (
            not isinstance(init.type, MemTensorType)
            or (init.type.rank != 0 and init.type.sizes != dst.sizes)
        ):
            raise ValueError(
                "matmul tensor init must match the destination shape; a rank-0 scalar is also supported"
            )

    @verifies(ReduceOp)
    def _verify_reduce(self, op: ReduceOp) -> None:
        if op.fn not in ReduceOp.NEUTRAL_ELEMENTS:
            raise ValueError(f"unknown reduction function {op.fn!r}")
        _verify_collective_attr(op)
        dst, src = op.dst.type, op.src.type
        if not isinstance(dst, MemTensorType) or not isinstance(
            src,
            MemTensorType,
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
                src.sizes[:-1],
                dst.sizes[:-1],
                strict=True,
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
        expected = ElementwiseOp.ARITIES.get(op.fn)
        if expected is None:
            raise ValueError(f"unknown elementwise function {op.fn!r}")
        if op.num_srcs != expected:
            raise ValueError(
                f"elementwise {op.fn} expects {expected} sources, got {op.num_srcs}"
            )
        if type(op.ilp) is not bool:
            raise ValueError(f"elementwise ilp must be a bool; got {op.ilp!r}")
        if op.fn == "fill":
            dst, src = op.dst.type, op.src(0).type
            assert isinstance(dst, MemTensorType)
            if not isinstance(src, MemTensorType) or src.rank != 0:
                raise ValueError("elementwise fill requires a rank-0 source")
            if src.dtype != dst.dtype:
                raise ValueError("elementwise fill requires matching dtypes")

    @verifies(ReshapeOp)
    def _verify_reshape(self, op: ReshapeOp) -> None:
        arg = op.arg.type
        result = op.result().type
        if not isinstance(arg, MemTensorType) or not isinstance(
            result,
            MemTensorType,
        ):
            raise ValueError("reshape operands must be memory tensors")
        if type(arg) is not type(result):
            raise ValueError("reshape must preserve the memory space")
        if arg.dtype != result.dtype:
            raise ValueError("reshape must preserve the element type")
        if isinstance(arg, AddrMemTensorType):
            assert isinstance(result, AddrMemTensorType)
            preserves_strided_layout = tuple(
                (size, stride)
                for size, stride in zip(arg.sizes, arg.strides, strict=True)
                if size != 1
            ) == tuple(
                (size, stride)
                for size, stride in zip(
                    result.sizes,
                    result.strides,
                    strict=True,
                )
                if size != 1
            )
            if (
                not (arg.is_canonical_strides and result.is_canonical_strides)
                and not preserves_strided_layout
            ):
                raise ValueError(
                    "reshape operands must have canonical strides or preserve "
                    "a singleton-only layout change"
                )
        if (
            arg.num_elements is not None
            and result.num_elements is not None
            and arg.num_elements != result.num_elements
        ):
            raise ValueError("reshape must preserve the number of elements")

    @verifies(MatmulOp, stage=VerificationStage.PTX_LOWERING)
    def _verify_matmul_initialized(self, op: MatmulOp) -> None:
        # Matmul lowerings accumulate; PromoteBuffers moves an initializer onto
        # the register accumulator before then.
        if op.init is not None:
            raise ValueError("matmul initializer must be promoted before PTX lowering")

    @verifies(KernelOp, stage=VerificationStage.PTX_LOWERING)
    def _verify_tmem_allocation_sequence(self, kernel):
        from ..analysis.warp_guards import cta_uniform_guards, warp_guard_truth

        operations = list(iter_ops(kernel))
        for inner in operations:
            self._verify_tmem_types(inner)
        groups = {v.type.cta_group for op in operations for v in (*op.operands, *op.results)
                  if isinstance(v.type, TmemTensorType)}
        if len(groups) > 1:
            raise ValueError('all tcgen05 operations in a kernel must use the same CTA group')

        def transfer(ops, prior):
            for op in ops:
                if isinstance(op, AllocOp) and isinstance(op.result().type, TmemTensorType):
                    columns = op.result().type.columns
                    if any(previous is not None and previous < columns for previous in prior):
                        raise ValueError('TMEM allocation columns must not increase in execution order')
                    prior = frozenset({columns})
                elif isinstance(op, IfOp):
                    uniform = cta_uniform_guards((op.condition,), kernel)
                    truth = warp_guard_truth(op.condition, kernel, 0) if uniform else None
                    if truth is not False:
                        taken = transfer(op.body.ops, prior)
                        prior = taken if truth is True else prior | taken
                elif isinstance(op, ForOp):
                    count = (max(0, (op.ub - op.lb + op.step - 1) // op.step)
                             if all(type(v) is int for v in (op.lb, op.ub, op.step)) and op.step > 0 else None)
                    entry = prior
                    for _ in range(min(count, 6) if count is not None else 6):
                        next_prior = transfer(op.body.ops, prior)
                        if count is None:
                            next_prior |= entry
                        if next_prior == prior:
                            break
                        prior = next_prior
            return prior

        # Six possible frontiers: no allocation yet, or a power-of-two size
        # from 32 through 512. Loops stabilize or expose an increase within six.
        transfer(kernel.body.ops, frozenset({None}))

    @verifies(Op, stage=VerificationStage.PTX_LOWERING)
    def _verify_static_shapes(self, op: Op) -> None:
        # Elected issues, tensor memory moves, waiter selection and gather
        # election synchronize complete warps.
        # A guard outside an async group must therefore select whole warps.
        # Optional store atoms inside a group predicate the transfer after
        # election and do not narrow participation in the election itself.
        full_warp = (isinstance(op, AsyncOp) and op.lock.type.kind in ('mbar', 'bulk', 'tcgen05')
                   or isinstance(op, AwaitOp) and op.lock.type.kind in ('mbar', 'tcgen05') and op.waiter_warps is not None
                   or isinstance(op, BarrierOp) and op.scope in ('block_leader', 'cluster_to_leader')
                   or isinstance(op, (TmaMoveOp, WarpIdsOp))
                   or isinstance(op, MoveOp) and (op.attrs.get('tma')
                       or isinstance(op.src.type, TmemTensorType) or isinstance(op.dst.type, TmemTensorType))
                   or isinstance(op, DeallocOp) and isinstance(op.value.type, TmemTensorType)
                   or isinstance(op, ElementwiseOp) and isinstance(op.dst.type, TmemTensorType)
                   or isinstance(op, MatmulOp) and isinstance(op.dst.type, TmemTensorType))
        kernel = find_ancestor_of_type(op, KernelOp)
        tmem_alloc = isinstance(op, AllocOp) and isinstance(op.result().type, TmemTensorType)
        tmem_operation = tmem_alloc or any(isinstance(v.type, TmemTensorType) for v in op.operands)
        if tmem_operation and (kernel is None or kernel.num_threads is None or kernel.num_threads < 32):
            raise ValueError('TMEM synchronization requires a complete allocating warp')
        if tmem_alloc:
            from ..analysis.warp_guards import cta_uniform_guards

            guards = [ancestor.condition for ancestor in reversed(list(iter_ancestors(op)))
                      if isinstance(ancestor, IfOp)]
            if not cta_uniform_guards(guards, kernel):
                raise ValueError('TMEM allocation requires CTA-uniform control for address publication')
            if op.result().type.cta_group == 2:
                from ..analysis.index_proofs import uniform_guard_scope
                from ..analysis.warp_guards import warp_guard_truth

                def grid_uniform(value):
                    producer = value.definition
                    if isinstance(producer, Block):
                        return producer.parent_op is kernel
                    return isinstance(producer, (IndexConstantOp, IndexAddOp, IndexMulOp,
                        IndexDivOp, IndexModOp, IndexLtOp, SelectOp)) and all(
                            grid_uniform(arg) for arg in producer.operands)

                if any(not (grid_uniform(condition)
                        or warp_guard_truth(condition, kernel, 0) is not None
                        or (scope := uniform_guard_scope(condition, kernel)) is not None
                        and CollectiveScope.CLUSTER.leq(scope)) for condition in guards):
                    raise ValueError('paired TMEM allocation requires a peer-uniform guard')
        if isinstance(op, DeallocOp) and isinstance(op.value.type, TmemTensorType):
            value = op.value
            while isinstance(value.definition, TileOp):
                if tuple(value.definition.static_indices) != (0, 0):
                    raise ValueError('TMEM deallocation must use the allocation base')
                value = value.definition.arg
            if not isinstance(value.definition, AllocOp):
                raise ValueError('TMEM deallocation requires a known local allocation base')
            if value.type.columns != op.value.type.columns:
                raise ValueError('TMEM deallocation must match the original allocation column capacity')
        fragment = next((v.type for v in op.operands if isinstance(v.type, WgrmemTensorType)), None)
        if tmem_operation and isinstance(op, (MoveOp, ElementwiseOp)):
            required = fragment.warps if fragment is not None else kernel.num_threads // 32
            actors = execution_warps(op)
            if actors is None:
                actors = next((execution_warps(parent) for parent in iter_ancestors(op)
                               if isinstance(parent, IfOp) and execution_warps(parent) is not None), None)
            valid = (kernel.num_threads == required * 32 and required in (4, 8)
                     if actors is None else fragment is not None and required == 4
                     and actors in (frozenset(range(4)), frozenset(range(4, 8))))
            # The completed persistent region assigns each whole job to an
            # aligned four-warp group during expansion. Its serial body has
            # no participant annotation yet; check the generated groups again
            # before they become PTX templates.
            persistent = find_ancestor_of_type(op, PersistentPipelineOp)
            valid |= (actors is None and fragment is not None and required == 4
                      and persistent is not None and persistent.attrs.get("job_regions") is True
                      and kernel.block_dim == (256, 1, 1))
            if not valid:
                raise ValueError('TMEM fragment warps must match the CTA or a declared aligned participant group')
        if full_warp and find_ancestor_of_type(op, AsyncOp) is None:
            from ..analysis.warp_guards import warp_uniform_guards

            guards = [ancestor.condition for ancestor in reversed(list(iter_ancestors(op)))
                      if isinstance(ancestor, IfOp)]
            if not warp_uniform_guards(guards, kernel):
                raise ValueError('full-warp synchronization requires a warp-uniform guard')
        if isinstance(op, BarrierOp) and (op.scope in ('block', 'grid', 'cluster', 'leader_to_cluster')
                or op.scope == 'block_leader' and op.attrs.get('phase') != 'wait'):
            # Arrival phases must account for every declared participant.
            # A wait-only gather phase adds no new arrivals.
            from ..analysis.warp_guards import cta_uniform_guards, participant_uniform_guards

            guards = [ancestor.condition for ancestor in reversed(list(iter_ancestors(op)))
                      if isinstance(ancestor, IfOp)]
            actors = execution_warps(op)
            valid = (cta_uniform_guards(guards, kernel) if actors is None else
                     participant_uniform_guards(guards, kernel, actors))
            if not valid:
                raise ValueError('counted rendezvous requires uniform control across its participants')
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


def verify_expanded_gpu_hardware(region: Op) -> None:
    """Check hardware contracts while late-generated operations are still IR.

    Dynamic event views can issue and await through different SSA aliases.
    Their ownership protocols are verified at the completed-region boundary;
    ordinary named-lock pairing checks do not apply to those implementations.
    Guard participation, static shapes, TMEM views and allocator contracts do.
    """
    verifier = VerifyGpuDialect(VerificationStage.PTX_LOWERING)
    for op in iter_ops(region):
        if isinstance(op, PersistentPipelineOp) or (
                isinstance(op, OperandPipelineOp) and op is not region):
            raise ValueError("late hardware verification requires expanded semantic regions")
        verifier._verify_tmem_types(op)
        verifier._verify_participants(op)
        _verify_issuer_warp(op)
        verifier._verify_static_shapes(op)
        if isinstance(op, BarrierOp):
            verifier._verify_barrier(op)
        if isinstance(op, TileOp):
            verifier._verify_tile(op)
        if isinstance(op, KernelOp):
            verifier._verify_tmem_allocation_sequence(op)
