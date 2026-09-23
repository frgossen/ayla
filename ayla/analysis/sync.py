"""Analyze cross-thread synchronization hazards."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Literal, cast

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdsOp,
    ClusterBlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    CollectiveScope,
    ElementwiseOp,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LaneIdOp,
    MatmulOp,
    MemTensorType,
    MemoryEffect,
    MoveOp,
    PTXTemplateOp,
    PtrAddOp,
    ReinterpretOp,
    ReshapeOp,
    SelectOp,
    ThreadIdsOp,
    TileOp,
    TmaMoveOp,
    ToPtrOp,
    WarpGroupIdOp,
    memory_effects,
    root_buffers,
)
from ..base.ir import Op, Value, find_ancestor_of_type
from .dataflow import DataflowAnalysis, DataflowState, transfers


class BarrierScope(enum.IntEnum):
    r"""The cross-thread barrier to emit for a hazard.

    Mirrors CollectiveScope, but where that names the granularity at which an
    access *executes*, this names the *sync* between those granularities. The two
    correspond one-to-one except for the asymmetric block leader syncs,
    BLOCK_TO_LEADER and LEADER_TO_BLOCK::

             GRID
              |
           CLUSTER
              |
            BLOCK
            /   \
    WARPGROUP   BLOCK_TO_LEADER / LEADER_TO_BLOCK
           |     |
         WARP    |
            \   /
            THREAD
    """

    THREAD = 0
    WARP = 1
    WARPGROUP = 2
    BLOCK_TO_LEADER = 3
    LEADER_TO_BLOCK = 4
    BLOCK = 5
    CLUSTER = 6
    GRID = 7

    def join(self, other: BarrierScope) -> BarrierScope:
        """The join (least upper bound) on the BarrierScope lattice."""
        if self == other:
            return self
        if self == BarrierScope.THREAD:
            return other
        if other == BarrierScope.THREAD:
            return self
        if self == BarrierScope.GRID or other == BarrierScope.GRID:
            return BarrierScope.GRID
        if self == BarrierScope.CLUSTER or other == BarrierScope.CLUSTER:
            return BarrierScope.CLUSTER
        if self == BarrierScope.BLOCK or other == BarrierScope.BLOCK:
            return BarrierScope.BLOCK
        if {self, other} == {BarrierScope.WARP, BarrierScope.WARPGROUP}:
            return BarrierScope.WARPGROUP
        return BarrierScope.BLOCK

    def leq(self, other: BarrierScope) -> bool:
        """The partial order (<=) of the BarrierScope lattice."""
        return self.join(other) == other


def _barrier_scope(prior: CollectiveScope, current: CollectiveScope) -> BarrierScope:
    """The barrier scope needed to order a r/w hazard from ``prior`` to ``current``."""
    if prior == current == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.THREAD
    if current == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.BLOCK_TO_LEADER
    if prior == CollectiveScope.BLOCK_LEADER:
        return BarrierScope.LEADER_TO_BLOCK
    # Neither side is a leader: the symmetric barrier for the coarser scope.
    joined = prior.join(current)
    if joined == CollectiveScope.THREAD:
        return BarrierScope.THREAD
    if joined == CollectiveScope.WARP:
        return BarrierScope.WARP
    if joined == CollectiveScope.WARPGROUP:
        return BarrierScope.WARPGROUP
    if joined == CollectiveScope.BLOCK:
        return BarrierScope.BLOCK
    if joined == CollectiveScope.CLUSTER:
        return BarrierScope.CLUSTER
    if joined == CollectiveScope.GRID:
        return BarrierScope.GRID
    raise NotImplementedError(f"_barrier_scope: unhandled join {joined!r}")


def _barrier_covers(barrier: BarrierScope, scope: CollectiveScope) -> bool:
    """Whether ``barrier`` orders a prior effect at collective ``scope`` -- i.e.
    the effect is discharged from the live set once the barrier is placed."""
    if barrier == BarrierScope.THREAD:
        return scope == CollectiveScope.THREAD
    if barrier == BarrierScope.WARP:
        return scope in (CollectiveScope.THREAD, CollectiveScope.WARP)
    if barrier == BarrierScope.WARPGROUP:
        return scope in (
            CollectiveScope.THREAD,
            CollectiveScope.WARP,
            CollectiveScope.WARPGROUP,
        )
    if barrier == BarrierScope.BLOCK_TO_LEADER:
        return scope == CollectiveScope.THREAD  # gather: nothing cross-thread
    if barrier == BarrierScope.LEADER_TO_BLOCK:
        # broadcast: t0's own (BLOCK_LEADER) work only, not all-thread BLOCK
        return scope in (CollectiveScope.THREAD, CollectiveScope.BLOCK_LEADER)
    if barrier == BarrierScope.BLOCK:
        return scope not in (CollectiveScope.CLUSTER, CollectiveScope.GRID)
    if barrier == BarrierScope.CLUSTER:
        return scope != CollectiveScope.GRID
    if barrier == BarrierScope.GRID:
        return True
    raise NotImplementedError(f"_barrier_covers: unhandled barrier {barrier!r}")


@dataclasses.dataclass(frozen=True, order=True)
class ParallelIV:
    """An IV of an implicit hardware-parallel loop."""

    level: CollectiveLevel
    axis: CollectiveAxis


@dataclasses.dataclass(frozen=True)
class AffineExpr:
    """A normalized affine expression over semantic hardware indices."""

    terms: tuple[tuple[ParallelIV, int], ...] = ()
    constant: int = 0

    def __add__(self, other: AffineExpr) -> AffineExpr:
        coefficients = dict(self.terms)
        for iv, coefficient in other.terms:
            coefficients[iv] = coefficients.get(iv, 0) + coefficient
        return AffineExpr(
            tuple(
                sorted(
                    (iv, coefficient)
                    for iv, coefficient in coefficients.items()
                    if coefficient
                )
            ),
            self.constant + other.constant,
        )

    def __mul__(self, other: AffineExpr | int) -> AffineExpr | None:
        if isinstance(other, int):
            if other == 0:
                return AffineExpr()
            return AffineExpr(
                tuple((iv, other * coefficient) for iv, coefficient in self.terms),
                other * self.constant,
            )
        if not self.terms:
            return other * self.constant
        if not other.terms:
            return self * other.constant
        return None


@dataclasses.dataclass(frozen=True)
class IVRange:
    """The half-open range of one parallel induction variable."""

    iv: ParallelIV
    lower: int
    upper: int | None


@dataclasses.dataclass(frozen=True)
class AffineTileAccess:
    """A root-relative logical tile indexed by implicit parallel IVs.

    Each affine index identifies the tile's first element in one root-buffer
    dimension; ``sizes`` is the corresponding result-view shape.
    """

    indices: tuple[AffineExpr, ...]
    sizes: tuple[int, ...]
    iv_ranges: tuple[IVRange, ...]


@dataclasses.dataclass(frozen=True)
class UnknownTileAccess:
    """A tile access outside the supported affine fragment."""


@dataclasses.dataclass(frozen=True)
class TileAwareMemoryEffect:
    """A memory effect enriched with its root-relative tile access."""

    effect: MemoryEffect
    tile_access: AffineTileAccess | UnknownTileAccess


@dataclasses.dataclass(frozen=True)
class ConcurrentMemoryEffect:
    """One outstanding memory effect and its normalized access region."""

    kind: Literal["R", "W"]
    scope: CollectiveScope
    tile_access: AffineTileAccess | UnknownTileAccess
    pending_await: Value | None


def _affine_bounds(
    expr: AffineExpr,
    iv_ranges: tuple[IVRange, ...],
) -> tuple[int, int] | None:
    """Return the inclusive bounds of ``expr``, or unknown."""
    ranges = {iv_range.iv: iv_range for iv_range in iv_ranges}
    lower = upper = expr.constant
    for iv, coefficient in expr.terms:
        iv_range = ranges.get(iv)
        if iv_range is None:
            return None
        if iv_range.upper is None or iv_range.upper <= iv_range.lower:
            return None
        first = coefficient * iv_range.lower
        last = coefficient * (iv_range.upper - 1)
        lower += min(first, last)
        upper += max(first, last)
    return lower, upper


def _difference_may_overlap(
    a: AffineExpr,
    a_ranges: tuple[IVRange, ...],
    a_size: int,
    b: AffineExpr,
    b_ranges: tuple[IVRange, ...],
    b_size: int,
) -> bool:
    """Whether two independently instantiated affine intervals may overlap.

    The attainable start differences are over-approximated by their bounds and
    congruence. This deliberately avoids enumerating the occurrence domain.
    """
    constant = a.constant - b.constant
    lower: int | None = constant
    upper: int | None = constant
    divisor = 0

    for expr, iv_ranges, sign in ((a, a_ranges, 1), (b, b_ranges, -1)):
        ranges = {iv_range.iv: iv_range for iv_range in iv_ranges}
        for iv, original_coefficient in expr.terms:
            coefficient = sign * original_coefficient
            iv_range = ranges.get(iv)
            if iv_range is None:
                return True
            if iv_range.upper is not None and iv_range.upper <= iv_range.lower:
                return False

            first = coefficient * iv_range.lower
            if iv_range.upper is None:
                term_lower = first if coefficient > 0 else None
                term_upper = first if coefficient < 0 else None
                divisor = math.gcd(divisor, abs(coefficient))
            else:
                last = coefficient * (iv_range.upper - 1)
                term_lower = min(first, last)
                term_upper = max(first, last)
                if iv_range.upper - iv_range.lower > 1:
                    divisor = math.gcd(divisor, abs(coefficient))

            lower = (
                lower + term_lower
                if lower is not None and term_lower is not None
                else None
            )
            upper = (
                upper + term_upper
                if upper is not None and term_upper is not None
                else None
            )

    # [a, a + a_size) and [b, b + b_size) overlap exactly when their
    # start difference lies in this closed integer interval.
    overlap_lower = -(a_size - 1)
    overlap_upper = b_size - 1
    if lower is not None:
        overlap_lower = max(overlap_lower, lower)
    if upper is not None:
        overlap_upper = min(overlap_upper, upper)
    if overlap_lower > overlap_upper:
        return False
    if divisor == 0:
        return overlap_lower <= constant <= overlap_upper

    first_congruent = overlap_lower + ((constant - overlap_lower) % divisor)
    return first_congruent <= overlap_upper


def _has_injective_gmem_layout(root: Value) -> bool:
    """Prove that distinct in-bounds logical indices cannot alias."""
    root_type = root.type
    if not isinstance(root_type, GmemTensorType) or root_type.is_dynamic:
        return False
    sizes = cast(tuple[int, ...], tuple(root_type.sizes))
    strides = root_type.strides
    if any(stride is None for stride in strides):
        return False

    # Order dimensions from the finest to the coarsest stride. At every step,
    # one increment in the new dimension must exceed the complete address span
    # of all finer dimensions. This accepts dense, permuted, and padded layouts
    # while rejecting overlapping ones.
    span = 0
    for stride, size in sorted(
        zip(cast(tuple[int, ...], strides), sizes, strict=True),
        key=lambda pair: abs(pair[0]),
    ):
        if size <= 1:
            continue
        if abs(stride) <= span:
            return False
        span += abs(stride) * (size - 1)
    return True


def _access_is_in_bounds(
    access: AffineTileAccess,
    root: Value,
) -> bool:
    root_type = root.type
    assert isinstance(root_type, GmemTensorType)
    if len(access.indices) != root_type.rank:
        return False
    for index, size, root_size in zip(
        access.indices,
        access.sizes,
        root_type.sizes,
        strict=True,
    ):
        bounds = _affine_bounds(index, access.iv_ranges)
        if bounds is None or root_size is None:
            return False
        lower, upper = bounds
        if lower < 0 or upper + size > root_size:
            return False
    return True


def _affine_tile_accesses_are_disjoint(
    root: Value,
    a: AffineTileAccess | UnknownTileAccess,
    b: AffineTileAccess | UnknownTileAccess,
) -> bool:
    """Prove that every occurrence of two accesses is physically disjoint."""
    if not isinstance(a, AffineTileAccess) or not isinstance(b, AffineTileAccess):
        return False
    if (
        not _has_injective_gmem_layout(root)
        or not _access_is_in_bounds(a, root)
        or not _access_is_in_bounds(b, root)
        or len(a.indices) != len(b.indices)
    ):
        return False

    return any(
        not _difference_may_overlap(
            a_index,
            a.iv_ranges,
            a_size,
            b_index,
            b.iv_ranges,
            b_size,
        )
        for a_index, a_size, b_index, b_size in zip(
            a.indices,
            a.sizes,
            b.indices,
            b.sizes,
            strict=True,
        )
    )


def _as_affine_expr(
    value: Value | int,
    allowed_ivs: set[ParallelIV],
) -> AffineExpr | None:
    if isinstance(value, int):
        return AffineExpr(constant=value)
    definition = value.definition
    if isinstance(definition, IndexConstantOp):
        return AffineExpr(constant=definition.value)
    if isinstance(definition, IndexAddOp):
        lhs = _as_affine_expr(definition.lhs, allowed_ivs)
        rhs = _as_affine_expr(definition.rhs, allowed_ivs)
        return lhs + rhs if lhs is not None and rhs is not None else None
    if isinstance(definition, IndexMulOp):
        lhs = _as_affine_expr(definition.lhs, allowed_ivs)
        rhs = _as_affine_expr(definition.rhs, allowed_ivs)
        return lhs * rhs if lhs is not None and rhs is not None else None
    level = None
    if isinstance(definition, BlockIdsOp):
        level = CollectiveLevel.BLOCK
    elif isinstance(definition, ThreadIdsOp):
        level = CollectiveLevel.THREAD
    elif isinstance(definition, WarpGroupIdOp):
        level = CollectiveLevel.WG
    if level is not None:
        variable = ParallelIV(
            level,
            CollectiveAxis(value.result_index),
        )
        return AffineExpr(((variable, 1),)) if variable in allowed_ivs else None
    return None


def _iv_range(kernel: KernelOp, iv: ParallelIV) -> IVRange:
    if iv.level == CollectiveLevel.CLUSTER:
        grid = kernel.grid_dim[iv.axis]
        cluster = kernel.cluster_dim[iv.axis]
        upper = grid // cluster if grid is not None and cluster is not None else None
    elif iv.level == CollectiveLevel.BLOCK:
        upper = kernel.grid_dim[iv.axis]
    elif iv.level == CollectiveLevel.WG:
        assert iv.axis == CollectiveAxis.X
        block_x = kernel.block_dim[CollectiveAxis.X]
        assert block_x is None or block_x % 128 == 0
        upper = None if block_x is None else block_x // 128
    elif iv.level == CollectiveLevel.WARP:
        assert iv.axis == CollectiveAxis.X
        block_x = kernel.block_dim[CollectiveAxis.X]
        assert block_x is None or block_x % 32 == 0
        upper = None if block_x is None else block_x // 32
    elif iv.level == CollectiveLevel.THREAD:
        upper = kernel.block_dim[iv.axis]
    else:
        raise ValueError(f"no IV exists at collective level {iv.level.name.lower()}")
    return IVRange(iv, 0, upper)


def _iv_ranges(op: Op) -> tuple[IVRange, ...]:
    """Return ranges of the implicit hardware IVs that execute ``op``."""
    iv_levels_by_collective = {
        CollectiveLevel.GRID: (),
        CollectiveLevel.CLUSTER: (CollectiveLevel.CLUSTER,),
        CollectiveLevel.BLOCK: (CollectiveLevel.BLOCK,),
        CollectiveLevel.WG: (CollectiveLevel.BLOCK, CollectiveLevel.WG),
        CollectiveLevel.WARP: (CollectiveLevel.BLOCK, CollectiveLevel.WARP),
        CollectiveLevel.THREAD: (
            CollectiveLevel.BLOCK,
            CollectiveLevel.THREAD,
        ),
    }
    kernel = find_ancestor_of_type(op, KernelOp)
    if kernel is None:
        return ()
    collective = op.attrs.get("collective")
    if collective is None:
        # TODO: Give ElementwiseOp and initialized AllocOp explicit collective
        # attributes, then remove this thread-level fallback.
        collective = {axis: CollectiveLevel.THREAD for axis in CollectiveAxis}
    assert isinstance(collective, dict)

    ivs = (
        ParallelIV(iv_level, axis)
        for axis in CollectiveAxis
        for iv_level in iv_levels_by_collective[collective[axis]]
    )
    return tuple(_iv_range(kernel, iv) for iv in ivs)


def _affine_tile_access(
    value: Value,
    iv_ranges: tuple[IVRange, ...],
) -> AffineTileAccess | UnknownTileAccess:
    """Resolve a tile chain to a logical box relative to its unique root."""
    roots = root_buffers(value)
    if len(roots) != 1:
        return UnknownTileAccess()
    root = next(iter(roots))
    if not isinstance(value.type, MemTensorType) or value.type.is_dynamic:
        return UnknownTileAccess()

    sizes = cast(tuple[int, ...], tuple(value.type.sizes))
    indices = [AffineExpr() for _ in sizes]
    allowed_ivs = {iv_range.iv for iv_range in iv_ranges}
    while value is not root:
        tile = value.definition
        if not isinstance(tile, TileOp):
            return UnknownTileAccess()
        tile_sizes = tile.result().type.sizes
        for dim, (index, size) in enumerate(
            zip(tile.indices, tile_sizes, strict=True),
        ):
            affine = _as_affine_expr(index, allowed_ivs)
            scaled = affine * size if affine is not None and size is not None else None
            if scaled is None:
                return UnknownTileAccess()
            indices[dim] += scaled
        value = tile.arg

    return AffineTileAccess(
        tuple(indices),
        sizes,
        iv_ranges,
    )


def _tile_aware_memory_effects(
    op: Op,
) -> list[TileAwareMemoryEffect] | None:
    effects = memory_effects(op)
    if effects is None:
        return None
    iv_ranges = _iv_ranges(op)
    return [
        TileAwareMemoryEffect(
            effect,
            _affine_tile_access(effect.value, iv_ranges),
        )
        for effect in effects
    ]


@dataclasses.dataclass
class SyncState(DataflowState):
    """The forward analysis state: outstanding concurrent memory effects keyed by
    root buffer, named-barrier ids in use until a full-block sync, and ids in use
    until an await on a particular lock."""

    concurrent_memory_effects: dict[object, set[ConcurrentMemoryEffect]] = (
        dataclasses.field(default_factory=dict)
    )
    named_barrier_ids: set[int] = dataclasses.field(default_factory=set)
    named_barrier_ids_pending_await: dict[Value, set[int]] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def empty(cls) -> SyncState:
        return cls()

    def clone(self) -> SyncState:
        """A copy with fresh sets, safe for in-place per-op mutation."""
        return self.join(SyncState.empty())

    def join(self, other: SyncState) -> SyncState:
        """The dataflow join: per-buffer union of effects, union of barrier ids."""
        a, b = self.concurrent_memory_effects, other.concurrent_memory_effects
        p, q = (
            self.named_barrier_ids_pending_await,
            other.named_barrier_ids_pending_await,
        )
        return SyncState(
            {buf: a.get(buf, set()) | b.get(buf, set()) for buf in a.keys() | b.keys()},
            self.named_barrier_ids | other.named_barrier_ids,
            {
                lock: p.get(lock, set()) | q.get(lock, set())
                for lock in p.keys() | q.keys()
            },
        )


def _after_barrier(
    live: SyncState,
    barrier: BarrierScope,
) -> SyncState:
    """Discharge the settled effects ordered by ``barrier``."""
    live = live.clone()
    if barrier in (
        BarrierScope.BLOCK,
        BarrierScope.CLUSTER,
        BarrierScope.GRID,
    ):
        live.named_barrier_ids.clear()
        live.named_barrier_ids_pending_await.clear()
    for buf, effects in live.concurrent_memory_effects.items():
        live.concurrent_memory_effects[buf] = {
            effect
            for effect in effects
            if effect.pending_await is not None
            or not _barrier_covers(barrier, effect.scope)
        }
    return live


@dataclasses.dataclass(frozen=True)
class BarrierTarget:
    """The barrier before an op: its ``scope`` and, for a block-leader barrier,
    the named-barrier ``named_id`` to lower to (``None`` otherwise)."""

    scope: BarrierScope
    named_id: int | None = None


# The analysis result: each effect-bearing op mapped to the barrier before it.
BarrierTargetsType = dict["Op", "BarrierTarget"]


# PTX exposes 16 barrier resources per CTA. ID 0 is reserved for the full-block
# ``bar.sync 0`` emitted by the lowering, leaving IDs 1..15 for named barriers.
_MAX_BARRIER_IDS = 16


def _async_collective_scope(lock: Value) -> CollectiveScope:
    """The collective scope an async op on ``lock`` synchronizes when awaited."""
    kind = lock.type.kind
    if kind == "wgmma":
        return CollectiveScope.WARPGROUP
    if kind == "mbar":
        # A TMA-load mbar await is a leader completion: t0 issues the copy, all
        # threads wait the mbarrier for it. So it closes BLOCK_LEADER (the leader
        # write) but not a full-block all-thread effect.
        return CollectiveScope.BLOCK_LEADER
    if kind == "bulk":
        # Bulk async-groups are per issuing thread, not warp-group-wide. TMA
        # stores are issued and committed by t0, so t0's wait completes the
        # leader-owned read of shared memory. This is the completed memory
        # effect's collective scope, not the execution width of wait_group.
        return CollectiveScope.BLOCK_LEADER
    raise NotImplementedError(f"_async_collective_scope: unmodeled lock kind {kind!r}")


def _transfer_memory_effects(
    op: Op,
    live: SyncState,
    targets: BarrierTargetsType,
    async_lock: Value | None = None,
) -> SyncState:
    """Transfer one synchronous or asynchronous memory-effect operation."""
    effects = _tile_aware_memory_effects(op)
    if effects is None:
        raise NotImplementedError(f"unmodelled effect-bearing op: {type(op).__name__}")
    # The barrier this op needs, as the join over its hazards. Each hazard
    # resolves per-pair (:func:`_barrier_scope`).
    barrier_scope: BarrierScope = BarrierScope.THREAD
    for tile_effect in effects:
        effect = tile_effect.effect
        for root in effect.roots:
            for e in live.concurrent_memory_effects.get(root, ()):
                # A hazard needs a write (RAR is safe).
                if e.kind == "W" or effect.kind == "W":
                    if _affine_tile_accesses_are_disjoint(
                        root,
                        e.tile_access,
                        tile_effect.tile_access,
                    ):
                        continue
                    barrier_scope = barrier_scope.join(
                        _barrier_scope(e.scope, effect.scope),
                    )
    # A block-leader barrier is named. Its id remains in use until either a
    # full-block sync, or (for an mbar async) the await where everyone waits for
    # t0. Reuse any id whose prior lifetime has ended.
    named_id = None
    if barrier_scope == BarrierScope.BLOCK_TO_LEADER:
        recorded = targets.get(op)
        pending_await_ids = {
            barrier_id
            for ids in live.named_barrier_ids_pending_await.values()
            for barrier_id in ids
        }
        used_ids = live.named_barrier_ids | pending_await_ids
        if recorded is not None and recorded.named_id is not None:
            named_id = recorded.named_id
            # TODO: Insert or upgrade to a full-block sync when a loop backedge
            # keeps this named barrier live, instead of rejecting the program.
            assert named_id not in used_ids, (
                f"named barrier {named_id} remains in use across a loop "
                f"backedge before {op}; named barriers are only released by a "
                "full-block sync or their mbar await, so add a full-block sync "
                "before the backedge; "
                f"pending awaits={live.named_barrier_ids_pending_await}"
            )
        if named_id is None:
            # A barrier can first become necessary on a later loop-fixpoint
            # iteration. Do not let it steal an id already assigned to another
            # static barrier site that has not been revisited yet.
            reserved_ids = (
                {
                    target.named_id
                    for target_op, target in targets.items()
                    if target_op is not op and target.named_id is not None
                }
                if recorded is not None
                else set()
            )
            available_ids = set(range(1, _MAX_BARRIER_IDS)) - used_ids - reserved_ids
            if not available_ids:
                raise ValueError(
                    "exhausted named barrier IDs: PTX provides 16 IDs and ID 0 "
                    "is reserved for full-block barriers"
                )
            named_id = min(available_ids)
        # BLOCK_TO_LEADER also applies to leader-issued bulk TMA stores. Only
        # an mbar await is an all-thread rendezvous that ends this handoff.
        if async_lock is not None and async_lock.type.kind == "mbar":
            live.named_barrier_ids_pending_await.setdefault(
                async_lock,
                set(),
            ).add(named_id)
        else:
            live.named_barrier_ids.add(named_id)
    targets[op] = BarrierTarget(barrier_scope, named_id)
    # A barrier orders settled accesses it covers; pending async effects survive
    # until their await completes them.
    live = _after_barrier(live, barrier_scope)
    # An async op's effects that its await fully orders wait pending on the lock
    # (the await will clear them). A coarser effect the await only partially
    # orders -- and every synchronous op's effect -- is live immediately, left for
    # a later barrier to order.
    async_scope = (
        _async_collective_scope(async_lock) if async_lock is not None else None
    )
    for tile_effect in effects:
        effect = tile_effect.effect
        pending = (
            async_lock
            if async_scope is not None and effect.scope.leq(async_scope)
            else None
        )
        for root in effect.roots:
            concurrent_effect = ConcurrentMemoryEffect(
                kind=effect.kind,
                scope=effect.scope,
                tile_access=tile_effect.tile_access,
                pending_await=pending,
            )
            live.concurrent_memory_effects.setdefault(root, set()).add(
                concurrent_effect,
            )
    return live


class SyncAnalysis(DataflowAnalysis[SyncState]):
    def __init__(self) -> None:
        super().__init__(SyncState, direction=self.Direction.FORWARD)
        self.targets: BarrierTargetsType = {}

    @transfers(
        BlockIdsOp,
        ClusterBlockIdsOp,
        IndexAddOp,
        IndexConstantOp,
        IndexDivOp,
        IndexLtOp,
        IndexModOp,
        IndexMulOp,
        LaneIdOp,
        PtrAddOp,
        ReinterpretOp,
        ReshapeOp,
        SelectOp,
        ThreadIdsOp,
        TileOp,
        ToPtrOp,
        WarpGroupIdOp,
    )
    def transfer_no_effect(self, op: Op, state: SyncState) -> SyncState:
        return state

    @transfers(PTXTemplateOp)
    def transfer_ptx(self, op: PTXTemplateOp, state: SyncState) -> SyncState:
        # Raw PTX is opaque, so we can't compute its effects. Skip it. This is
        # unsound in general (a template may read/write memory), but it lets
        # hand-written PTX kernels (e.g. fastcu) through on the assumption they
        # self-manage their sync.
        # TODO: gate on a kernel attribute marking the IR hazard-free or
        # already synced instead.
        return state

    @transfers(BarrierOp)
    def transfer_barrier(
        self,
        op: BarrierOp,
        state: SyncState,
    ) -> SyncState:
        # Schedulers may author structural barriers before this pass, such as
        # block barriers between persistent-grid waves or grid barriers between
        # kernel phases. Account for their ordering so we do not insert a
        # redundant barrier after them.
        try:
            scope = BarrierScope[op.scope.upper()]
        except KeyError as error:
            raise ValueError(f"unknown barrier scope {op.scope!r}") from error
        return _after_barrier(state, scope)

    @transfers(AwaitOp)
    def transfer_await(self, op: AwaitOp, state: SyncState) -> SyncState:
        # Clear every effect pending on this lock -- only the ones this await
        # fully orders were made pending on it. Effects it only partially orders
        # were never pending and stay live for a barrier.
        for buffer, effects in state.concurrent_memory_effects.items():
            state.concurrent_memory_effects[buffer] = {
                effect for effect in effects if effect.pending_await != op.lock
            }
        # An mbar await also completes a block-leader handoff: all threads wait
        # for t0's async work, so no arriving warp can lap the named barrier.
        state.named_barrier_ids_pending_await.pop(op.lock, None)
        return state

    @transfers(AllocOp)
    def transfer_alloc(self, op: AllocOp, state: SyncState) -> SyncState:
        if op.init is None:
            return state
        return _transfer_memory_effects(op, state, self.targets)

    @transfers(AsyncOp)
    def transfer_async(self, op: AsyncOp, state: SyncState) -> SyncState:
        return _transfer_memory_effects(
            op,
            state,
            self.targets,
            async_lock=op.lock,
        )

    @transfers(ElementwiseOp, MatmulOp, MoveOp, TmaMoveOp)
    def transfer_memory_effects(self, op: Op, state: SyncState) -> SyncState:
        return _transfer_memory_effects(op, state, self.targets)
