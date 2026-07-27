"""Insert cross-thread synchronization barriers on shared-buffer hazards."""
from __future__ import annotations

import enum

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    ConvertOp,
    ForOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    MoveOp,
    RmemTensorType,
    SmemTensorType,
    WrmemTensorType,
    is_side_effect_free,
    root_buffer,
)
from ..base.ir import Block, Op, Type, find_op
from ..base.pipeline import Pass


class CollectiveScope(enum.IntEnum):
    """The granularity at which an effect is completely independent of concurrent
    operations."""
    THREAD = 0
    WARP = 1
    WARPGROUP = 2
    BLOCK = 3
    GRID = 4


# The dataflow state: outstanding concurrent memory effects per root buffer,
# each a (kind, root_buffer, scope) triple.
ConcurrentMemoryEffectsType = dict[
    object, set[tuple[str, object, "CollectiveScope"]]
]

# The analysis result: each effect-bearing op mapped to the coarsest scope its
# accesses must be synced at.
BarrierTargetsType = dict["Op", "CollectiveScope"]


def _type_collective_scope(t: Type) -> CollectiveScope:
    if isinstance(t, RmemTensorType):
        return CollectiveScope.THREAD
    if isinstance(t, WrmemTensorType):
        return CollectiveScope.WARPGROUP
    if isinstance(t, SmemTensorType):
        return CollectiveScope.BLOCK
    if isinstance(t, GmemTensorType):
        return CollectiveScope.GRID
    raise NotImplementedError(
        f"_type_collective_scope: unmodeled buffer type {type(t).__name__}"
    )


def _memory_effects_with_collective_scope(
    op: Op,
) -> set[tuple[str, object, CollectiveScope]] | None:
    """Like ``memory_effects(op)``, but tags each effect with the collective
    scope it is independent at; ``None`` if unmodeled."""
    if isinstance(op, MatmulOp):
        dst = root_buffer(op.dst)
        return {
            ("R", root_buffer(op.lhs), _type_collective_scope(op.lhs.type)),
            ("R", root_buffer(op.rhs), _type_collective_scope(op.rhs.type)),
            ("R", dst, CollectiveScope.THREAD),
            ("W", dst, CollectiveScope.THREAD),
        }
    if isinstance(op, ConvertOp):
        return {
            ("W", root_buffer(op.dst), CollectiveScope.THREAD),
            ("R", root_buffer(op.src), CollectiveScope.THREAD),
        }
    if isinstance(op, MoveOp):
        src_scope = (
            CollectiveScope.THREAD
            if isinstance(op.src.type, WrmemTensorType)
            else _type_collective_scope(op.src.type)
        )
        return {
            ("W", root_buffer(op.dst), _type_collective_scope(op.dst.type)),
            ("R", root_buffer(op.src), src_scope),
        }
    if isinstance(op, AllocOp):
        buf = op.results[0]
        scope = (
            CollectiveScope.THREAD
            if isinstance(buf.type, WrmemTensorType)
            else _type_collective_scope(buf.type)
        )
        return {("W", root_buffer(buf), scope)}
    if isinstance(op, AsyncOp):
        out: set[tuple[str, object, CollectiveScope]] = set()
        for inner in op.body.entry_block.ops:
            inner_effects = _memory_effects_with_collective_scope(inner)
            if inner_effects is None:
                return None
            out |= inner_effects
        return out
    return None


def _join(
    a: ConcurrentMemoryEffectsType, b: ConcurrentMemoryEffectsType,
) -> ConcurrentMemoryEffectsType:
    """The join of this (standard forward) dataflow analysis: the per-buffer
    union of the concurrent-effect-map state."""
    return {
        buf: a.get(buf, set()) | b.get(buf, set())
        for buf in a.keys() | b.keys()
    }


def _transfer(
    ops: list[Op], live_in: ConcurrentMemoryEffectsType, targets: BarrierTargetsType,
) -> ConcurrentMemoryEffectsType:
    """Thread the concurrent-effect map through a straight-line ``ops`` sequence,
    recording into ``targets`` the barrier scope each op needs; return the
    live-out. Pure w.r.t. ``live_in``. A ``for`` op recurses via
    :func:`_transfer_loop`."""
    live = _join(live_in, {})
    for op in ops:
        if isinstance(op, ForOp):
            live = _transfer_loop(op, live, targets)
            continue
        if is_side_effect_free(op):
            continue
        if isinstance(op, AwaitOp):
            # TODO: an await completes its async's outstanding effects and should
            # resolve them against live state; we don't model that yet, so skip.
            continue
        effects = _memory_effects_with_collective_scope(op)
        if effects is None:
            raise NotImplementedError(
                f"unmodelled effect-bearing op: {type(op).__name__}"
            )
        barrier_scope: CollectiveScope = CollectiveScope.THREAD
        for kind_y, buf, scope_y in effects:
            for kind_x, _buf_x, scope_x in live.get(buf, ()):
                # A hazard needs a write (RAR is safe).
                if kind_x == "W" or kind_y == "W":
                    barrier_scope = max(barrier_scope, scope_x, scope_y)
        targets[op] = barrier_scope
        for buf in live:
            live[buf] = {(k, b, s) for (k, b, s) in live[buf] if s > barrier_scope}
        for effect in effects:
            live.setdefault(effect[1], set()).add(effect)
    return live


def _transfer_loop(
    loop: ForOp, live_in: ConcurrentMemoryEffectsType, targets: BarrierTargetsType,
) -> ConcurrentMemoryEffectsType:
    """Analyze a ``for`` body to a fixpoint over its header (the loop-carried
    state at the top of the body), then return the loop's live-out."""
    body = loop.body.entry_block.ops
    while True:
        live_out = _transfer(body, live_in, targets)  # state after one iteration
        joined = _join(live_in, live_out)             # fold the back edge back in
        if joined == live_in:
            return live_out
        live_in = joined


def _insert_barriers(block: Block, targets: BarrierTargetsType) -> None:
    """Insert a ``barrier`` before every op that needs a cross-thread sync
    (scope above THREAD)."""
    new_ops: list[Op] = []
    for op in block.ops:
        scope = targets.get(op, CollectiveScope.THREAD)
        if scope != CollectiveScope.THREAD:
            new_ops.append(BarrierOp.build(scope=scope.name.lower()))
        new_ops.append(op)
        for region in op.regions:
            for inner in region.blocks:
                _insert_barriers(inner, targets)
    block.ops = new_ops


def _analyze(module: ModuleOp) -> BarrierTargetsType:
    """Analyze ``module`` and return the barrier scope each op needs before it
    (the analysis; the dataflow lives in :func:`_transfer`)."""
    kernel = find_op(module, lambda o: isinstance(o, KernelOp))
    targets: BarrierTargetsType = {}
    _transfer(list(kernel.body.entry_block.ops), {}, targets)
    return targets


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        kernel = find_op(module, lambda o: isinstance(o, KernelOp))
        targets = _analyze(module)      # analysis: where do barriers go
        _insert_barriers(kernel.body.entry_block, targets)  # transform: insert them
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    return InsertSyncs().run(module)
