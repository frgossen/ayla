"""Insert cross-thread synchronization barriers on shared-buffer hazards."""
from __future__ import annotations

import enum

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    ConvertOp,
    EXPLICIT_BARRIERS_ATTR,
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
    """The set of threads that share a buffer, ordered by inclusiveness. A
    hazard is cross-thread unless both accesses are ``THREAD``; the participant
    set of a barrier ordering X→Y is ``max(scope_x, scope_y)``. Note ``bar.sync``
    only reaches up to ``BLOCK`` — a ``GRID`` hazard is beyond it (needs a grid
    sync / separate kernel).
    """
    THREAD = 0      # per-thread registers (rmem)
    WARPGROUP = 1   # the warpgroup's wgmma accumulator (wrmem)
    BLOCK = 2       # block-shared memory (smem)
    GRID = 3        # global memory (gmem), visible across the whole grid


# The dataflow state: outstanding concurrent memory effects per root buffer,
# each a (kind, root_buffer, scope) triple.
ConcurrentMemoryEffects = dict[
    object, set[tuple[str, object, "CollectiveScope"]]
]

# The analysis result: each effect-bearing op mapped to the coarsest scope its
# accesses must be synced at. THREAD means no cross-thread hazard, i.e. no
# barrier; only coarser scopes get one emitted.
BarrierTargets = dict["Op", "CollectiveScope"]


def _type_scope(t: Type) -> CollectiveScope:
    """The scope at which a buffer of type ``t`` is shared, from its memory
    space: per-thread registers (rmem), the warpgroup's wgmma accumulator
    (wrmem), block-shared smem, and grid-wide gmem."""
    if isinstance(t, RmemTensorType):
        return CollectiveScope.THREAD
    if isinstance(t, WrmemTensorType):
        return CollectiveScope.WARPGROUP
    if isinstance(t, SmemTensorType):
        return CollectiveScope.BLOCK
    if isinstance(t, GmemTensorType):
        return CollectiveScope.GRID
    raise NotImplementedError(
        f"_type_scope: unmodeled buffer type {type(t).__name__}"
    )


def _memory_effects_with_scope(
    op: Op,
) -> set[tuple[str, object, CollectiveScope]] | None:
    """The op's memory accesses as a set of ``(kind, root_buffer, scope)``
    triples, or ``None`` for an op this analysis doesn't model.

    ``matmul``/``move``/``alloc`` touch buffers at their operand's memory-space
    scope (:func:`_type_scope`); ``convert`` is element-wise and thread-local.
    ``root_buffer`` strips tile views, and an ``async`` wrapper contributes the
    union of its body's effects.
    """
    if isinstance(op, MatmulOp):
        dst = root_buffer(op.dst)
        # The accumulator (dst) accesses are thread-local for this pass: the wgmma
        # does its own synchronization (commit/wait, baked into the lowering), so
        # no bar.sync is needed for it. The smem operand reads are block-scoped —
        # a thread can't read smem correctly until the others have produced it.
        return {
            ("R", root_buffer(op.lhs), _type_scope(op.lhs.type)),
            ("R", root_buffer(op.rhs), _type_scope(op.rhs.type)),
            ("R", dst, CollectiveScope.THREAD),
            ("W", dst, CollectiveScope.THREAD),
        }
    if isinstance(op, ConvertOp):
        # A convert is element-wise: each thread converts its own register
        # fragment, so its accesses are thread-local regardless of the operand's
        # memory space (e.g. downcasting a wrmem accumulator in the epilogue).
        return {
            ("W", root_buffer(op.dst), CollectiveScope.THREAD),
            ("R", root_buffer(op.src), CollectiveScope.THREAD),
        }
    if isinstance(op, MoveOp):  # reads src, writes dst
        # Reading a wrmem accumulator is thread-local (per-thread register
        # fragments), e.g. the epilogue store of the accumulator to gmem.
        src_scope = (
            CollectiveScope.THREAD
            if isinstance(op.src.type, WrmemTensorType)
            else _type_scope(op.src.type)
        )
        return {
            ("W", root_buffer(op.dst), _type_scope(op.dst.type)),
            ("R", root_buffer(op.src), src_scope),
        }
    if isinstance(op, AllocOp):
        # An init'd alloc writes its buffer; a bare alloc is side-effect-free
        # and is skipped before reaching here. Initializing a wrmem accumulator
        # writes per-thread register fragments, so it's thread-local.
        buf = op.results[0]
        scope = (
            CollectiveScope.THREAD
            if isinstance(buf.type, WrmemTensorType)
            else _type_scope(buf.type)
        )
        return {("W", root_buffer(buf), scope)}
    if isinstance(op, AsyncOp):
        out: set[tuple[str, object, CollectiveScope]] = set()
        for inner in op.body.entry_block.ops:
            inner_effects = _memory_effects_with_scope(inner)
            if inner_effects is None:
                return None  # an unmodeled inner op makes the whole async unknown
            out |= inner_effects
        return out
    return None


def _join(
    a: ConcurrentMemoryEffects, b: ConcurrentMemoryEffects,
) -> ConcurrentMemoryEffects:
    """Per-buffer union of two concurrent-effect maps (the dataflow join)."""
    return {
        buf: a.get(buf, set()) | b.get(buf, set())
        for buf in a.keys() | b.keys()
    }


def _transfer(
    ops: list[Op], live_in: ConcurrentMemoryEffects, targets: BarrierTargets,
) -> ConcurrentMemoryEffects:
    """Thread the concurrent-effect map through a straight-line ``ops`` sequence,
    recording into ``targets`` each op before which a barrier is needed, mapped to
    the scope it must reach; return the live-out. Pure w.r.t. ``live_in`` (copies
    it). A ``for`` op is analyzed via its own header fixpoint
    (:func:`_transfer_loop`)."""
    live = _join(live_in, {})  # fresh copy; we mutate ``live`` below
    for op in ops:
        if isinstance(op, ForOp):
            live = _transfer_loop(op, live, targets)
            continue
        if is_side_effect_free(op):
            continue  # reads/writes no memory: nothing to order (alloc, tile view)
        if isinstance(op, AwaitOp):
            # TODO: an await completes its async's outstanding effects and should
            # resolve them against live state; we don't model that yet, so skip.
            continue
        effects = _memory_effects_with_scope(op)
        # Everything effect-bearing left (matmul, move, async) is modelled;
        # anything else was skipped above, so this must not be None.
        if effects is None:
            raise NotImplementedError(
                f"unmodelled effect-bearing op: {type(op).__name__}"
            )
        # Coarsest scope at which this op conflicts with an outstanding access.
        # THREAD is the join-identity: it stays THREAD iff no hazard was found
        # (any hazard has a non-THREAD access, so it lifts barrier_scope above it).
        barrier_scope: CollectiveScope = CollectiveScope.THREAD
        for kind_y, buf, scope_y in effects:
            for kind_x, _buf_x, scope_x in live.get(buf, ()):
                # A hazard needs a write (RAR is safe). A THREAD/THREAD hazard is
                # ordered by program order and can't lift barrier_scope above the
                # THREAD identity, so it needs no barrier.
                if kind_x == "W" or kind_y == "W":
                    barrier_scope = max(barrier_scope, scope_x, scope_y)
        # Record the scope this op must sync at; THREAD means no cross-thread
        # hazard, so no barrier is emitted for it (see _insert_barriers).
        targets[op] = barrier_scope
        # A barrier orders prior accesses at its scope or finer; keep only the
        # effects it can't sequence (strictly coarser), dropping emptied buffers.
        for buf in live:
            live[buf] = {(k, b, s) for (k, b, s) in live[buf] if s > barrier_scope}
        for effect in effects:
            live.setdefault(effect[1], set()).add(effect)
    return live


def _transfer_loop(
    loop: ForOp, live_in: ConcurrentMemoryEffects, targets: BarrierTargets,
) -> ConcurrentMemoryEffects:
    """Analyze a ``for`` body to a fixpoint over its header (the loop-carried
    state at the top of the body), then return the loop's live-out.

    Accumulating (may) iteration over a finite lattice, so it converges; barrier
    targets accumulate monotonically. Assumes a single-block body.
    """
    body = loop.body.entry_block.ops
    # ``live_in`` is the loop-carried state at the top of the body; it grows each
    # round. Safe to rebind: _transfer/_join never mutate their inputs.
    while True:
        live_out = _transfer(body, live_in, targets)  # state after one iteration
        joined = _join(live_in, live_out)             # fold the back edge back in
        if joined == live_in:
            return live_out
        live_in = joined


def _insert_barriers(block: Block, targets: BarrierTargets) -> None:
    """Insert a ``barrier`` before every op that needs a cross-thread sync
    (scope above THREAD), recursing into nested region blocks (loop bodies,
    async regions)."""
    new_ops: list[Op] = []
    for op in block.ops:
        scope = targets.get(op, CollectiveScope.THREAD)
        if scope != CollectiveScope.THREAD:
            new_ops.append(BarrierOp.build(attrs={"scope": scope.name.lower()}))
        new_ops.append(op)
        for region in op.regions:
            for inner in region.blocks:
                _insert_barriers(inner, targets)
    block.ops = new_ops


def _insert_in_block(block: Block) -> None:
    targets: BarrierTargets = {}
    _transfer(list(block.ops), {}, targets)
    _insert_barriers(block, targets)


class InsertSyncs(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        # One kernel for now; its body may contain nested for loops.
        kernel = find_op(module, lambda o: isinstance(o, KernelOp))
        _insert_in_block(kernel.body.entry_block)
        kernel.attrs[EXPLICIT_BARRIERS_ATTR] = True
        return module


def insert_syncs(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return InsertSyncs().run(module)
