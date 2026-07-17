"""Verify that every cross-thread shared-buffer hazard is covered by a sync.

This is the coverage-analysis foundation of the sync framework (see the design
discussion). It does not insert anything — it *checks* the implicit "all threads
in lock step" assumption, turning it into a verified property and pinpointing any
shared-memory hazard that isn't ordered by a real synchronization.

Model — two independent axes (a hazard X→Y on a shared buffer, ≥1 writer, is
safe only if BOTH hold):

  1. completion  — X's memory op has finished. Synchronous ops finish in program
     order (``completion=None``); async ops (TMA, wgmma) finish only when their
     lock is drained by an ``await``.
  2. cross-thread order — if X and Y may run on different threads there must be a
     happens-before spanning both. Same thread (both ``THREAD`` scope) ⇒ free.
     Otherwise it needs a full CTA barrier between them, OR — for a RAW whose
     producer is async on an mbarrier — the ``await`` of that mbarrier (the
     arrive/await handshake spans threads and completes the producer at once).

Each op contributes:
  - ``_shared_effects(op)``  -> [(kind, buffer, scope, completion)]
  - sync it provides: ``_drains`` (lock an await drains), ``_mbar_handshake``
    (await on an mbarrier — cross-thread RAW witness), ``_cta_barrier`` (a
    full-CTA rendezvous; an await(wgmma) emits one in codegen, as do cooperative
    gmem↔smem moves and the mbarrier alloc).

Coverage of X (earlier) → Y (later) on buffer B:
  - completed: X sync, or an ``await(X.completion)`` lies between X and Y.
  - ordered:   both THREAD-scope, or a CTA barrier lies between X and Y, or X is
               async-on-mbar and its ``await`` (the handshake) lies between.

Loops: a ForOp body is analyzed with its op list doubled, so cross-iteration
(back-edge) hazards — e.g. this iteration's reload vs next iteration's read — are
checked together with their (repeated) witnesses.

Limitation (documented, not silent): hazards that cross a region boundary
(prologue access in the kernel block vs a consumer inside the loop body) are not
yet tracked — that needs the full cross-region dataflow. The steady-state loop
body, which carries the RAW/WAR we care about, is covered.
"""
from __future__ import annotations

from .gpu_dialect import (
    AsyncOp,
    AwaitOp,
    ForOp,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    memory_effects,
    root_buffer,
)
from .ir import Block, Op

# Thread scopes, ordered by inclusiveness. A hazard is cross-thread unless both
# accesses are THREAD-scope (each thread touching its own slice).
THREAD, WARPGROUP, CTA = "thread", "warpgroup", "cta"


class SyncError(Exception):
    """A cross-thread shared-buffer hazard with no covering synchronization."""


def _op_scope(op: Op) -> str:
    """Which threads collectively perform ``op``'s memory accesses."""
    if isinstance(op, MatmulOp):
        # wgmma (smem operands) is warpgroup-collective; FMA is per-thread.
        return WARPGROUP if isinstance(op.lhs.type, SmemTensorType) else THREAD
    if isinstance(op, MoveOp):
        types = {type(op.dst.type), type(op.src.type)}
        # gmem↔smem (cooperative load/store or a TMA transfer) is CTA-wide;
        # smem↔rmem is per-thread.
        return CTA if any(t is SmemTensorType for t in types) and len(types) == 2 else THREAD
    return THREAD


def _is_shared(buffer: Op) -> bool:
    return isinstance(buffer.type, SmemTensorType)


def _shared_effects(op: Op):
    """``[(kind, buffer, scope, completion)]`` for ``op``'s shared-buffer effects.

    For an ``async`` region the inner ops carry the effects and the region's lock
    is their completion. Bare ops complete synchronously (``completion=None``).
    """
    out = []
    if isinstance(op, AsyncOp):
        completion = op.lock
        for inner in op.body.entry_block.ops:
            for kind, buf in memory_effects(inner) or []:
                if _is_shared(buf):
                    out.append((kind, buf, _op_scope(inner), completion))
        return out
    if op.regions:
        # Region ops (e.g. ForOp) are analyzed by recursion into their bodies;
        # their witnessing awaits live inside, so don't surface them as opaque
        # accesses here. (Cross-region hazards are the documented dataflow gap.)
        return out
    for kind, buf in memory_effects(op) or []:
        if _is_shared(buf):
            out.append((kind, buf, _op_scope(op), None))
    return out


def _drains(op: Op):
    """The lock an ``await`` drains, else ``None``."""
    return op.lock if isinstance(op, AwaitOp) else None


def _is_mbar_handshake(op: Op) -> bool:
    """An ``await`` on an mbarrier — a cross-thread RAW witness (arrive/await)."""
    return isinstance(op, AwaitOp) and op.lock.type.kind == "mbar"


def _provides_cta_barrier(op: Op, wgmma_await_barrier: bool) -> bool:
    """Whether ``op`` emits a full-CTA rendezvous.

    ``await(wgmma)`` emits one in codegen (the WAR ``bar.sync``); cooperative
    gmem↔smem moves and the mbarrier alloc bar.sync internally. The
    ``wgmma_await_barrier`` flag models whether the wgmma await carries that
    bar.sync — set it False to check what breaks without it.
    """
    if isinstance(op, AwaitOp) and op.lock.type.kind == "wgmma":
        return wgmma_await_barrier
    if isinstance(op, MoveOp) and _op_scope(op) == CTA:
        return True
    return False


def _check_block_ops(ops: list[Op], wgmma_await_barrier: bool, where: str) -> None:
    """Coverage check over a linear op list (a block, or a doubled loop body)."""
    # Precompute per-index sync facts.
    drains = [_drains(o) for o in ops]
    handshake = [_is_mbar_handshake(o) for o in ops]
    cta_bar = [_provides_cta_barrier(o, wgmma_await_barrier) for o in ops]

    def covered(ix: int, comp, scope_x: str, iy: int, scope_y: str) -> bool:
        # completion: X sync, or an await(X.completion) in (ix, iy].
        if comp is not None:
            if not any(drains[k] is comp for k in range(ix + 1, iy + 1)):
                return False
        # cross-thread order: both per-thread ⇒ fine.
        if scope_x == THREAD and scope_y == THREAD:
            return True
        # a CTA barrier strictly between, or the mbar handshake that drains X.
        for k in range(ix + 1, iy):
            if cta_bar[k]:
                return True
        if comp is not None and comp.type.kind == "mbar":
            if any(handshake[k] and drains[k] is comp for k in range(ix + 1, iy + 1)):
                return True
        return False

    # Per buffer, the live accesses (index, kind, scope, completion).
    live: dict[Op, list[tuple[int, str, str, object]]] = {}
    for iy, op in enumerate(ops):
        for kind_y, buf, scope_y, _comp_y in _shared_effects(op):
            for ix, kind_x, scope_x, comp_x in live.get(buf, []):
                if kind_x == "R" and kind_y == "R":
                    continue  # RAR: no hazard
                if not covered(ix, comp_x, scope_x, iy, scope_y):
                    raise SyncError(
                        f"{where}: uncovered {kind_x}->{kind_y} hazard on buffer "
                        f"'{buf.name or buf}' between {ops[ix].op_id} ({scope_x}) "
                        f"and {op.op_id} ({scope_y})"
                    )
            live.setdefault(buf, []).append((iy, kind_y, scope_y, _comp_y))


def _check_block(block: Block, wgmma_await_barrier: bool, where: str) -> None:
    ops = list(block.ops)
    # Recurse into nested regions first.
    for op in ops:
        for region in op.regions:
            is_loop = isinstance(op, ForOp)
            for i, b in enumerate(region.blocks):
                sub = f"{where}/{op.op_id}"
                if is_loop:
                    # Double the body so back-edge hazards are checked together
                    # with their repeated witnesses.
                    _check_block_ops(
                        list(b.ops) + list(b.ops),
                        wgmma_await_barrier, sub + " (cyclic)",
                    )
                else:
                    _check_block(b, wgmma_await_barrier, sub)
    _check_block_ops(ops, wgmma_await_barrier, where)


def verify_sync(module: ModuleOp, *, wgmma_await_barrier: bool = True) -> None:
    """Raise :class:`SyncError` if any cross-thread shared-buffer hazard in
    ``module`` lacks a covering synchronization. No-op (returns) if all covered.
    """
    for op in module.body.entry_block.ops:
        for region in op.regions:
            for block in region.blocks:
                _check_block(block, wgmma_await_barrier, op.op_id)
