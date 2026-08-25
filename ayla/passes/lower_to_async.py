"""Lower hardware-async ops into ``async { op } + await`` pairs.

Wraps the ops that run asynchronously on the hardware — a TMA bulk transfer
(``move`` with ``tma=True`` or a lowered ``tma_move``) and a wgmma matmul
(smem×smem→wrmem with 4 warps) — in an ``async`` region carrying the lock,
followed by an ``await``. The lock kind is ``mbar`` for a TMA load, ``bulk``
for a TMA store, ``wgmma`` for a matmul.

Other moves/matmuls stay synchronous. Already-``async`` ops pass through, so
running the pass twice is idempotent. Later passes reorder/batch the awaits to
overlap data movement with compute.
"""
from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TmaMoveOp,
    WrmemTensorType,
)
from ..base.ir import Block, Op, Region, Value
from ..base.lowering_pass import LoweringPass


def _tma_lock_kind(dst: Value) -> str:
    """Lock kind for a TMA transfer: ``mbar`` for a load (→smem), ``bulk``
    for a store (→gmem)."""
    return "mbar" if isinstance(dst.type, SmemTensorType) else "bulk"


class LowerToAsync(LoweringPass):

    def __init__(self) -> None:
        super().__init__()
        self.register(MoveOp, type(self).lower_move)
        self.register(TmaMoveOp, type(self).lower_tma_move)
        self.register(MatmulOp, type(self).lower_matmul)
        self.register(AsyncOp, type(self).lower_async)

    def _wrap(
        self, inner: Op, kind: str, attrs: dict[str, object],
    ) -> tuple[list[Op], list[object]]:
        """Wrap ``inner`` in ``async(lock){ inner } + await(lock)``.

        The label is tagged ``{label}_async`` on the async op and
        ``{label}_await`` on the await (mirroring each other), so label-matching
        passes see it on the top-level ops rather than buried in the region.
        """
        lock = AllocOp.build_with_fresh_id(LockType.build(kind=kind))
        label = attrs.get("label")
        async_op = AsyncOp.build(
            lock.result(), body=Region(blocks=[Block(ops=[inner])]),
            attrs={"label": f"{label}_async"} if label is not None else None,
        )
        await_op = AwaitOp.build(
            lock.result(),
            attrs={"label": f"{label}_await"} if label is not None else None,
        )
        return [lock, async_op, await_op], []

    def lower_async(
        self, op: AsyncOp,
    ) -> tuple[list[Op], list[object]]:
        # No-op: already async. Returning it untouched (rather than falling
        # through to pass-through) keeps us from recursing into the body and
        # lowering its inner op a second time — idempotency for a second run.
        return [op], list(op.results)

    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not op.attrs.get("tma"):
            return None
        dst = self.value_map[op.dst]
        src = self.value_map[op.src]
        move = MoveOp.build(dst=dst, src=src, attrs=op.attrs)
        return self._wrap(move, _tma_lock_kind(dst), op.attrs)

    def lower_tma_move(
        self, op: TmaMoveOp,
    ) -> tuple[list[Op], list[object]]:
        dst = self.value_map[op.dst]
        src = self.value_map[op.src]
        indices = [self.value_map[v] for v in op.indices]
        tma = TmaMoveOp.build(dst=dst, src=src, indices=indices, attrs=op.attrs)
        return self._wrap(tma, _tma_lock_kind(dst), op.attrs)

    def lower_matmul(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]] | None:
        lhs = self.value_map[op.lhs]
        rhs = self.value_map[op.rhs]
        dst = self.value_map[op.dst]
        # Only wgmma matmuls (smem×smem→wrmem, 4 warps) are async; plain FMA
        # matmuls stay synchronous. Same idea as ``lower_move``'s tma gate.
        # TODO: lower *all* matmuls as async (FMA ones too), wrapping every
        # matmul uniformly — the non-async ones get a trivial lock whose await
        # is a no-op (async + immediate await ≡ sync). That removes this gate
        # and the sync/async split entirely.
        if not (isinstance(lhs.type, SmemTensorType)
                and isinstance(rhs.type, SmemTensorType)
                and isinstance(dst.type, WrmemTensorType)
                and dst.type.warps == 4):
            return None
        matmul = MatmulOp.build(lhs=lhs, rhs=rhs, dst=dst, attrs=op.attrs)
        return self._wrap(matmul, "wgmma", op.attrs)


def lower_to_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToAsync().run(module)
