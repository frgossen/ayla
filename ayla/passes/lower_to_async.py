"""Lower hardware-async ops into ``async { op } + await`` pairs.

Wraps the ops that run asynchronously on the hardware — a TMA bulk transfer
(``move`` with ``tma=True`` or a lowered ``tma_move``) and a wgmma matmul
(smem×smem→wgrmem with 4 warps) — in an ``async`` block carrying the lock,
followed by an ``await``. The lock kind is ``mbar`` for a TMA load, ``bulk``
for a TMA store, ``wgmma`` for a matmul. wgmma groups complete in issue order,
so a matmul gets no await here: insert_syncs awaits only the groups that the
work after it needs, leaving newer ones in flight.

Other moves/matmuls stay synchronous. Already-``async`` ops pass through, so
running the pass twice is idempotent. Later passes reorder/batch the awaits to
overlap data movement with compute.
"""
from __future__ import annotations

from typing import cast

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
    WgrmemTensorType,
)
from ..base.ir import Block, Op, Value
from ..base.lowering_pass import LoweringPass, lowers


def _tma_lock_kind(dst: Value) -> str:
    """Lock kind for a TMA transfer: ``mbar`` for a load (→smem), ``bulk``
    for a store (→gmem)."""
    return "mbar" if isinstance(dst.type, SmemTensorType) else "bulk"


class LowerToAsync(LoweringPass):

    def _wrap(
        self, inner: Op, kind: str, attrs: dict[str, object],
    ) -> tuple[list[Op], list[object]]:
        """Wrap ``inner`` in ``async(lock){ inner } + await(lock)``.

        The label is tagged ``{label}_async`` on the async op and
        ``{label}_await`` on the await (mirroring each other), so label-matching
        passes see it on the top-level ops rather than buried in the block.
        """
        assert isinstance(inner, (MatmulOp, MoveOp, TmaMoveOp))
        lock = AllocOp.build_with_fresh_id(LockType.build(kind=kind))
        label = attrs.get("label")
        async_op = AsyncOp.build(
            lock.result(),
            collective=inner.collective,
            body=Block(ops=[inner]),
            attrs={"label": f"{label}_async"} if label is not None else None,
        )
        if kind == "wgmma":
            return [lock, async_op], []
        await_op = AwaitOp.build(
            lock.result(),
            attrs={"label": f"{label}_await"} if label is not None else None,
        )
        return [lock, async_op, await_op], []

    @lowers(AsyncOp)
    def lower_async(
        self, op: AsyncOp,
    ) -> tuple[list[Op], list[object]]:
        # No-op: already async. Returning it untouched (rather than falling
        # through to pass-through) keeps us from recursing into the body and
        # lowering its inner op a second time — idempotency for a second run.
        return [op], list(op.results)

    @lowers(MoveOp)
    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not op.attrs.get("tma"):
            return None
        dst = cast(Value, self.value_map[op.dst])
        src = cast(Value, self.value_map[op.src])
        move = MoveOp.build(
            dst=dst,
            src=src,
            collective=op.collective,
            multicast=op.multicast,
            attrs=op.extra_attrs,
        )
        return self._wrap(move, _tma_lock_kind(dst), op.attrs)

    @lowers(TmaMoveOp)
    def lower_tma_move(
        self, op: TmaMoveOp,
    ) -> tuple[list[Op], list[object]]:
        dst = cast(Value, self.value_map[op.dst])
        src = cast(Value, self.value_map[op.src])
        indices = [cast(Value, self.value_map[v]) for v in op.indices]
        tma = TmaMoveOp.build(
            dst=dst,
            src=src,
            indices=indices,
            collective=op.collective,
            l2_cache_hint=op.l2_cache_hint,
            multicast=op.multicast,
            attrs=op.extra_attrs,
        )
        return self._wrap(tma, _tma_lock_kind(dst), op.attrs)

    @lowers(MatmulOp)
    def lower_matmul(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]] | None:
        lhs = cast(Value, self.value_map[op.lhs])
        rhs = cast(Value, self.value_map[op.rhs])
        dst = cast(Value, self.value_map[op.dst])
        # Only wgmma matmuls (smem×smem→wgrmem, 4 warps) are async; plain FMA
        # matmuls stay synchronous. Same idea as ``lower_move``'s tma gate.
        # TODO: lower *all* matmuls as async (FMA ones too), wrapping every
        # matmul uniformly — the non-async ones get a trivial lock whose await
        # is a no-op (async + immediate await ≡ sync). That removes this gate
        # and the sync/async split entirely.
        if not (isinstance(lhs.type, SmemTensorType)
                and isinstance(rhs.type, SmemTensorType)
                and isinstance(dst.type, WgrmemTensorType)
                and dst.type.warps == 4):
            return None
        matmul = MatmulOp.build(
            lhs=lhs,
            rhs=rhs,
            dst=dst,
            collective=op.collective,
            attrs=op.extra_attrs,
            init=op.init,
            init_value=(
                cast(Value, self.value_map[op.init_value])
                if op.init_value is not None else None
            ),
        )
        return self._wrap(matmul, "wgmma", op.extra_attrs)


def lower_to_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToAsync().run(module)
