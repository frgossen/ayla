"""Fuse consecutive same-kind async ops into one async group.

A run of consecutive single-op ``async(lock){ inner }`` ops of the same fusion
kind is merged into one ``async(lock){ inner; inner; … }`` block followed by a
single ``await``. The kind is decided by the op *inside* the async (see
:func:`_fusion_kind`), so what fuses is fixed by what codegen supports:

- **wgmma** matmuls — launch into one wgmma commit group, awaited once. Safe
  with no buffer-conflict analysis: a wgmma reads smem and writes wgrmem, so one
  matmul's output is never another's input; the only shared buffer is the
  accumulator (read+written by accumulating matmuls), ordered by the hardware
  within a commit group.
- **TMA gmem→smem loads** — share one mbarrier (one combined ``arrive.expect_tx``
  over their summed bytes), awaited once, instead of one mbarrier per load. Safe
  because the loads target distinct smem buffers and are all consumed after the
  single await (fastcu's single-``full``-barrier idea). TMA *stores* are not
  fused (no fused-store codegen, and unvalidated).

Only *consecutive* ops of the *same* kind fuse; a different/unsupported op ends
the run. An already-fused (multi-op) async, or a singleton whose await has been
moved away by scheduling, is left untouched, so the pass is idempotent.
Side-effect-free ops between members (lock allocs, tile views) are transparent
— emitted before the merged block — and the per-member awaits are dropped.

**Placement matters** (caller's responsibility): fuse TMA loads *before*
pipelining, while each stage's loads and awaits are still adjacent. After
pipelining a load's await moves to the loop top, away from the load, so it would
not be dropped onto the fused lock — caught by the await assert in
:meth:`FuseAsync.run_block` rather than silently deadlocking. wgmma matmuls fuse
after pipelining + kk-unroll, where they become consecutive. Both points just
call ``fuse_async(module)``; the single-op + same-kind gating does the rest.
"""
from __future__ import annotations

from ..gpu_dialect import (
    AsyncOp,
    AwaitOp,
    CollectiveAxis,
    CollectiveLevel,
    MatmulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    is_side_effect_free,
)
from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass


def _fusion_kind(op: Op) -> str | None:
    """Fusion category of ``op`` if it is a single-op async we can fuse, else
    ``None``. Multi-op (already-fused) asyncs return ``None`` — idempotency."""
    if not isinstance(op, AsyncOp):
        return None
    body = op.body.ops
    if len(body) != 1:
        return None
    inner = body[0]
    if isinstance(inner, MatmulOp):
        return "wgmma"  # matmuls → one wgmma commit group
    if (isinstance(inner, MoveOp)
            and inner.attrs.get("tma")
            and isinstance(inner.dst.type, SmemTensorType)):  # dst smem ⇒ load
        return "tma_load"  # gmem→smem loads → one mbarrier
    return None


class FuseAsync(Pass):

    def run_block(self, block: Block) -> None:
        super().run_block(block)

        ops = block.ops
        awaited_locks = {
            op.lock for op in ops if isinstance(op, AwaitOp)
        }

        def has_adjacent_await(index: int, lock: Value) -> bool:
            """Whether the next effectful op is this async's await."""
            for candidate in ops[index + 1:]:
                if is_side_effect_free(candidate):
                    continue
                return (
                    isinstance(candidate, AwaitOp)
                    and candidate.lock is lock
                )
            return False

        new_ops: list[Op] = []
        group_inner: list[Op] = []
        group_lock: Value | None = None
        group_kind: str | None = None
        group_attrs: dict[str, object] | None = None
        group_collective: dict[CollectiveAxis, CollectiveLevel] | None = None
        # Member locks still awaiting their (to-be-dropped) await. Each fused
        # async adds its lock; its await removes it. Must be empty at finalize.
        in_flight_locks: set[Value] = set()

        def finalize() -> None:
            nonlocal group_inner, group_lock, group_kind, group_attrs, group_collective
            nonlocal in_flight_locks
            if group_collective is not None:
                assert group_lock is not None
                # Every member's await must have been dropped (its lock removed)
                # so the group's single await covers it. A leftover lock means a
                # member's await wasn't adjacent (e.g. TMA loads fused *after*
                # pipelining, whose awaits sit at the loop top) — that member's
                # work would never be awaited and the kernel would deadlock.
                assert not in_flight_locks, (
                    f"fuse_async: {len(in_flight_locks)} fused async op(s) have "
                    f"no adjacent await; members' awaits must be adjacent (fuse "
                    f"TMA loads before pipelining)"
                )
                new_ops.append(AsyncOp.build(
                    group_lock,
                    collective=group_collective,
                    body=Block(ops=group_inner),
                    attrs=group_attrs,
                ))
                new_ops.append(AwaitOp.build(group_lock))
            group_inner = []
            group_lock = None
            group_kind = None
            group_attrs = None
            group_collective = None
            in_flight_locks = set()

        for index, op in enumerate(ops):
            kind = _fusion_kind(op)
            if (
                kind is not None
                and isinstance(op, AsyncOp)
                and op.lock in awaited_locks
                and not has_adjacent_await(index, op.lock)
            ):
                # Scheduling may deliberately move an await away from its
                # singleton async. That pair has already been processed and
                # cannot safely participate in a new fusion run.
                kind = None
            if kind is not None:
                assert isinstance(op, AsyncOp)
                if group_kind is not None:
                    assert group_lock is not None
                    if (
                        kind != group_kind
                        or op.lock.type != group_lock.type
                        or op.collective != group_collective
                    ):
                        finalize()  # incompatible async ends the run
                if group_collective is None:
                    group_lock = op.lock
                    group_kind = kind
                    group_attrs = op.extra_attrs or None
                    group_collective = op.collective
                in_flight_locks.add(op.lock)
                group_inner.extend(op.body.ops)
            elif isinstance(op, AwaitOp) and op.lock in in_flight_locks:
                in_flight_locks.discard(op.lock)  # member's await; group covers it
            elif is_side_effect_free(op):
                # Transparent: safe to emit before the merged block since its
                # operands are defined outside the (single-kind) group.
                new_ops.append(op)
            else:
                finalize()
                new_ops.append(op)

        finalize()
        block.ops = new_ops


def fuse_async(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return FuseAsync().run(module)
