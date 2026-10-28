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
pipelining a load's await moves to the loop top, away from the load, so the
load's run is replayed unfused rather than leaving the load unawaited. This also
lets a later call skip loads that were pipelined as singletons, such as a
multicast LHS next to a local RHS. wgmma matmuls fuse
after pipelining + kk-unroll, where they become consecutive. Both points just
call ``fuse_async(module)``; the single-op + same-kind gating does the rest.
"""
from __future__ import annotations

from itertools import takewhile

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
    join_collectives,
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
        new_ops: list[Op] = []
        group_lock: Value | None = None
        group_kind: str | None = None
        group_attrs: dict[str, object] | None = None
        group_collective: dict[CollectiveAxis, CollectiveLevel] | None = None
        # The members and the awaits dropped for them, as found. Fusing takes
        # the members' bodies; a run that cannot fuse is replayed as is.
        group_ops: list[Op] = []
        # Member locks still awaiting their (to-be-dropped) await. Each fused
        # async adds its lock; its await removes it. Left over at finalize,
        # the run is replayed.
        in_flight_locks: set[Value] = set()

        def finalize() -> None:
            nonlocal group_lock, group_kind, group_attrs, group_collective
            nonlocal in_flight_locks, group_ops
            if in_flight_locks:
                # A member's await is not adjacent (e.g. TMA loads seen *after*
                # pipelining, whose awaits sit at the loop top). Fusing would
                # leave that member's work unawaited, so replay the run as found.
                new_ops.extend(group_ops)
            elif group_collective is not None:
                assert group_lock is not None
                new_ops.append(AsyncOp.build(
                    group_lock,
                    collective=group_collective,
                    body=Block(ops=[
                        inner
                        for member in group_ops
                        if isinstance(member, AsyncOp)
                        for inner in member.body.ops
                    ]),
                    attrs=group_attrs,
                ))
                new_ops.append(AwaitOp.build(group_lock))
            group_lock = None
            group_kind = None
            group_attrs = None
            group_collective = None
            in_flight_locks = set()
            group_ops = []

        for index, op in enumerate(ops):
            kind = _fusion_kind(op)
            # Scheduling may deliberately move an await away from its
            # singleton async. That pair has already been processed and
            # cannot join a new fusion run. An await that follows a run of
            # same-kind launches and their awaits is still adjacent.
            if isinstance(op, AsyncOp) and op.lock in awaited_locks:
                run = takewhile(
                    lambda o: is_side_effect_free(o)
                    or isinstance(o, AwaitOp)
                    or _fusion_kind(o) == kind,
                    ops[index + 1:],
                )
                if not any(
                    isinstance(o, AwaitOp) and o.lock is op.lock for o in run
                ):
                    kind = None
            if kind is not None:
                assert isinstance(op, AsyncOp)
                if group_kind is not None:
                    assert group_lock is not None
                    if (
                        kind != group_kind
                        or op.lock.type != group_lock.type
                        or (
                            kind != "tma_load"
                            and op.collective != group_collective
                        )
                    ):
                        finalize()  # incompatible async ends the run
                if group_collective is None:
                    group_lock = op.lock
                    group_kind = kind
                    group_attrs = op.extra_attrs or None
                    group_collective = op.collective
                elif kind == "tma_load":
                    group_collective = join_collectives(
                        group_collective, op.collective,
                    )
                in_flight_locks.add(op.lock)
                group_ops.append(op)
            elif isinstance(op, AwaitOp) and op.lock in in_flight_locks:
                in_flight_locks.discard(op.lock)  # member's await; group covers it
                group_ops.append(op)
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
