"""Compact operand pipelining with dynamic banks and explicit event epochs.

The semantic region is a completed serial reduction. Its implementation is
expanded after synchronization analysis, using the ordinary engine lowering.
This keeps the dynamic event generations inside a bounded ownership protocol;
it does not relax the alias rules of unrelated operations.
"""

from ..base.ir import Block, Value, clone_op, iter_ops
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    ForOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexType,
    LockType,
    LockViewOp,
    OperandPipelineOp,
    SmemTensorType,
    GmemTensorType,
    TmaDescType,
    MoveOp,
    TmaMoveOp,
    TmemTensorType,
    MatmulOp,
    TileOp,
    is_side_effect_free,
    physical_layout,
    execution_issuer_warp,
)


def _parts(op):
    issues = [x for x in op.body.ops if isinstance(x, AsyncOp)]
    if len(issues) != 2 or [x.lock.type.kind for x in issues] != ["mbar", "tcgen05"]:
        raise ValueError("rolled K requires one combined TMA and one MMA batch")
    load, mma = issues
    if any(len(x.completion_locks) != 1 for x in issues):
        raise ValueError("rolled K requires exactly one completion per issue")
    if any(isinstance(x, AllocOp) for x in op.body.ops):
        raise ValueError("rolled K requires hoisted operand and event allocations")
    if any(
        not isinstance(x, (MoveOp, TmaMoveOp))
        or not isinstance(x.dst.type, SmemTensorType)
        or not isinstance(x.src.type, (GmemTensorType, TmaDescType))
        for x in load.body.ops
    ):
        raise ValueError("rolled K requires global-to-shared operand transfers")
    if any(
        not isinstance(x, MatmulOp)
        or not isinstance(x.dst.type, TmemTensorType)
        or not x.accumulate
        for x in mma.body.ops
    ):
        raise ValueError(
            "rolled K requires accumulating TCGEN batches after a peeled first step"
        )
    waits = [x for x in op.body.ops if isinstance(x, AwaitOp)]
    if len(waits) != 2 or {x.lock for x in waits} != {load.lock, mma.lock}:
        raise ValueError("rolled K requires a completed serial iteration")
    if [x for x in op.body.ops if not is_side_effect_free(x)] != [
        load,
        waits[0],
        mma,
        waits[1],
    ] or waits[0].lock is not load.lock:
        raise ValueError("rolled K requires load/await then MMA/await ordering")
    if any(
        not is_side_effect_free(x) and x not in [*issues, *waits] for x in op.body.ops
    ):
        raise ValueError("rolled K cannot contain unrelated effects")
    if any(x.attrs.get("participant_warps")
           for x in [load, mma, *load.body.ops, *mma.body.ops]):
        raise ValueError("rolled K requires a shared loop without participant regions")
    if mma.lock.type.attrs.get("cta_group", 1) == 2 and (
        execution_issuer_warp(load) or execution_issuer_warp(mma)
    ):
        raise ValueError("paired rolled K requires the common elected issuer")
    if any(execution_issuer_warp(inner) != execution_issuer_warp(issue)
           for issue in [load, mma] for inner in issue.body.ops):
        raise ValueError("rolled K requires one elected issuer per engine batch")
    return load, mma


class LowerRolledK(LoweringPass):
    @lowers(OperandPipelineOp)
    def lower_pipeline(self, op):
        load, mma = _parts(op)
        depth, lb, ub = (op.attrs[x] for x in ("depth", "lb", "ub"))
        queue = op.attrs.get('mma_queue_depth', 1)
        lead = depth - queue
        group = mma.lock.type.attrs.get("cta_group", 1)
        load_issuer = execution_issuer_warp(load)
        mma_issuer = execution_issuer_warp(mma)
        if op.attrs.get("carried_events"):
            raise ValueError("per-job carried events require the cross-job resource pipeline")
        ready = AllocOp.build_with_fresh_id(
            LockType.build("mbar", {"count": depth, "cta_group": group})
        )
        free = AllocOp.build_with_fresh_id(
            LockType.build("tcgen05", {
                **mma.lock.type.attrs, "count": depth, "cta_group": group,
            })
        )
        ready_value, free_value = ready.result(), free.result()
        banks = op.operands
        from ..base.ir import find_ancestor_of_type
        from ..gpu_dialect import KernelOp
        kernel = find_ancestor_of_type(op, KernelOp)
        scope = "cluster" if group == 2 or kernel is not None and kernel.is_clustered else "block"

        def event(arena, logical, ops, *, observation=True):
            if not observation:
                span = IndexConstantOp.build(depth)
                bank = IndexModOp.build(logical, span.result())
                unused = IndexConstantOp.build(0)
                view = LockViewOp.build(arena, bank.result(), unused.result())
                ops.extend([span, bank, unused, view])
                return view.result()
            span = IndexConstantOp.build(depth)
            shift = IndexConstantOp.build(-lb)
            epoch_index = IndexAddOp.build(logical, shift.result())
            epoch = IndexDivOp.build(epoch_index.result(), span.result())
            two = IndexConstantOp.build(2)
            phase = IndexModOp.build(epoch.result(), two.result())
            bank = IndexModOp.build(logical, span.result())
            ops.extend([span, shift, epoch_index, epoch, two, phase, bank])
            view = LockViewOp.build(arena, bank.result(), phase.result())
            ops.append(view)
            return view.result()

        pure = [
            x
            for x in op.body.ops
            if is_side_effect_free(x) and not isinstance(x, AllocOp)
        ]

        def iteration(logical, issue, arena, ops):
            values = {op.body.arguments[0]: logical}
            span = IndexConstantOp.build(depth)
            bank = IndexModOp.build(logical, span.result())
            ops.extend([span, bank])
            for original in banks:
                root = original.definition.arg
                layout = physical_layout(original.type)
                indices = ([0, bank.result()] if layout is not None
                           and layout[1] == "block_major" else [bank.result(), 0])
                view = TileOp.build(root, indices, original.type)
                ops.append(view)
                values[original] = view.result()
            for x in pure:
                ops.append(clone_op(x, values))
            lock = event(arena, logical, ops, observation=False)
            values[issue.lock] = lock
            if issue is mma:
                # Data readiness is observed by the actual MMA issuer. On a
                # pair, the existing await lowering polls only the leader.
                data = event(ready_value, logical, ops)
                ops.append(AwaitOp.build(data, {"waiter_warps": [mma_issuer]}))

            def rebase(value):
                if value in values:
                    return values[value]
                definition = value.definition
                if not isinstance(definition, TileOp):
                    return value
                parent = rebase(definition.arg)
                if parent is definition.arg:
                    return value
                copied = clone_op(definition, values)
                ops.append(copied)
                return copied.result()

            for inner in iter_ops(issue):
                for value in inner.operands:
                    rebase(value)
            copied = clone_op(issue, values)
            copied.attrs.pop("release_prior", None)
            ops.append(copied)

        fill_iv = Value(name=None, type=IndexType())
        fill_ops = []
        iteration(fill_iv, load, ready_value, fill_ops)
        fill = ForOp.build(
            lb=lb,
            ub=min(ub, lb + lead),
            step=1,
            body=Block(arguments=[fill_iv], ops=fill_ops),
            attrs={"label": "rolled_k_fill", "unroll": False},
        )

        iv = Value(name=None, type=IndexType())
        body = []
        iteration(iv, mma, free_value, body)
        ahead = IndexConstantOp.build(lead)
        future = IndexAddOp.build(iv, ahead.result())
        limit = IndexConstantOp.build(ub)
        active = IndexLtOp.build(future.result(), limit.result())
        body.extend([ahead, future, limit, active])
        refill = []
        minus_one = IndexConstantOp.build(-queue)
        previous = IndexAddOp.build(iv, minus_one.result())
        first = IndexConstantOp.build(lb + queue - 1)
        reused = IndexLtOp.build(first.result(), iv)
        refill.extend([minus_one, previous, first, reused])
        reuse_ops = []
        reuse_event = event(free_value, previous.result(), reuse_ops)
        # The refilling issuer observes operand-read completion directly.
        # The MMA issuer can continue issuing other ready banks in this same
        # loop. Bank-ready completion then bounds its progress and proves the
        # old read-completion generation was observed before bank/event reuse.
        reuse_ops.append(AwaitOp.build(reuse_event, {"waiter_warps": [load_issuer]}))
        refill.append(IfOp.build(reused.result(), Block(ops=reuse_ops)))
        iteration(future.result(), load, ready_value, refill)
        body.append(IfOp.build(active.result(), Block(ops=refill)))
        if kernel is not None and kernel.attrs.get("rolled_refill_before_compute", False):
            # Earlier TMA issue can hide cold operand latency, but observing
            # the previous bank before MMA limits outstanding compute batches.
            # Keep this measured ordering choice explicitly opt-in.
            mma_at = next(i for i, item in enumerate(body) if isinstance(item, AsyncOp))
            body = body[mma_at + 1 :] + body[: mma_at + 1]
        main = ForOp.build(
            lb=lb,
            ub=ub,
            step=1,
            body=Block(arguments=[iv], ops=body),
            attrs={"label": "rolled_k_main", "unroll": False},
        )
        first_batch = []
        if op.attrs.get('initialize_accumulator') or queue > 1:
            from ._loop_utils import clone_loop_iteration

            # The steady loop starts with valid refill/release ordinals, so
            # its bank and phase arithmetic can become register recurrences.
            for offset in range(min(queue, main.trip_count - 1)):
                initial_iv = IndexConstantOp.build(lb + offset)
                batch = [initial_iv, *clone_loop_iteration(main, initial_iv.result())]
                if offset == 0 and op.attrs.get('initialize_accumulator'):
                    initial_mma = next(inner for root in batch for inner in iter_ops(root)
                        if isinstance(inner, AsyncOp) and inner.lock.type.kind == 'tcgen05')
                    initial_mma.body.ops[0].attrs['accumulate'] = False
                first_batch.extend(batch)
            main.lb += min(queue, main.trip_count - 1)
        from ..pipelines.ring_schedule import LoopSchedule
        from .unroll_loop import UnrollLoop
        shape = LoopSchedule(**(kernel.attrs.get("loop_schedule", {}).get("reduction", {})
                                if kernel is not None else {}))
        main_ops = [main]
        if shape.peel_front or shape.peel_back:
            from .peel_loop import PeelLoop
            main_ops, _ = PeelLoop("rolled_k_main", shape.peel_front, shape.peel_back).lower_for(main)
        if shape.unroll > 1:
            transformed = []
            for piece in main_ops:
                if isinstance(piece, ForOp) and piece.trip_count >= shape.unroll:
                    unrolled, _ = UnrollLoop(shape.unroll, "rolled_k_main").lower_for(piece)
                    transformed.extend(unrolled)
                else:
                    transformed.append(piece)
            main_ops = transformed
        tail = []
        last = IndexConstantOp.build(ub - 1)
        tail.append(last)
        done = event(free_value, last.result(), tail)
        # Same-issuer TCGEN operations are ordered. The final event covers
        # every preceding batch, and the collective publishes it to readers.
        if kernel is not None and kernel.attrs.get('cluster_bank_pipeline'):
            # The final engine completion does not order delivery of every
            # preceding multicast notification. Every phase must be observed
            # before another arrival, even when final MMA covers its data.
            # Observe each bank's final generation before the next job.
            drain_iv = Value(name=None, type=IndexType())
            drain_ops = []
            drain_event = event(free_value, drain_iv, drain_ops)
            drain_ops.append(AwaitOp.build(drain_event, {'waiter_warps': [mma_issuer]}))
            tail.append(ForOp.build(
                lb=max(lb, ub - depth), ub=ub, step=1,
                body=Block(arguments=[drain_iv], ops=drain_ops),
                attrs={'label': 'rolled_k_drain', 'unroll': False},
            ))
            tail.append(BarrierOp.build(scope))
        else:
            tail.extend(
                [AwaitOp.build(done, {"waiter_warps": [mma_issuer]}), BarrierOp.build(scope)]
            )
        return [
            ready, free, BarrierOp.build(scope),
            fill,
            *first_batch,
            *main_ops,
            *tail,
        ], []
