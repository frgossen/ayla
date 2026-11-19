"""Compact operand pipelining with dynamic banks and explicit event epochs.

The semantic region is a completed serial reduction. Its implementation is
expanded after synchronization analysis, using the ordinary engine lowering.
This keeps the dynamic event generations inside a bounded ownership protocol;
it does not relax the alias rules of unrelated operations.
"""

from ..base.ir import Block, Value, clone_op, clone_type, find_ops_of_type, iter_ops
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
    IndexMulOp,
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
    root_buffers,
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


class PipelineRolledK(LoweringPass):
    def __init__(self, depth, label="k"):
        if type(depth) is not int or depth < 2:
            raise ValueError("rolled K depth must be an integer >= 2")
        super().__init__(depth=depth, label=label)

    def run(self, module):
        if module.body.ops[0].attrs.get("compact_output_pipeline"):
            from .peel_loop import PeelLoop

            module = PeelLoop(self.label, back_iterations=1).run(module)
        for loop in list(find_ops_of_type(module, ForOp)):
            if loop.attrs.get("label") != self.label:
                continue
            parent = loop.parent_block
            ops, _ = self.lower_for(loop)
            index = parent.ops.index(loop)
            parent.ops[index : index + 1] = ops
        return module

    @lowers(ForOp)
    def lower_for(self, op):
        if op.attrs.get("label") != self.label:
            return None
        if op.step != 1 or op.trip_count < 1:
            raise ValueError("rolled K requires a nonempty unit-step reduction")
        load, mma = _parts(op)
        banks = []
        for move in load.body.ops:
            roots = root_buffers(move.dst)
            if len(roots) != 1:
                raise ValueError(
                    "rolled K requires independently allocated operand banks"
                )
            root = next(iter(roots))
            if root in [x[0] for x in banks]:
                continue
            if not isinstance(root.definition, AllocOp) or not isinstance(
                root.type, SmemTensorType
            ):
                raise ValueError("rolled K operand must be shared allocation")
            single = clone_type(root.type)
            if single.rank != 2 or tuple(single.strides) != (single.sizes[1], 1):
                raise ValueError("rolled K currently requires row-major operand banks")
            layout = physical_layout(single)
            blocked = layout is not None and layout[1] == "block_major"
            if blocked and single.sizes[1] % layout[0]:
                raise ValueError("rolled K requires complete physical K blocks")
            arena = clone_type(single)
            if blocked:
                # A block-major stage stores each K block after all its M
                # rows. Extending M interleaves banks within those blocks and
                # changes the layout written by a stage-sized TMA transfer.
                # Extend K instead: every stage remains one contiguous bank,
                # with the same M extent and physical K-block stride.
                arena.sizes = [single.sizes[0], single.sizes[1] * self.depth]
                arena.attrs["strides"] = [arena.sizes[1], 1]
            else:
                arena.sizes = [single.sizes[0] * self.depth, single.sizes[1]]
            root.type = arena
            view = TileOp.build(root, [0, 0], single)
            live = set(root.users())
            for user, index in list(root.uses):
                if user is not view and user in live:
                    user.operands[index] = view.result()
            block = root.definition.parent_block
            block.ops.insert(block.ops.index(root.definition) + 1, view)
            banks.append((root, view.result()))
        region = OperandPipelineOp.build(op, self.depth)
        region.operands = [view for _, view in banks]
        if op.parent_block is not None:
            parent = op.parent_block.parent_op
            while parent is not None and not (
                isinstance(parent, ForOp) and parent.attrs.get("label") != self.label
            ):
                parent = parent.parent_block.parent_op if parent.parent_block else None
            if parent is not None:
                kernel = parent
                while kernel.parent_block and kernel.parent_block.parent_op:
                    kernel = kernel.parent_block.parent_op
                    if hasattr(kernel, "num_threads"):
                        break
                if kernel.attrs.get("compact_output_pipeline"):
                    group = mma.lock.type.attrs.get("cta_group", 1)
                    ready = AllocOp.build_with_fresh_id(
                        LockType.build(
                            "mbar", {"count": self.depth, "cta_group": group}
                        ),
                        attrs={"label": "operand_bank_ready"},
                    )
                    free = AllocOp.build_with_fresh_id(
                        LockType.build(
                            "tcgen05", {"count": self.depth, "cta_group": group}
                        ),
                        attrs={"label": "operand_bank_release"},
                    )
                    kernel.body.ops[0:0] = [
                        ready,
                        free,
                        BarrierOp.build("cluster" if group == 2 else "block"),
                    ]
                    region.operands.extend([parent.iv, ready.result(), free.result()])
                    region.attrs["carried_events"] = True
        return [region], []


class LowerRolledK(LoweringPass):
    @lowers(OperandPipelineOp)
    def lower_pipeline(self, op):
        load, mma = _parts(op)
        depth, lb, ub = (op.attrs[x] for x in ("depth", "lb", "ub"))
        group = mma.lock.type.attrs.get("cta_group", 1)
        load_issuer = execution_issuer_warp(load)
        mma_issuer = execution_issuer_warp(mma)
        carried = op.attrs.get("carried_events", False)
        if carried:
            job, ready_value, free_value = op.operands[-3:]
            banks = op.operands[:-3]
        else:
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
            if carried:
                # Each bank has its own number of generations per job. K need
                # not be divisible by depth, so a single job phase is unsound.
                neg = IndexConstantOp.build(-1)
                neg_bank = IndexMulOp.build(bank.result(), neg.result())
                first = IndexConstantOp.build(lb - 1 + depth)
                last = IndexConstantOp.build(ub - 1 + depth)
                first_n = IndexAddOp.build(first.result(), neg_bank.result())
                last_n = IndexAddOp.build(last.result(), neg_bank.result())
                first_q = IndexDivOp.build(first_n.result(), span.result())
                last_q = IndexDivOp.build(last_n.result(), span.result())
                minus_first = IndexMulOp.build(first_q.result(), neg.result())
                count = IndexAddOp.build(last_q.result(), minus_first.result())
                offset = IndexMulOp.build(job, count.result())
                full_epoch = IndexAddOp.build(epoch.result(), offset.result())
                phase = IndexModOp.build(full_epoch.result(), two.result())
                ops.extend(
                    [
                        neg,
                        neg_bank,
                        first,
                        last,
                        first_n,
                        last_n,
                        first_q,
                        last_q,
                        minus_first,
                        count,
                        offset,
                        full_epoch,
                        phase,
                    ]
                )
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
            ub=min(ub, lb + depth - 1),
            step=1,
            body=Block(arguments=[fill_iv], ops=fill_ops),
            attrs={"label": "rolled_k_fill", "unroll": False},
        )

        iv = Value(name=None, type=IndexType())
        body = []
        iteration(iv, mma, free_value, body)
        ahead = IndexConstantOp.build(depth - 1)
        future = IndexAddOp.build(iv, ahead.result())
        limit = IndexConstantOp.build(ub)
        active = IndexLtOp.build(future.result(), limit.result())
        body.extend([ahead, future, limit, active])
        refill = []
        minus_one = IndexConstantOp.build(-1)
        previous = IndexAddOp.build(iv, minus_one.result())
        first = IndexConstantOp.build(lb)
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
        tail = []
        last = IndexConstantOp.build(ub - 1)
        tail.append(last)
        done = event(free_value, last.result(), tail)
        # Same-issuer TCGEN operations are ordered. The final event covers
        # every preceding batch, and the collective publishes it to readers.
        if carried or kernel is not None and kernel.attrs.get('cluster_bank_pipeline'):
            # The final engine completion does not order delivery of every
            # preceding multicast notification. Carried events additionally
            # require a successful wait on every phase before another arrival
            # on that event, even when the final MMA already covers its data.
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
            *([] if carried else [ready, free, BarrierOp.build(scope)]),
            fill,
            main,
            *tail,
        ], []
