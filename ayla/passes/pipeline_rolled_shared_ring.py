"""Compact persistent operand/frame scheduling, without separate actor loops.

Bank ownership uses a global K ordinal, not a per-output-job phase. The serial
region is verified before sync insertion; its bounded implementation is
expanded afterwards, as for operand_pipeline. All engines use the ordinary
AsyncOp/LockViewOp lowering, including paired TMA and MMA.
"""

import math

from ..base.ir import (
    Block,
    Value,
    clone_op,
    dependency_slice,
    clone_type,
    find_ancestor_of_type,
    find_ops_of_type,
    iter_ops,
)
from ..base.pipeline import Pass
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    CollectiveAxis,
    DeallocOp,
    ForOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    IndexType,
    KernelOp,
    LockViewOp,
    MatmulOp,
    OperandPipelineOp,
    PersistentPipelineOp,
    TileOp,
    TmemTensorType,
    effective_operands,
    execution_issuer_warp,
    is_side_effect_free,
    physical_layout,
)
from ..pipelines.async_resources import allocate_event, SplitBarrier, tma_writeback_fragments
from ..pipelines.ring_schedule import RingSchedule, reject_unaccepted_effects
from .pipeline_rolled_k import _parts


def _source(op):
    [loop] = op.body.ops
    body = list(loop.body.ops)
    guards = [x for x in body if isinstance(x, IfOp)]
    if guards:
        if len(guards) != 1:
            raise ValueError("rolled shared ring requires one output validity guard")
        guard = guards[0]
        body[body.index(guard) : body.index(guard) + 1] = guard.body.ops
    regions = [x for x in body if isinstance(x, OperandPipelineOp)]
    if len(regions) != 1:
        raise ValueError("rolled shared ring requires one compact K reduction")
    reduction = regions[0]
    load, mma = _parts(reduction)
    if reduction.attrs["lb"] != 1 or reduction.attrs.get("carried_events"):
        raise ValueError("rolled shared ring requires only the first K step peeled")
    fragments = tma_writeback_fragments(body, TmemTensorType)
    if not fragments or len(reduction.operands) != 2:
        raise ValueError(
            "rolled shared ring requires two operands and collective writeback"
        )
    if any(
        execution_issuer_warp(x) or x.attrs.get("participant_warps")
        for x in [load, mma, *load.body.ops, *mma.body.ops]
    ):
        raise ValueError("rolled shared ring requires the common elected issuer")
    peeled = [x for x in body[: body.index(reduction)] if isinstance(x, AsyncOp)]
    if (
        len(peeled) != 2
        or [x.lock.type.kind for x in peeled] != ["mbar", "tcgen05"]
        or not peeled[1].body.ops
        or peeled[1].body.ops[0].accumulate
        or any(
            not isinstance(x, MatmulOp) or (i and not x.accumulate)
            for i, x in enumerate(peeled[1].body.ops)
        )
    ):
        raise ValueError("rolled shared ring requires an initializing first MMA batch")
    accepted = {
        reduction,
        *peeled,
        *(x for f in fragments for x in f),
        *(x for x in body if isinstance(x, AllocOp) and x.result() is op.operands[0]),
    }
    reject_unaccepted_effects(body, accepted, "rolled shared ring cannot discard unrelated effects")
    return loop, reduction, load, mma, fragments


def prepare_rolled_ring(module, context, *, operand_stages, overlap, job_regions,
                        mma_queue_depth):
    kernel, loop, allocation = context.kernel, context.loop, context.allocation
    arena = allocation.result()
    single = clone_type(arena.type)
    group, grid, tiles = context.group, context.grid, context.tile_count
    region = PersistentPipelineOp.build(loop, arena, operand_stages)
    region.attrs.update(
        overlap=overlap,
        tile_count=tiles,
        grid=grid,
        slot_columns=single.sizes[1],
        mma_queue_depth=mma_queue_depth,
        job_regions=job_regions,
    )
    if job_regions:
        if (
            group != 2
            or kernel.block_dim != (128, 1, 1)
            or not overlap
            or mma_queue_depth != 1
        ):
            raise ValueError(
                "job regions require a paired common ring with four-warp fragments"
            )
        kernel.attrs["block_dim"] = (256, 1, 1)
    _, reduction, _, _, _ = _source(region)
    if operand_stages > reduction.attrs["ub"]:
        raise ValueError("rolled shared ring needs one K batch per initial bank")
    if allocation.parent_block is not kernel.body:
        allocation.parent_block.ops.remove(allocation)
        kernel.body.ops.insert(kernel.body.ops.index(loop), allocation)
    for dealloc in list(find_ops_of_type(kernel, DeallocOp)):
        if dealloc.value is arena:
            dealloc.parent_block.ops.remove(dealloc)
    # Preserve the logical tile shape while reserving both accumulators.
    # The late scheduler substitutes these views, never reallocates TMEM.
    arena.type = clone_type(single)
    arena.type.sizes = [single.sizes[0], single.sizes[1] * 2]
    arena.type.attrs["columns"] = single.columns * 2
    first = TileOp.build(arena, [0, 0], single)
    for user, index in list(arena.uses):
        if user not in (first, region):
            user.operands[index] = first.result()
    region.operands.append(first.result())
    kernel.body.ops.insert(kernel.body.ops.index(allocation) + 1, first)
    kernel.body.ops[kernel.body.ops.index(loop)] = region
    kernel.body.ops.append(DeallocOp.build(arena))
    # Buffers and events keep one identity through the loop backedge.
    for storage in list(find_ops_of_type(region, AllocOp)):
        storage.parent_block.ops.remove(storage)
        kernel.body.ops.insert(kernel.body.ops.index(region), storage)
    kernel.attrs.update(
        rolled_shared_ring=True,
        persistent_output_slots=2,
        shared_operand_output_ring=True,
        ordered_mma_completion=True,
    )
    kernel.attrs["name"] += "_rolled_shared_ring" + (
        "_overlap" if overlap else "_serial"
    )
    return module


class LowerRolledSharedRing(LoweringPass):
    @lowers(PersistentPipelineOp)
    def lower_pipeline(self, op):
        source, reduction, load, mma, fragments = _source(op)
        depth, steps = op.attrs["depth"], reduction.attrs["ub"]
        job_regions = op.attrs.get("job_regions", False)
        grid, tiles = op.attrs["grid"], op.attrs["tile_count"]
        arena, first_slot = op.operands
        group = arena.type.attrs.get("cta_group", 1)
        warps = find_ancestor_of_type(op, KernelOp).num_threads // 32
        overlap = op.attrs["overlap"]
        schedule = RingSchedule(depth, steps, len(fragments), overlap, job_regions)
        ops = []

        def const(n, dst):
            x = IndexConstantOp.build(n)
            dst.append(x)
            return x.result()

        def binary(cls, a, b, dst):
            x = cls.build(a, b)
            dst.append(x)
            return x.result()


        def event(kind, count, label, attrs=None):
            x = allocate_event(kind, label,
                               {**(attrs or {}), **({"count": count} if count else {})})
            ops.append(x)
            return x.result()

        ready = event("mbar", depth, "operand_bank_ready", {"cta_group": group})
        free = event("tcgen05", depth, "operand_bank_release", {"cta_group": group})
        frame_ready = [
            event("tcgen05", None, f"accumulator_ready_{s}", {"cta_group": group})
            for s in range(2)
        ]
        frame_free = [SplitBarrier.build(group, warps, f"last_tmem_read_{s}")
                      for s in range(2)]
        ops.extend(release.allocation for release in frame_free)
        slots = []
        for s in range(2):
            view = TileOp.build(arena, [0, s], first_slot.type)
            ops.append(view)
            slots.append(view.result())
        # Bulk waits observe the issuing thread's FIFO. This compact schedule
        # conservatively drains it before staging-buffer reuse.
        store_locks = list(dict.fromkeys(fragment[-1].lock for fragment in fragments))
        span, stage_count, two, one, zero, minus_one = [
            const(v, ops) for v in (depth, steps, 2, 1, 0, -1)
        ]
        gx, limit = const(grid, ops), const(tiles, ops)
        bx = BlockIdOp.build(CollectiveAxis.X)
        ops.append(bx)

        def valid(job, dst):
            linear = binary(IndexMulOp, job, gx, dst)
            tile = binary(IndexAddOp, bx.result(), linear, dst)
            return binary(IndexLtOp, tile, limit, dst)

        def logical(job, k, dst):
            return binary(IndexAddOp, binary(IndexMulOp, job, stage_count, dst), k, dst)

        def bank_event(storage, ordinal, dst):
            bank = binary(IndexModOp, ordinal, span, dst)
            epoch = binary(IndexDivOp, ordinal, span, dst)
            phase = binary(IndexModOp, epoch, two, dst)
            view = LockViewOp.build(storage, bank, phase)
            dst.append(view)
            return view.result(), bank

        def copy_work(originals, job, k, slot, ordinal, dst):
            values = {
                source.iv: job,
                reduction.body.arguments[0]: k,
                first_slot: slots[slot],
            }
            bank = binary(IndexModOp, ordinal, span, dst)
            for view in reduction.operands:
                layout = physical_layout(view.type)
                indices = (
                    [0, bank] if layout and layout[1] == "block_major" else [bank, 0]
                )
                dynamic = TileOp.build(view.definition.arg, indices, view.type)
                dst.append(dynamic)
                values[view] = dynamic.result()

            dependencies = dependency_slice(
                (operand for original in originals for nested in iter_ops(original)
                 for operand in effective_operands(nested)),
                lambda value: value not in values
                and not isinstance(value.definition, AllocOp)
                and is_side_effect_free(value.definition),
                effective_operands,
            )
            dst.extend(clone_op(producer, values) for producer in dependencies)
            return [clone_op(original, values) for original in originals]

        def issue_load(job, k, ordinal, dst):
            copied = copy_work([load], job, k, 0, ordinal, dst)[0]
            copied.operands = [bank_event(ready, ordinal, dst)[0]]
            copied.attrs.update(label="ring_operand_issue")
            dst.append(copied)

        def split(slot, phase):
            return frame_free[slot].barrier(
                phase, "last_tmem_read" if phase == "arrive" else "tmem_slot_free")

        def output_fragment(job, slot, index, dst, *, finish=False):
            inner = []
            copied = copy_work(fragments[index], job, zero, slot, zero, inner)
            inner.extend(schedule.writeback_fragment(
                copied, index, frame_ready[slot], frame_free[slot], finish=finish,
                explicit_barriers=True,
            ))
            predicate = valid(job, dst)
            region = IfOp.build(predicate, Block(ops=inner))
            if job_regions:
                region.attrs["writeback_slot"] = slot
            dst.append(region)

        def compute(job, slot, dst):
            body = []
            # First uses have no old reader; subsequent writes acquire only
            # their own slot's last-read release, not the output store queue.
            reuse = binary(IndexLtOp, one, job, body)
            body.append(IfOp.build(reuse, Block(ops=[split(slot, "wait")])))
            iv = Value(name=None, type=IndexType())
            kbody = []
            for action in schedule.stage_order:
                if action == "issue":
                    ordinal = logical(job, iv, kbody)
                    data, _ = bank_event(ready, ordinal, kbody)
                    kbody.append(
                        AwaitOp.build(data, {"waiter_warps": [0], "label": "operand_ready"})
                    )
                    batch = copy_work([mma], job, iv, slot, ordinal, kbody)[0]
                    batch.operands = [bank_event(free, ordinal, kbody)[0]]
                    batch.attrs.update(label="ring_mma_issue")
                    is_first = binary(IndexLtOp, iv, one, kbody)
                    is_rest = binary(IndexLtOp, zero, iv, kbody)
                    last_k = const(steps - 2, kbody)
                    is_last = binary(IndexLtOp, last_k, iv, kbody)
                    not_last = binary(IndexLtOp, iv, const(steps - 1, kbody), kbody)
                    initial = clone_op(batch, {})
                    initial.body.ops[0].attrs["accumulate"] = False
                    final = clone_op(batch, {})
                    final.operands.append(frame_ready[slot])
                    kbody.extend(
                        [
                            IfOp.build(is_first, Block(ops=[initial])),
                            IfOp.build(
                                is_rest,
                                Block(
                                    ops=[
                                        IfOp.build(not_last, Block(ops=[batch])),
                                        IfOp.build(is_last, Block(ops=[final])),
                                    ]
                                ),
                            ),
                        ]
                    )
                elif action == "finish":
                    old = binary(IndexAddOp, job, const(-2, kbody), kbody)
                    pending = []
                    output_fragment(old, slot, len(fragments) - 1, pending, finish=True)
                    pending_valid = binary(IndexLtOp, one, job, kbody)
                    kbody.append(
                        IfOp.build(
                            pending_valid,
                            Block(ops=[IfOp.build(is_first, Block(ops=pending))]),
                        )
                    )
                elif action == "refill":
                    # Refill P-1 batches ahead, observing the previous MMA's
                    # bank release even across the output-job backedge.
                    prior = binary(IndexAddOp, ordinal, minus_one, kbody)
                    refill = [
                        AwaitOp.build(
                            bank_event(free, prior, kbody)[0],
                            {"waiter_warps": [0], "label": "bank_release"},
                        )
                    ]
                    future_k = binary(IndexAddOp, iv, const(schedule.prefetch_lead, refill), refill)
                    future = binary(IndexAddOp, ordinal, const(schedule.prefetch_lead, refill), refill)
                    current = binary(IndexLtOp, future_k, stage_count, refill)
                    successor = binary(IndexLtOp, const(steps - 1, refill), future_k, refill)
                    same = []
                    issue_load(job, future_k, future, same)
                    next_ops = []
                    next_job = binary(IndexAddOp, job, one, next_ops)
                    next_k = binary(IndexAddOp, future_k, const(-steps, next_ops), next_ops)
                    next_valid = valid(next_job, next_ops)
                    next_load = []
                    issue_load(next_job, next_k, future, next_load)
                    next_ops.append(IfOp.build(next_valid, Block(ops=next_load)))
                    refill.extend(
                        [
                            IfOp.build(current, Block(ops=same)),
                            IfOp.build(successor, Block(ops=next_ops)),
                        ]
                    )
                    has_prior = binary(IndexLtOp, zero, ordinal, kbody)
                    kbody.append(IfOp.build(has_prior, Block(ops=refill)))
                elif action == "writeback":
                    previous = binary(IndexAddOp, job, minus_one, kbody)
                    has_previous = binary(IndexLtOp, zero, job, kbody)
                    output_ops = []
                    for index in range(len(fragments)):
                        at = schedule.fragment_steps[index]
                        less = binary(IndexLtOp, iv, const(at + 1, output_ops), output_ops)
                        more = (
                            binary(IndexLtOp, const(at - 1, output_ops), iv, output_ops)
                            if at
                            else None
                        )
                        fragment = []
                        output_fragment(previous, 1 - slot, index, fragment)
                        # Index comparisons are unsigned: -1 < k cannot express
                        # the lower bound of a fragment scheduled at K step zero.
                        guarded = (
                            fragment
                            if more is None
                            else [IfOp.build(more, Block(ops=fragment))]
                        )
                        output_ops.append(IfOp.build(less, Block(ops=guarded)))
                    kbody.append(IfOp.build(has_previous, Block(ops=output_ops)))
            body.append(
                ForOp.build(
                    0,
                    steps,
                    1,
                    Block(arguments=[iv], ops=kbody),
                    {"label": "shared_ring_rolled_k", "unroll": False},
                )
            )
            if job_regions:
                previous = binary(IndexAddOp, job, minus_one, body)
                previous_valid = binary(IndexLtOp, zero, job, body)
                writeback = []
                for index in range(len(fragments)):
                    output_fragment(previous, 1 - slot, index, writeback)
                output_fragment(
                    previous, 1 - slot, len(fragments) - 1, writeback, finish=True
                )
                body.append(IfOp.build(previous_valid, Block(ops=writeback)))
            if not overlap:
                for index in range(len(fragments)):
                    output_fragment(job, slot, index, body)
            predicate = valid(job, dst)
            region = IfOp.build(predicate, Block(ops=body))
            if job_regions:
                region.attrs["compute_slot"] = slot
            dst.append(region)

        # Publish event initialization before any observer runs.
        ops.append(
            BarrierOp.build(
                "cluster" if group == 2 else "block",
                attrs={"initialization_publication": True},
            )
        )
        fill_iv = Value(name=None, type=IndexType())
        fill = []
        issue_load(zero, fill_iv, fill_iv, fill)
        ops.append(
            ForOp.build(
                0,
                depth,
                1,
                Block(arguments=[fill_iv], ops=fill),
                {"label": "shared_ring_fill", "unroll": False},
            )
        )
        job = Value(name=None, type=IndexType())
        jobs = []
        compute(job, 0, jobs)
        next_job = binary(IndexAddOp, job, one, jobs)
        compute(next_job, 1, jobs)
        count = math.ceil(tiles / grid)
        ops.append(
            ForOp.build(
                0,
                count,
                2,
                Block(arguments=[job], ops=jobs),
                {"label": "shared_operand_output_ring", "unroll": False},
            )
        )
        # Each CTA (or rank pair) has floor/ceil(grid) jobs. Do not consume an
        # event from a skipped tail job, or forget the carried final fragment.
        remainder = const(tiles - (count - 1) * grid, ops)
        has_last = binary(IndexLtOp, bx.result(), remainder, ops)
        for jobs_count, predicate in [
            (count, has_last),
            (
                count - 1,
                binary(
                    IndexLtOp,
                    const(tiles - (count - 1) * grid - 1, ops),
                    bx.result(),
                    ops,
                ),
            ),
        ]:
            tail = []
            last = const(jobs_count - 1, tail)
            if overlap:
                previous = const(jobs_count - 2, tail)
                if not job_regions:
                    output_fragment(
                        previous,
                        (jobs_count - 2) % 2,
                        len(fragments) - 1,
                        tail,
                        finish=True,
                    )
                for index in range(len(fragments)):
                    output_fragment(last, (jobs_count - 1) % 2, index, tail)
                output_fragment(
                    last, (jobs_count - 1) % 2, len(fragments) - 1, tail, finish=True
                )
            final_ordinal = logical(last, const(steps - 1, tail), tail)
            tail.append(
                AwaitOp.build(
                    bank_event(free, final_ordinal, tail)[0],
                    {"waiter_warps": [0], "label": "bank_terminal_drain"},
                )
            )
            ops.append(IfOp.build(predicate, Block(ops=tail)))
        ops.extend(
            AwaitOp.build(lock, {"label": "output_terminal_drain"})
            for lock in store_locks
        )
        ops.append(BarrierOp.build("cluster" if group == 2 else "block"))
        if job_regions:
            from .pipeline_job_participants import schedule_job_participants

            return schedule_job_participants((ops, []), op)
        return ops, []


class PrepareRolledSharedRing(Pass):
    """Expand and simplify the compact ring after synchronization analysis."""

    def run(self, module):
        if not module.body.ops[0].attrs.get("rolled_shared_ring"):
            return module
        from .cse import _cse_block
        from .dce import DCE
        from .licm import LICM
        from .peel_loop import PeelLoop
        from ..pipelines.canonicalize import Canonicalize

        module = LowerRolledSharedRing().run(module)
        module = PeelLoop(
            "shared_ring_rolled_k", front_iterations=1, back_iterations=1
        ).run(module)
        # Edge predicates become constant after peeling. Simplify before
        # descriptor hoisting, while memory effects are still visible.
        module = Canonicalize().run(module)
        module = Canonicalize().run(module)
        _cse_block(module.body, {})
        if module.body.ops[0].attrs.get("hoist_mma_descriptors"):
            LICM(hoist_pure_through_guards=True).run(module)
        _cse_block(module.body, {})
        DCE().run_block(module.body)
        from .insert_syncs import drop_vacuous_publications

        drop_vacuous_publications(module.body.ops[0].body)
        return module
