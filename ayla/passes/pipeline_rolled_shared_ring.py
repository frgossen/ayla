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
    root_buffers,
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
                        mma_queue_depth, output_writeback_window=None, frames=2):
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
    if frames != 2:
        if frames != 1 or overlap or job_regions:
            raise ValueError("single-frame ring must be serial without job regions")
        region.attrs["frames"] = frames
    if output_writeback_window is not None:
        region.attrs['output_writeback_window'] = output_writeback_window
    if job_regions:
        if (
            group not in (1, 2)
            or kernel.block_dim != (128, 1, 1)
            or not overlap
            or not 1 <= mma_queue_depth < operand_stages
        ):
            raise ValueError(
                "job regions require a common ring with four-warp fragments"
            )
        kernel.attrs["block_dim"] = (256, 1, 1)
    if not find_ops_of_type(region, OperandPipelineOp):
        # A single peeled K batch has no residual reduction loop. Retain
        # its source as a zero-iteration compact template, so the common
        # persistent emitter can still number banks across output jobs.
        body = region.body.ops[0].body
        guards = [x for x in body.ops if isinstance(x, IfOp)]
        if guards:
            if len(guards) != 1:
                raise ValueError("single-stage ring requires one output guard")
            body = guards[0].body
        issues = [x for x in body.ops if isinstance(x, AsyncOp)
                  and x.lock.type.kind in ("mbar", "tcgen05")]
        if len(issues) != 2:
            raise ValueError("single-stage ring requires one load and MMA batch")
        load, mma = issues
        waits = [next(x for x in body.ops if isinstance(x, AwaitOp)
                      and x.lock is issue.lock) for issue in issues]
        values = {}
        template = [clone_op(x, values) for x in (load, waits[0], mma, waits[1])]
        template[2].body.ops[0].attrs["accumulate"] = True
        iv = Value(name=None, type=IndexType())
        dummy = ForOp.build(1, 2, 1, Block(arguments=[iv], ops=template),
                            {"label": "k"})
        position = body.ops.index(waits[1]) + 1
        body.ops.insert(position, dummy)
        from .pipeline_rolled_k import PipelineRolledK
        [compact], _ = PipelineRolledK(operand_stages).lower_for(dummy)
        compact.attrs["ub"] = 1
        body.ops[position:position + 1] = [compact]
        # This region is not attached to the kernel until preparation ends;
        # normal root-use traversal cannot yet see its cloned operations.
        for view in compact.operands:
            root = view.definition.arg
            for operation in iter_ops(region):
                if operation is view.definition:
                    continue
                for index, value in enumerate(list(operation.operands)):
                    if value is root:
                        operation.operands[index] = view
    _source(region)
    if allocation.parent_block is not kernel.body:
        allocation.parent_block.ops.remove(allocation)
        kernel.body.ops.insert(kernel.body.ops.index(loop), allocation)
    for dealloc in list(find_ops_of_type(kernel, DeallocOp)):
        if dealloc.value is arena:
            dealloc.parent_block.ops.remove(dealloc)
    # Preserve the logical tile shape while reserving both accumulators.
    # The late scheduler substitutes these views, never reallocates TMEM.
    arena.type = clone_type(single)
    arena.type.sizes = [single.sizes[0], single.sizes[1] * frames]
    arena.type.attrs["columns"] = single.columns * frames
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
        persistent_output_slots=frames,
        shared_operand_output_ring=True,
        ordered_mma_completion=True,
    )
    kernel.attrs["name"] += "_rolled_shared_ring" + (
        "_overlap" if overlap else "_serial"
    )
    if frames == 1:
        kernel.attrs["name"] += "_frame1"
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
        frames = op.attrs.get("frames", 2)
        schedule = RingSchedule(depth, steps, len(fragments), overlap, job_regions,
                                op.attrs.get('output_writeback_window'),
                                mma_queue_depth=op.attrs.get('mma_queue_depth', 1))
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
            for s in range(frames)
        ]
        frame_free = [SplitBarrier.build(group, warps, f"last_tmem_read_{s}")
                      for s in range(frames)]
        ops.extend(release.allocation for release in frame_free)
        slots = []
        for s in range(frames):
            view = TileOp.build(arena, [0, s], first_slot.type)
            ops.append(view)
            slots.append(view.result())
        # Bulk waits observe the issuing thread's FIFO and protect staging
        # reuse; the optional queue retains only newer disjoint-buffer reads.
        store_locks = list(dict.fromkeys(fragment[-1].lock for fragment in fragments))
        store_limit = find_ancestor_of_type(op, KernelOp).attrs.get(
            "rolled_output_store_pending_groups", 0
        )
        if type(store_limit) is not int or not 0 <= store_limit <= 7:
            raise ValueError("rolled output store queue must be an integer from zero to seven")
        # Stores occur in fragment order for every output job. The last
        # fragment may be carried into the next job, but is issued before
        # that job's first fragment. Each rotating participant has its own
        # staging and issuer FIFO and repeats this same fragment sequence.
        # Wait for the most recent possibly aliasing staging write, while
        # allowing newer disjoint-buffer groups to remain pending. Distinct
        # views of a common allocation stay conservatively aliasing.
        staging_roots = [root_buffers(fragment[2].dst) for fragment in fragments]
        store_pending = []
        for index, roots in enumerate(staging_roots):
            previous = max(
                position for position in range(index - len(fragments), index)
                if roots & staging_roots[position % len(fragments)]
            )
            store_pending.append(min(store_limit, index - previous - 1))
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
                wait_attrs={"pending": store_pending[index]} if store_limit else None,
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
            reuse = binary(IndexLtOp, one if frames == 2 else zero, job, body)
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
                    if steps == 1:
                        initial.operands.append(frame_ready[slot])
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
                    # Refill P-Q ahead after the matching ordinal i-Q bank
                    # release. Q newer MMAs may remain queued; ownership
                    # and phases continue across the output-job backedge.
                    prior = binary(IndexAddOp, ordinal, const(-schedule.mma_queue_depth, kbody), kbody)
                    refill = [
                        AwaitOp.build(
                            bank_event(free, prior, kbody)[0],
                            {"waiter_warps": [0], "label": "bank_release"},
                        )
                    ]
                    lead = schedule.prefetch_lead
                    future_k = binary(IndexAddOp, iv, const(lead, refill), refill)
                    future = binary(IndexAddOp, ordinal, const(lead, refill), refill)
                    # The target batch belongs to job + d, where d = future_k // steps.
                    # A ring deeper than one job's K batches reaches past the successor.
                    offsets = range((steps - 1 + lead) // steps + 1)
                    conditions = []
                    for d in offsets:
                        below = (
                            binary(IndexLtOp, future_k,
                                   stage_count if d == 0 else const((d + 1) * steps, refill),
                                   refill)
                            if (d + 1) * steps <= steps - 1 + lead
                            else None
                        )
                        above = (
                            binary(IndexLtOp, const(d * steps - 1, refill), future_k, refill)
                            if d
                            else None
                        )
                        conditions.append([c for c in (above, below) if c is not None])
                    for d, guards in zip(offsets, conditions):
                        target = []
                        if not d:
                            issue_load(job, future_k, future, target)
                        else:
                            target_job = binary(
                                IndexAddOp, job, one if d == 1 else const(d, target), target
                            )
                            target_k = binary(IndexAddOp, future_k, const(-d * steps, target), target)
                            target_valid = valid(target_job, target)
                            load_ops = []
                            issue_load(target_job, target_k, future, load_ops)
                            target.append(IfOp.build(target_valid, Block(ops=load_ops)))
                        for guard in reversed(guards):
                            target = [IfOp.build(guard, Block(ops=target))]
                        refill.extend(target)
                    has_prior = binary(IndexLtOp, const(schedule.mma_queue_depth - 1, kbody), ordinal, kbody)
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
        # The initial fill covers ordinals 0..depth-1: job d owns K batches
        # [d*steps, (d+1)*steps). Jobs past the first are guarded like refills.
        for d in range(math.ceil(depth / steps)):
            fill_iv = Value(name=None, type=IndexType())
            fill = []
            if d:
                fill_job = const(d, fill)
                fill_ordinal = binary(IndexAddOp, fill_iv, const(d * steps, fill), fill)
                load_ops = []
                issue_load(fill_job, fill_iv, fill_ordinal, load_ops)
                fill.append(IfOp.build(valid(fill_job, fill), Block(ops=load_ops)))
            else:
                issue_load(zero, fill_iv, fill_iv, fill)
            ops.append(
                ForOp.build(
                    0,
                    min(steps, depth - d * steps),
                    1,
                    Block(arguments=[fill_iv], ops=fill),
                    {"label": "shared_ring_fill", "unroll": False},
                )
            )
        job = Value(name=None, type=IndexType())
        jobs = []
        compute(job, 0, jobs)
        next_job = binary(IndexAddOp, job, one, jobs)
        compute(next_job, 1 if frames == 2 else 0, jobs)
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
            if jobs_count == 0:
                continue
            tail = []
            last = const(jobs_count - 1, tail)
            if overlap:
                previous = const(jobs_count - 2, tail)
                if not job_regions and jobs_count >= 2:
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
        # Bulk groups belong to issuing threads, not to software locks. One
        # unconditional full wait drains every thread's queue, including both
        # output issuers after job-participant scheduling.
        if store_locks:
            ops.append(AwaitOp.build(store_locks[0], {"label": "output_terminal_drain"}))
        # Store source reads and TMEM reads need retirement, not publication.
        # Destination writes complete independently in the bulk engine.
        ops.append(BarrierOp.build("cluster" if group == 2 else "block",
            attrs={"publishes": False} if group == 2 else None))
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
        unroll = module.body.ops[0].attrs.get("rolled_k_unroll", 1)
        if unroll > 1:
            from .unroll_loop import UnrollLoop

            module = UnrollLoop(unroll, "shared_ring_rolled_k").run(module)
            # Fully unrolling leaves a one-trip loop. Fold it before peeling,
            # which otherwise requires at least two remaining iterations.
            module = Canonicalize().run(module)
        # A one-batch reduction has the same first/last iteration.
        # Fold its bounded loop once before the two-edge peeling pass.
        if any(loop.trip_count == 1 and
               loop.attrs.get("label") == "shared_ring_rolled_k"
               for loop in find_ops_of_type(module, ForOp)):
            module = Canonicalize().run(module)
        module = PeelLoop(
            "shared_ring_rolled_k", front_iterations=1, back_iterations=1
        ).run(module)
        # Clone only the existing bounded output-pair loop. The induction
        # variable retains its absolute job ordinal; bank phases, output
        # ownership, event waits and issue order remain in the cloned body.
        kernel = module.body.ops[0]
        if kernel.attrs.get("peel_first_output_pair"):
            module = PeelLoop(
                "shared_operand_output_ring", front_iterations=1
            ).run(module)
        job_unroll = kernel.attrs.get("rolled_job_unroll", 1)
        if job_unroll > 1:
            from .unroll_loop import UnrollLoop

            module = UnrollLoop(job_unroll, "shared_operand_output_ring").run(module)
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
