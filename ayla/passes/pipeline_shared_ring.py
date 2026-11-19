"""Carry one operand ring through a common persistent compute/output loop.

MMA completions release operand banks. A separate final commit signals each
accumulator. A split CTA gather releases a TMEM slot after its final read;
TMA store waits protect shared output staging independently. All warps execute
all loops and writeback operations. There are no actor-specific role loops.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.ir import Block, Op, Value, clone_op, find_ops_of_type, dependency_slice
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    CollectiveAxis,
    DeallocOp,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexLtOp,
    IndexMulOp,
    IndexType,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    RegMemTensorType,
    SmemTensorType,
    TmemTensorType,
    execution_issuer_warp,
    is_side_effect_free,
    root_buffers,
    effective_operands,
)
from ..pipelines.async_resources import allocate_event, SplitBarrier, tma_writeback_fragments
from ..pipelines.ring_schedule import RingInput, RingSchedule, reject_unaccepted_effects
from .pack_tmem_allocs import PackTmemAllocs
from .unroll_loop import UnrollLoop
from ..pipelines.canonicalize import Canonicalize


@dataclass
class _Job:
    guaranteed: bool
    valid: Value
    invalid_next: Value
    definitions: set[Op]
    loads: list[AsyncOp]
    computes: list[AsyncOp]
    fragments: list[list[Op]]
    emitted: set[Op]
    bank_shift: int


class PipelineSharedRing(Pass):
    """A two-frame pipeline with a continuous operand ring.

    One schedule governs both fixed-bank emission and compact dynamic-bank
    emission. Static emission exposes K stages in a rolled job supercycle;
    dynamic emission retains a verified region until after sync insertion.
    """

    def __init__(self, *, operand_stages: int, overlap: bool = True,
                 rolled_operand_pipeline=False, job_regions=False, mma_queue_depth=1,
                 output_writeback_window=None, frames=2):
        if type(operand_stages) is not int or operand_stages < 2:
            raise ValueError("shared ring requires at least two operand banks")
        if rolled_operand_pipeline and (type(mma_queue_depth) is not int or not 1 <= mma_queue_depth < operand_stages):
            raise ValueError("rolled MMA queue depth must fit below the operand bank count")
        if job_regions and not rolled_operand_pipeline:
            raise ValueError("job regions require a rolled shared ring")
        if type(frames) is not int or frames not in (1, 2) or frames == 1 and (not rolled_operand_pipeline or overlap or job_regions):
            raise ValueError("one frame requires a serial rolled ring without job regions")
        super().__init__(operand_stages=operand_stages, overlap=overlap,
                         rolled_operand_pipeline=rolled_operand_pipeline,
                         job_regions=job_regions, mma_queue_depth=mma_queue_depth,
                         output_writeback_window=output_writeback_window, frames=frames)

    def run(self, module: ModuleOp) -> ModuleOp:
        if not self.rolled_operand_pipeline:
            module = Canonicalize().run(
                UnrollLoop(None, label_filter=r"k(?:_.*)?").run(module)
            )
        context = RingInput.from_module(module, allow_single_job=self.rolled_operand_pipeline)
        if self.rolled_operand_pipeline:
            from .pipeline_rolled_shared_ring import prepare_rolled_ring

            return prepare_rolled_ring(module, context, operand_stages=self.operand_stages,
                                       overlap=self.overlap, job_regions=self.job_regions,
                                       mma_queue_depth=self.mma_queue_depth,
                                       output_writeback_window=self.output_writeback_window, frames=self.frames)
        return self._run_static(module, context)

    def _run_static(self, module, context):
        kernel, loop, source = context.kernel, context.loop, context.allocation
        cta_group, grid, tile_count = context.group, context.grid, context.tile_count
        full_jobs = tile_count // grid
        body = list(loop.body.ops)
        # Only the ordinary output-job guard may be flattened. The pass
        # rebuilds that exact job validity around each effect. K/control or
        # participant regions are not speculated.
        guards = [op for op in body if isinstance(op, IfOp)]
        if guards:
            if len(guards) != 1 or any(
                op.blocks and not isinstance(op, AsyncOp) for op in guards[0].body.ops
            ):
                raise ValueError("shared ring requires straight-line guarded jobs")
            guard = guards[0]
            body[body.index(guard) : body.index(guard) + 1] = list(guard.body.ops)
        if any(op.blocks and not isinstance(op, AsyncOp) for op in body):
            raise ValueError("shared ring requires an exposed straight-line K schedule")
        issues = [op for op in body if isinstance(op, AsyncOp)]
        if any(
            execution_issuer_warp(op) != 0
            or "participant_warps" in op.attrs
            or op.attrs.get("release_prior")
            or len(op.completion_locks) != 1
            for op in issues
        ):
            raise ValueError("shared ring requires the common warp-zero issue schedule")
        computes = [op for op in issues if op.lock.type.kind == "tcgen05"]
        loads = [op for op in issues if op.lock.type.kind == "mbar"]
        steps, banks = len(computes), self.operand_stages
        if not steps or banks > steps or len(loads) != steps:
            raise ValueError("shared ring requires one batch per bank in the first job")
        if any(
            not op.body.ops
            or any(not isinstance(inner, MatmulOp) for inner in op.body.ops)
            for op in computes
        ):
            raise ValueError("shared ring requires MMA batches")
        if any(
            not op.body.ops
            or any(
                not isinstance(inner, MoveOp)
                or not isinstance(inner.src.type, GmemTensorType)
                or not isinstance(inner.dst.type, SmemTensorType)
                or not inner.attrs.get("tma")
                for inner in op.body.ops
            )
            for op in loads
        ):
            raise ValueError("shared ring requires fused operand TMA batches")
        # Associate each MMA with its actually preceding operand issue. A
        # future refill cannot pass an MMA using that bank in a valid input.
        available = {}
        ordered_loads = []
        for op in body:
            if op in loads:
                key = frozenset().union(
                    *(root_buffers(move.dst) for move in op.body.ops)
                )
                available[key] = op
            elif op in computes:
                key = frozenset().union(
                    *(root_buffers(mm.lhs) | root_buffers(mm.rhs) for mm in op.body.ops)
                )
                if key not in available:
                    raise ValueError(
                        "shared ring cannot associate an MMA with its operand bank"
                    )
                ordered_loads.append(available.pop(key))
        if len(set(ordered_loads)) != steps:
            raise ValueError(
                "shared ring requires one distinct operand issue per K batch"
            )
        bank_keys = [
            frozenset().union(*(root_buffers(move.dst) for move in load.body.ops))
            for load in ordered_loads
        ]
        if len(set(bank_keys[:banks])) != banks or any(
            key != bank_keys[i % banks] for i, key in enumerate(bank_keys)
        ):
            raise ValueError(
                "shared ring operand storage must cycle through distinct banks"
            )
        # Preserve the operand role when rotating storage. The source load's
        # move order identifies LHS/RHS allocations, including paired RHS
        # partitions. Their views are cloned with these physical roots mapped.
        storage_banks = [
            [next(iter(root_buffers(move.dst))) for move in load.body.ops]
            for load in ordered_loads[:banks]
        ]
        if any(len(root_buffers(move.dst)) != 1 for load in ordered_loads[:banks]
               for move in load.body.ops):
            raise ValueError("shared ring requires unambiguous operand allocations")
        if any(len(roots) != len(storage_banks[0]) for roots in storage_banks):
            raise ValueError("shared ring banks must have matching operand roles")
        fragments = tma_writeback_fragments(body, TmemTensorType)
        schedule = RingSchedule(banks, steps, len(fragments), self.overlap,
                                output_writeback_window=self.output_writeback_window)
        period = schedule.period
        reads = [fragment[0] for fragment in fragments]
        accepted = {
            source,
            *issues,
            *reads,
            *(op for group in fragments for op in group),
        }
        reject_unaccepted_effects(body, accepted, "shared ring encountered an unsupported side effect")
        locks = [
            a
            for a in find_ops_of_type(kernel, AllocOp)
            if isinstance(a.result().type, LockType)
        ]
        # A carried register fragment needs one identity at every job site.
        # Some layouts leave these allocations inside the source job loop;
        # cloning them per plan would make the backedge conversion read a new
        # uninitialized fragment. Hoist the original register storage first.
        carried_registers = [
            op
            for op in body
            if isinstance(op, AllocOp)
            and isinstance(op.result().type, RegMemTensorType)
        ]
        # Existing SMEM banks and register fragment allocations are preserved.
        # Only per-job completion contexts and TMEM lifetimes are replaced.
        definitions = [
            op
            for op in body
            if is_side_effect_free(op)
            and op is not source
            and op not in carried_registers
            and not (isinstance(op, AllocOp) and isinstance(op.result().type, LockType))
        ]
        # Earlier LICM can hoist constant SMEM views out of the output loop.
        # Rotating only allocation roots leaves those views pointing at the
        # old bank. Clone their pure dependency chains into each job as well.
        original_definitions = set(definitions)
        rotated_roots = {source.result(), *(root for bank in storage_banks for root in bank)}
        expanded_definitions, seen_definitions = [], set()

        for op in [*definitions, *ordered_loads, *computes,
                   *(op for fragment in fragments for op in fragment)]:
            for producer in dependency_slice(
                iter(effective_operands(op)),
                lambda value: value not in rotated_roots
                and value.definition not in seen_definitions
                and (value.definition in original_definitions
                     or (is_side_effect_free(value.definition)
                         and not isinstance(value.definition, AllocOp)
                         and bool(root_buffers(value) & rotated_roots))),
                effective_operands,
            ):
                seen_definitions.add(producer)
                expanded_definitions.append(producer)
            if op in original_definitions and op not in seen_definitions:
                seen_definitions.add(op)
                expanded_definitions.append(op)
        definitions = expanded_definitions
        slots = [
            AllocOp.build_with_fresh_id(
                source.result().type, attrs={"label": f"shared_ring_tmem_{slot}"}
            )
            for slot in range(2)
        ]

        group_attrs = {"cta_group": cta_group} if cta_group == 2 else None
        ready_banks = [allocate_event("mbar", f"operand_ready_{bank}", group_attrs) for bank in range(banks)]
        release_banks = [
            allocate_event("tcgen05", f"bank_release_{bank}", group_attrs) for bank in range(banks)
        ]
        frame_ready = [
            allocate_event("tcgen05", f"accumulator_ready_{slot}", group_attrs) for slot in range(2)
        ]
        frame_free = [SplitBarrier.build(cta_group, kernel.num_threads // 32,
                                        f"last_tmem_read_{slot}") for slot in range(2)]
        # Each staging bank has its own source lifetime. Sharing a single
        # completion identity makes writes to an idle bank wait for the most
        # recent store from a different bank, defeating multiple buffers.
        staging_banks = []
        for group in fragments:
            roots = frozenset(root_buffers(group[2].dst))
            if roots not in staging_banks:
                staging_banks.append(roots)
        output_releases = [
            allocate_event(
                "bulk",
                "output_source_release"
                if len(staging_banks) == 1
                else f"output_source_release_{index}",
            )
            for index in range(len(staging_banks))
        ]

        def output_release(group):
            bank = staging_banks.index(frozenset(root_buffers(group[2].dst)))
            return output_releases[bank]

        constants = [
            IndexConstantOp.build(v) for v in (grid, tile_count, tile_count - 1)
        ]
        grid_value, limit, last_tile = [op.result() for op in constants]
        bx = BlockIdOp.build(CollectiveAxis.X)

        def plan(iv, slot, *, guaranteed=False, bank_shift=0):
            vmap = {loop.iv: iv, source.result(): slots[slot].result()}
            for bank, roots in enumerate(storage_banks):
                destination = storage_banks[(bank + bank_shift) % banks]
                vmap.update(zip(roots, destination, strict=True))
            copied_defs = [clone_op(op, vmap) for op in definitions]
            copied_loads = [clone_op(op, vmap) for op in ordered_loads]
            copied_mmas = [clone_op(op, vmap) for op in computes]
            copied_frags = [[clone_op(op, vmap) for op in group] for group in fragments]
            for i, op in enumerate(copied_loads):
                bank = (i + bank_shift) % banks
                op.operands = [ready_banks[bank].result()]
                op.attrs.update(label=f"operand_issue_{i}", ring_bank=bank)
            for i, op in enumerate(copied_mmas):
                bank = (i + bank_shift) % banks
                op.operands = [release_banks[bank].result()]
                if i == steps - 1:
                    op.operands.append(frame_ready[slot].result())
                op.attrs.update(label=f"mma_issue_{i}", ring_bank=bank)
            for group in copied_frags:
                group[-1].operands = [output_release(group).result()]
            linear = IndexMulOp.build(iv, grid_value)
            tile = IndexAddOp.build(bx.result(), linear.result())
            valid = IndexLtOp.build(tile.result(), limit)
            next_tile = IndexAddOp.build(tile.result(), grid_value)
            invalid_next = IndexLtOp.build(last_tile, next_tile.result())
            copied_defs += [linear, tile, valid, next_tile, invalid_next]
            return _Job(
                guaranteed,
                valid.result(),
                invalid_next.result(),
                set(copied_defs),
                copied_loads,
                copied_mmas,
                copied_frags,
                set(),
                bank_shift,
            )

        def define(job, op, destination):
            for operand in effective_operands(op):
                producer = operand.definition
                if producer in job.definitions and producer not in job.emitted:
                    define(job, producer, destination)
                    destination.append(producer)
                    job.emitted.add(producer)

        def guarded(job, ops, destination):
            for op in ops:
                define(job, op, destination)
            if job.guaranteed:
                destination.extend(ops)
                return
            define(job, job.valid.definition, destination)
            if job.valid.definition not in job.emitted:
                destination.append(job.valid.definition)
                job.emitted.add(job.valid.definition)
            destination.append(IfOp.build(job.valid, Block(ops=ops)))

        def engine_wait(event, label):
            # Operand banks are touched only by TMA/MMA engines, issued by
            # warp zero. Other warps never consume their data or phases.
            # Keeping these observations local avoids a cross-warp gather
            # solely to permit event reuse. All warps still execute the same
            # job loop, accumulator waits, TMEM reads and staging operations.
            return AwaitOp.build(event.result(), {"label": label, "waiter_warps": [0]})

        def output(job, slot, start, end, destination):
            if start == end:
                return
            ops = []
            for index in range(start, end):
                ops.extend(schedule.writeback_fragment(
                    job.fragments[index], index, frame_ready[slot].result(),
                    frame_free[slot], await_ready=self.overlap,
                    wait_attrs={"label": "output_source_release"},
                ))
            guarded(job, ops, destination)

        def finish_output(job, destination):
            guarded(job, schedule.writeback_fragment(
                job.fragments[-1], len(fragments) - 1, None, None, finish=True,
                wait_attrs={"label": "output_source_release"},
            ), destination)

        def compute(
            job,
            successor,
            slot,
            previous,
            previous_slot,
            first_job,
            first_slot_use,
            destination,
            pending_output=None,
        ):
            if not first_slot_use:
                # Slot reuse occurs only after job j-2 has released its final
                # TMEM read. It is separate from old output store completion.
                # The first use of slot one has no predecessor.
                guarded(
                    job,
                    [
                        frame_free[slot].barrier('wait', 'tmem_slot_free')
                    ],
                    destination,
                )
            for i, mma in enumerate(job.computes):
                for action in schedule.stage_order:
                    if action == "issue":
                        guarded(job, [engine_wait(ready_banks[(i + job.bank_shift) % banks],
                                                  "operand_ready"), mma], destination)
                    elif action == "finish":
                        if i == 0 and pending_output is not None:
                            finish_output(pending_output, destination)
                    elif action == "refill":
                        if i or not first_job:
                            bank = (i - 1 + job.bank_shift) % banks
                            guarded(job, [engine_wait(release_banks[bank], "bank_release")],
                                    destination)
                            target = i + schedule.prefetch_lead
                            owner = job if target < steps else successor
                            guarded(owner, [owner.loads[target % steps]], destination)
                    elif action == "writeback" and previous is not None:
                        indices = schedule.fragments_at(i)
                        output(previous, previous_slot, indices.start, indices.stop,
                               destination)
            # Only a terminal job has no successor that will observe the last
            # bank's phase. Drain exactly that unobserved event at the tail.
            if not successor.guaranteed:
                tail_wait = engine_wait(release_banks[(steps - 1 + job.bank_shift) % banks], "bank_terminal_drain")
                define(job, job.invalid_next.definition, destination)
                if job.invalid_next.definition not in job.emitted:
                    destination.append(job.invalid_next.definition)
                    job.emitted.add(job.invalid_next.definition)
                terminal = IfOp.build(job.invalid_next, Block(ops=[tail_wait]))
                guarded(job, [terminal], destination)
            if not self.overlap:
                guarded(
                    job,
                    [
                        AwaitOp.build(
                            frame_ready[slot].result(), {"label": "accumulator_ready"}
                        )
                    ],
                    destination,
                )
                output(job, slot, 0, len(fragments), destination)

        zero = IndexConstantOp.build(0)
        one = IndexConstantOp.build(1)
        job0, job1 = (
            plan(zero.result(), 0, guaranteed=True),
            plan(one.result(), 1, guaranteed=True, bank_shift=steps % banks),
        )
        prologue = [zero, one]
        for load in job0.loads[:banks]:
            guarded(job0, [load], prologue)
        compute(job0, job1, 0, None, None, True, True, prologue)
        # Slot one's first use has no previous read-release. Peel it into the
        # shared prologue; subsequent rolled pairs reuse both slots.
        two = IndexConstantOp.build(2)
        job2 = plan(two.result(), 0, guaranteed=full_jobs > 2, bank_shift=2 * steps % banks)
        prologue.append(two)
        compute(job1, job2, 1, job0 if self.overlap else None, 0, False, True, prologue)
        iv = Value(name="job", type=IndexType())
        c_one = IndexConstantOp.build(1)
        next_iv = IndexAddOp.build(iv, c_one.result())
        after_iv = IndexAddOp.build(next_iv.result(), c_one.result())
        c_minus = IndexConstantOp.build(-1)
        prev_iv = IndexAddOp.build(iv, c_minus.result())
        older_iv = IndexAddOp.build(prev_iv.result(), c_minus.result())
        current0, current1 = (
            plan(iv, 0, guaranteed=True),
            plan(next_iv.result(), 1, guaranteed=True, bank_shift=steps % banks),
        )
        prior1 = plan(prev_iv.result(), 1, guaranteed=True, bank_shift=steps % banks)
        prior0 = plan(older_iv.result(), 0, guaranteed=True)
        following0 = plan(after_iv.result(), 0, guaranteed=True)
        main_ops = [c_one, next_iv, after_iv, c_minus, prev_iv, older_iv]
        compute(
            current0,
            current1,
            0,
            prior1 if self.overlap else None,
            1,
            False,
            False,
            main_ops,
            prior0,
        )
        compute(
            current1,
            following0,
            1,
            current0 if self.overlap else None,
            0,
            False,
            False,
            main_ops,
            prior1,
        )
        if period != 2:
            # The base advances by a complete bank/slot supercycle. Each
            # exposed job therefore has a fixed physical bank mapping even
            # though its output coordinates remain loop-carried.
            main_ops = []

            def indexed(offset):
                if offset == 0:
                    return iv
                constant = IndexConstantOp.build(offset)
                index = IndexAddOp.build(iv, constant.result())
                main_ops.extend([constant, index])
                return index.result()

            def main_plan(offset):
                return plan(
                    indexed(offset), (2 + offset) % 2, guaranteed=True,
                    bank_shift=(2 + offset) * steps % banks,
                )

            for offset in range(period):
                current = main_plan(offset)
                successor = main_plan(offset + 1)
                previous = main_plan(offset - 1)
                pending = main_plan(offset - 2)
                compute(
                    current, successor, (2 + offset) % 2,
                    previous if self.overlap else None, (1 + offset) % 2,
                    False, False, main_ops, pending,
                )
        # Peel the final pair too: every successor prefetched by the rolled
        # loop is then guaranteed valid. Path-insensitive sync analysis never
        # joins a skipped refill into the steady ring's event generations.
        epilogue_start = (
            max(2, (full_jobs // 2) * 2 - 2)
            if period == 2
            else 2 + max(0, (full_jobs - 4) // period) * period
        )
        main = ForOp.build(
            2,
            epilogue_start,
            period,
            Block(arguments=[iv], ops=main_ops),
            {"label": "shared_operand_output_ring"},
        )
        epilogue = []
        for j in range(epilogue_start, full_jobs + 1):
            index = IndexConstantOp.build(j)
            next_index = IndexConstantOp.build(j + 1)
            prev_index = IndexConstantOp.build(j - 1)
            older_index = IndexConstantOp.build(j - 2)
            epilogue += [index, next_index, prev_index, older_index]
            current = plan(index.result(), j % 2, guaranteed=j < full_jobs,
                           bank_shift=j * steps % banks)
            successor = plan(
                next_index.result(), (j + 1) % 2, guaranteed=j + 1 < full_jobs,
                bank_shift=(j + 1) * steps % banks,
            )
            previous = plan(prev_index.result(), (j - 1) % 2, guaranteed=True,
                            bank_shift=(j - 1) * steps % banks)
            pending_output = plan(older_index.result(), j % 2, guaranteed=True,
                                  bank_shift=(j - 2) * steps % banks)
            compute(
                current,
                successor,
                j % 2,
                previous if self.overlap else None,
                (j - 1) % 2,
                False,
                False,
                epilogue,
                pending_output,
            )
        if self.overlap:
            # The optional final job has no successor whose compute loop can
            # consume its output. Its valid guard drains exactly that frame.
            finish_output(previous, epilogue)
            output(current, full_jobs % 2, 0, len(fragments), epilogue)
            finish_output(current, epilogue)
        outside = [
            op
            for op in kernel.body.ops
            if op is not loop
            and op not in locks
            and not (isinstance(op, DeallocOp) and isinstance(op.value.type, LockType))
        ]
        kernel.body.ops = [
            *outside,
            *carried_registers,
            *slots,
            *ready_banks,
            *release_banks,
            *frame_ready,
            *(release.allocation for release in frame_free),
            *output_releases,
            *constants,
            bx,
            *prologue,
            main,
            *epilogue,
            *[
                AwaitOp.build(release.result(), {"label": "output_terminal_drain"})
                for release in output_releases
            ],
            BarrierOp.build("block"),
            *[DeallocOp.build(slot.result()) for slot in slots],
        ]
        kernel.attrs.pop("block_gather_backend", None)
        kernel.attrs.update(
            shared_operand_output_ring=True,
            persistent_output_slots=2,
            ordered_mma_completion=True,
        )
        kernel.attrs["name"] = (
            kernel.name + "_shared_ring" + ("_overlap" if self.overlap else "_serial")
        )
        return Canonicalize().run(
            PackTmemAllocs(
                track_column_regions=True, preserve_release_barriers=False
            ).run(module)
        )
