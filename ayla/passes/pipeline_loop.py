"""Software-pipeline labeled ops in a loop body.

For each ForOp whose label matches ``label_filter_loop``, ops whose
``attrs["label"]`` matches ``label_filter_op`` are
issued ``pipelining_offset`` iterations earlier than they appear in the
source loop. The transformation produces:

  - **prologue**: the labeled ops for the first ``pipelining_offset``
    iterations, with constant iv values.
  - **main loop** (trip-count reduced by ``pipelining_offset``): for
    each iteration ``k`` the unlabeled ops are emitted at ``iv = k``
    followed by the labeled ops at ``iv = k + pipelining_offset *
    step``. Emitting unlabeled first means for n=1 with the natural
    [labeled, unlabeled] source order the flat schedule is identical
    to the original loop (no reorder).
  - **epilogue**: the unlabeled ops for the last ``pipelining_offset``
    iterations.

With ``guarded_tail=True`` and offset one, the final drain stays in the
main loop and future work is guarded by the original loop bound. This
retains one consumer body, including when output iterations are partial.

Legality: see ``_check_legality``.

When the selected operations issue work to a partially awaitable hardware
queue, the loop already represents an unrolled software pipeline.  In that
case the pass pipelines completion instead: each stage partially awaits the
preceding stage and a full await after the loop drains the queue.
"""
from __future__ import annotations

from typing import cast
from dataclasses import dataclass

from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    CollectiveAxis,
    DeallocOp,
    ElementwiseOp,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    IndexType,
    KernelOp,
    LockType,
    LockViewOp,
    MatmulOp,
    ModuleOp,
    MoveOp,
    OperandPipelineOp,
    RegMemTensorType,
    SmemTensorType,
    TileOp,
    TmaMoveOp,
    TmemTensorType,
    WarpIdsOp,
    effective_operands,
    execution_issuer_warp,
    has_ordering_effect,
    is_side_effect_free,
    memory_effects,
    memory_effects_may_alias,
    ordered_async_roots,
    physical_layout,
    root_buffers,
)
from ..base.ir import (
    Block,
    Op,
    Value,
    clone_op,
    clone_type,
    dependency_slice,
    find_ancestor_of_type,
    find_ops_of_type,
    iter_ops,
)
from ..base.lowering_pass import LoweringPass, lowers
from ._utils import label_matches
from ..pipelines.ring_schedule import PipelineSchedule, RingInput, RingSchedule, reject_unaccepted_effects
from ..pipelines.async_resources import allocate_event, SplitBarrier, tma_writeback_fragments
from ..pipelines.canonicalize import Canonicalize
from .pack_tmem_allocs import PackTmemAllocs
from .unroll_loop import UnrollLoop
from ._loop_utils import clone_loop_iteration, guard_narrows, loop_guard


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


class PipelineLoop(LoweringPass):
    """Software-pipeline labeled ops in ForOp bodies."""

    label_filter_loop: str
    label_filter_op: str
    pipelining_offset: int

    def __init__(
        self,
        label_filter_loop: str = "",
        label_filter_op: str = "",
        pipelining_offset: int = 0,
        *,
        guarded_tail: bool = False,
        plan: PipelineSchedule | None = None,
        operand_stages: int = 3,
    ) -> None:
        super().__init__(
            label_filter_loop=label_filter_loop,
            label_filter_op=label_filter_op,
            pipelining_offset=pipelining_offset,
            guarded_tail=guarded_tail,
            plan=plan, operand_stages=operand_stages,
        )
        if guarded_tail and pipelining_offset != 1:
            raise ValueError('guarded pipeline tail requires offset one')

    def run(self, module):
        if self.plan is None:
            return super().run(module)
        if self.operand_stages < 2 or max(1, self.plan.mma_pending) >= self.operand_stages:
            raise ValueError("MMA queue depth must leave an operand refill bank")
        if not self.plan.cross_jobs:
            if not self.plan.rolled:
                return super().run(module)
            for loop in list(find_ops_of_type(module, ForOp)):
                if loop.attrs.get("label") == self.label_filter_loop:
                    parent = loop.parent_block
                    self._absorb_first_iteration(loop)
                    ops, _ = self.prepare_operands(loop)
                    index = parent.ops.index(loop)
                    parent.ops[index:index + 1] = ops
            return module
        if not self.plan.rolled:
            module = Canonicalize().run(UnrollLoop(None, r"k(?:_.*)?").run(module))
        context = RingInput.from_module(module, allow_single_job=self.plan.rolled)
        if self.plan.rolled:
            from .pipeline_rolled_shared_ring import prepare_rolled_ring
            return prepare_rolled_ring(module, context, operand_stages=self.operand_stages,
                overlap=self.plan.output_overlap, job_regions=self.plan.rotating,
                mma_queue_depth=max(1, self.plan.mma_pending),
                output_writeback_window=self.plan.writeback_window,
                frames=self.plan.accumulator_slots)
        return self._pipeline_resources_static(module, context)

    @staticmethod
    def _absorb_first_iteration(loop):
        """Put a standalone peeled first batch into the initial ring fill.

        Persistent regions keep their existing job protocol. A standalone
        reduction can start filling all banks before it waits for batch zero.
        """
        from .pipeline_rolled_k import _parts
        from ..base.ir import find_ancestor_of_type
        if loop.lb != 1 or find_ancestor_of_type(loop, ForOp) is not None:
            return
        parent = loop.parent_block
        prefix = [op for op in parent.ops[:parent.ops.index(loop)]
                  if not is_side_effect_free(op)]
        first = prefix[-4:]
        if (len(first) != 4 or not isinstance(first[0], AsyncOp)
                or not isinstance(first[1], AwaitOp)
                or not isinstance(first[2], AsyncOp)
                or not isinstance(first[3], AwaitOp)):
            return
        load, mma = _parts(loop)
        initial_load, load_wait, initial_mma, mma_wait = first
        if (initial_load.lock.type.kind != 'mbar' or initial_mma.lock.type.kind != 'tcgen05'
                or load_wait.lock is not initial_load.lock or mma_wait.lock is not initial_mma.lock
                or len(initial_mma.completion_locks) != 1
                or len(initial_load.completion_locks) != 1
                or execution_issuer_warp(initial_load) != execution_issuer_warp(load)
                or execution_issuer_warp(initial_mma) != execution_issuer_warp(mma)
                or not initial_mma.body.ops
                or not all(isinstance(mm, MatmulOp) for mm in initial_mma.body.ops)
                or initial_mma.body.ops[0].accumulate
                or not all(mm.accumulate for mm in initial_mma.body.ops[1:])):
            return
        def accessed(issue):
            return [root_buffers(value) for inner in issue.body.ops for value in inner.operands]
        if accessed(initial_load) != accessed(load) or accessed(initial_mma) != accessed(mma):
            return
        for op in first:
            parent.ops.remove(op)
        loop.lb = 0
        loop.attrs['initialize_accumulator'] = True

    def prepare_operands(self, op):
        from .pipeline_rolled_k import _parts
        if op.attrs.get("label") != self.label_filter_loop:
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
                arena.sizes = [single.sizes[0], single.sizes[1] * self.operand_stages]
                arena.attrs["strides"] = [arena.sizes[1], 1]
            else:
                arena.sizes = [single.sizes[0] * self.operand_stages, single.sizes[1]]
            root.type = arena
            view = TileOp.build(root, [0, 0], single)
            live = set(root.users())
            for user, index in list(root.uses):
                if user is not view and user in live:
                    user.operands[index] = view.result()
            block = root.definition.parent_block
            block.ops.insert(block.ops.index(root.definition) + 1, view)
            banks.append((root, view.result()))
        region = OperandPipelineOp.build(op, self.operand_stages)
        region.attrs['mma_queue_depth'] = max(1, self.plan.mma_pending)
        region.operands = [view for _, view in banks]
        return [region], []


    def _pipeline_queued_async(
        self, op: ForOp, selected: list[AsyncOp],
    ) -> tuple[list[Op], list[object]]:
        """Pipeline completion of an unrolled hardware async queue.

        Each selected issue keeps its own lock.  Replace its immediate await
        with a partial wait for the preceding issue, wrapping the first issue
        around to the last stage from the prior loop iteration.  A full wait
        after the loop drains the final outstanding issue.
        """
        replacements: dict[Op, Op] = {}
        previous_issues = [selected[-1], *selected[:-1]]
        for previous, issue in zip(previous_issues, selected, strict=True):
            matching = [
                user for user in issue.lock.users()
                if isinstance(user, AwaitOp) and user.parent_block is op.body
            ]
            if len(matching) != 1:
                raise ValueError(
                    "Pipeline: queued async issue must have exactly one "
                    "await in the same loop body"
                )
            [wait] = matching
            replacements[wait] = AwaitOp.build(
                previous.lock, attrs={**wait.attrs, "pending": 1},
            )
        op.body.ops = [
            replacements.get(body_op, body_op) for body_op in op.body.ops
        ]
        return [op, AwaitOp.build(selected[-1].lock)], []

    def _pipeline_event_async(
        self, op: ForOp, selected: list[AsyncOp],
    ) -> tuple[list[Op], list[object]]:
        """Carry the final ``pipelining_offset`` completion events into the next iteration.

        With offset d, up to d MMA batches stay queued across the loop back
        edge: the next iteration observes the last d batches' events before
        reusing their resources. Other waits stay with their buffer reuse.
        ScheduleAsync may issue independent work ahead of them, but must wait
        before reusing a lock. Peel the first iteration: its final events have
        not been issued yet.
        """
        depth = self.pipelining_offset
        if depth > len(selected):
            raise ValueError("Pipeline: queue depth exceeds the batches in one iteration")
        if not all(ordered_async_roots(issue) for issue in selected):
            raise ValueError("Pipeline: ordered event queue requires MMA batches")
        if len({issue.lock for issue in selected}) != len(selected):
            raise ValueError("Pipeline: queued batches require distinct completion events")
        completions: dict[AsyncOp, AwaitOp] = {}
        for issue in selected:
            following = [body_op for body_op in op.body.ops
                         if isinstance(body_op, AwaitOp) and body_op.lock is issue.lock]
            if len(following) != 1 or op.body.ops.index(following[0]) < op.body.ops.index(issue):
                raise ValueError("Pipeline: ordered event issue needs one following await")
            completions[issue] = following[0]
        from .schedule_async import _can_swap  # a wait may pass later same-engine issues

        carried = selected[-depth:]
        waits = [completions[issue] for issue in carried]
        for wait in waits:
            if any(
                trailing not in waits and not self._can_reorder(wait, trailing)
                and not _can_swap(wait, trailing)
                for trailing in op.body.ops[op.body.ops.index(wait) + 1 :]
            ):
                raise ValueError("Pipeline: completion cannot move past a dependent operation")
        op.body.ops = [body_op for body_op in op.body.ops if body_op not in waits]
        iv = IndexConstantOp.build(op.lb)
        prologue = [iv, *clone_loop_iteration(op, iv.result(), "_front0")]
        op.body.ops = [*waits, *op.body.ops]
        op.lb += op.step
        return [
            *prologue,
            *([op] if op.trip_count else []),
            *(AwaitOp.build(issue.lock) for issue in carried),
        ], []

    def _compute_labeled_and_shared(
        self, body_ops: list[Op],
    ) -> tuple[set[Op], set[Op]]:
        """Compute the labeled and shared sets of in-body ops.

        An op is **labeled** if it is not value-identical and its
        ``attrs["label"]`` matches the ``label_filter_op`` regex OR any
        labeled op transitively consumes its result.

        An op is **shared** if it is value-identical (pure) and matches or
        is pulled in as a dependency of a labeled op. Shared ops get cloned
        into both the labeled and unlabeled sections, so an unlabeled op may
        consume a pure op the filter matches.
        """
        in_body_def: dict[Value, Op] = {
            r: body_op for body_op in body_ops for r in body_op.results
        }
        matched = [
            body_op for body_op in body_ops
            if label_matches(body_op, self.label_filter_op)
        ]
        shared: set[Op] = {op for op in matched if is_side_effect_free(op)}
        labeled: set[Op] = set(matched) - shared
        worklist = list(matched)
        while worklist:
            cur = worklist.pop()
            for v in effective_operands(cur):
                defining = in_body_def.get(v)
                if defining is None or defining in labeled or defining in shared:
                    continue
                if is_side_effect_free(defining):
                    shared.add(defining)
                else:
                    labeled.add(defining)
                worklist.append(defining)

        return labeled, shared

    def _check_ssa_closure(
        self, body_ops: list[Op], labeled_set: set[Op], shared_set: set[Op],
    ) -> None:
        """Labeled and unlabeled in-body ops must be SSA-disjoint:
        neither set may consume an SSA result produced by the other.
        Shared ops (cloned into both) are excluded from this check.

        - labeled→unlabeled: a labeled op consuming an unlabeled result
          would be hoisted past its def.
        - unlabeled→labeled: an unlabeled op consuming a labeled result
          would, after pipelining, reference the labeled clone from a
          different iteration. Without loop-carried iter-args we cannot
          express that link, so we reject it.

        Labeled and unlabeled communicate only via side effects on
        buffers defined outside the loop body.  The check walks nested
        blocks so that references hidden inside e.g. an inner ForOp
        are detected.
        """
        in_body_def: dict[Value, Op] = {
            r: body_op for body_op in body_ops for r in body_op.results
        }
        for body_op in body_ops:
            if body_op in shared_set:
                continue
            is_labeled = body_op in labeled_set
            for v in effective_operands(body_op):
                defining = in_body_def.get(v)
                if defining is None or defining in shared_set:
                    continue
                defining_is_labeled = defining in labeled_set
                if is_labeled and not defining_is_labeled:
                    raise ValueError(
                        f"Pipeline: labeled op {body_op.op_id} depends "
                        f"on unlabeled in-body op {defining.op_id}; "
                        f"label it too or refactor the loop"
                    )
                if not is_labeled and defining_is_labeled:
                    raise ValueError(
                        f"Pipeline: unlabeled op {body_op.op_id} "
                        f"depends on labeled in-body op "
                        f"{defining.op_id}; labeled and unlabeled must "
                        f"communicate only via side effects on "
                        f"out-of-loop buffers"
                    )


    def _can_reorder(self, earlier: Op, later: Op) -> bool:
        """Whether `earlier` and `later` may be swapped without
        changing observable semantics.

        Strategy: side-effect-free ops are always safe to move. Other
        ops are checked via ``_memory_effects``: two ops conflict only
        if they touch the same root buffer and at least one is a
        write. Unknown effects (``None``) are treated as worst-case.
        """
        if has_ordering_effect(earlier) or has_ordering_effect(later):
            return False
        if is_side_effect_free(earlier) or is_side_effect_free(later):
            return True
        eff_earlier = memory_effects(earlier)
        eff_later = memory_effects(later)
        if eff_earlier is None or eff_later is None:
            return False
        for effect_a in eff_earlier:
            for effect_b in eff_later:
                if memory_effects_may_alias(effect_a, effect_b) and (
                    effect_a.kind == "W" or effect_b.kind == "W"
                ):
                    return False
        return True

    def _check_legal_for_offset_one(
        self, body_ops: list[Op], labeled_set: set[Op],
        shared_set: set[Op],
    ) -> None:
        """For ``pipelining_offset == 1`` the transformation moves all
        labeled ops to the front of each iteration (preserving their
        relative source order). Equivalently: for every (U, L) pair
        where unlabeled U precedes labeled L in source body order, U
        and L get swapped. Cross-iteration swaps do not occur for
        offset=1.

        Shared ops are excluded — they are cloned into both sections
        and do not cause reordering conflicts.

        We walk each such pair and reject if the swap cannot be shown
        safe by ``_can_reorder``.
        """
        unlabeled_seen: list[Op] = []
        for body_op in body_ops:
            if body_op in labeled_set:
                for earlier in unlabeled_seen:
                    if not self._can_reorder(earlier, body_op):
                        raise ValueError(
                            f"Pipeline: pipelining_offset=1 would "
                            f"reorder unlabeled op {earlier.op_id} and "
                            f"labeled op {body_op.op_id}, but they "
                            f"cannot be proven independent"
                        )
            elif body_op not in shared_set:
                unlabeled_seen.append(body_op)

    def _check_legal_for_offset_gt_one(
        self, body_ops: list[Op], labeled_set: set[Op],
        shared_set: set[Op],
    ) -> None:
        """For ``pipelining_offset > 1`` every labeled op is hoisted by
        at least one full iteration. So in addition to any intra-
        iteration swap, every (U, L) pair gets cross-iteration swaps
        regardless of their source order: L issued for iter k+n
        executes before U for iter k+1..k+n-1.

        Shared ops are excluded — they are cloned into both sections
        and do not cause reordering conflicts.

        Memory effects don't carry iteration info, so a single
        ``_can_reorder(U, L)`` check suffices per pair: if U and L
        touch the same root buffer with a write, they conflict no
        matter which iteration's data they're operating on. This will
        typically force the user to multi-buffer before pipelining
        with offset > 1.
        """
        labeled = [body_op for body_op in body_ops if body_op in labeled_set]
        unlabeled = [
            body_op for body_op in body_ops
            if body_op not in labeled_set and body_op not in shared_set
        ]
        for u in unlabeled:
            for l_op in labeled:
                if not self._can_reorder(u, l_op):
                    raise ValueError(
                        f"Pipeline: pipelining_offset>1 would reorder "
                        f"unlabeled op {u.op_id} and labeled op "
                        f"{l_op.op_id} across iterations, but they "
                        f"cannot be proven independent; consider "
                        f"multi-buffering the shared buffer"
                    )

    def _check_legality(
        self, body_ops: list[Op], labeled_set: set[Op], shared_set: set[Op],
    ) -> None:
        """Raise ValueError if pipelining `labeled_set` ahead by
        ``self.pipelining_offset`` iterations is not legal."""
        assert self.pipelining_offset > 0, (
            "Pipeline: _check_legality only meaningful for "
            "pipelining_offset > 0"
        )
        self._check_ssa_closure(body_ops, labeled_set, shared_set)
        if self.pipelining_offset == 1:
            self._check_legal_for_offset_one(body_ops, labeled_set, shared_set)
        else:
            self._check_legal_for_offset_gt_one(body_ops, labeled_set, shared_set)

    def _loop_matches(self, op: ForOp) -> bool:
        return label_matches(op, self.label_filter_loop)

    @lowers(ForOp)
    def lower_for(
        self, op: ForOp,
    ) -> tuple[list[Op], list[object]] | None:
        if self.pipelining_offset <= 0:
            return None

        # Lower inner loops first (bottom-up) so the body is fully
        # transformed before we pipeline this loop.
        self.lower_block(op.body)

        if not self._loop_matches(op):
            return [op], []

        lb, ub, step = op.lb, op.ub, op.step
        n = self.pipelining_offset
        if op.trip_count < n:
            raise ValueError(
                f"Pipeline: trip count ({op.trip_count}) is smaller than "
                f"pipelining_offset ({n})"
            )

        body_ops = op.body.ops
        selected_async = [
            body_op for body_op in body_ops
            if isinstance(body_op, AsyncOp)
            and label_matches(body_op, self.label_filter_op)
        ]
        if selected_async and all(
            isinstance(issue.lock.type, LockType)
            and issue.lock.type.orders_async_writes
            for issue in selected_async
        ):
            if self.guarded_tail:
                raise ValueError('guarded pipeline tail requires structured work regions')
            kinds = {cast(LockType, issue.lock.type).kind for issue in selected_async}
            if len(kinds) != 1:
                raise ValueError("Pipeline: queued issues must use the same mechanism")
            if cast(LockType, selected_async[0].lock.type).supports_partial_await:
                if self.pipelining_offset != 1:
                    raise ValueError(
                        "Pipeline: partially awaited queues require offset one"
                    )
                return self._pipeline_queued_async(op, selected_async)
            return self._pipeline_event_async(op, selected_async)

        guard = loop_guard(op)
        prefix = body_ops[:-1] if guard is not None else []
        if guard is not None:
            body_ops = guard.body.ops
        labeled_set, shared_set = self._compute_labeled_and_shared(body_ops)
        labeled = [o for o in body_ops if o in labeled_set]
        shared = [o for o in body_ops if o in shared_set]
        unlabeled = [o for o in body_ops if o not in labeled_set and o not in shared_set]
        if not labeled:
            return [op], []

        self._check_legality(body_ops, labeled_set, shared_set)

        def iteration(iv: Value, selected: list[Op]) -> list[Op]:
            vmap: dict[Value, Value] = {op.iv: iv}
            definitions = [clone_op(body_op, vmap) for body_op in prefix]
            cloned = [clone_op(body_op, vmap) for body_op in selected]
            if guard is None:
                return definitions + cloned
            return [*definitions, IfOp.build(vmap[guard.condition], Block(ops=cloned))]

        # Prologue: shared + labeled ops for iv = lb, lb+step, ..., lb+(n-1)*step.
        prologue_ops: list[Op] = []
        for i in range(n):
            iv_const = IndexConstantOp.build(value=lb + i * step)
            prologue_ops.append(iv_const)
            prologue_ops.extend(iteration(iv_const.result(), shared + labeled))

        # Main loop body: shared + unlabeled at iv, then shared + labeled at iv + n*step.
        new_iv = Value(name=op.iv.name, type=IndexType())
        ahead_const = IndexConstantOp.build(value=n * step)
        ahead_iv = IndexAddOp.build(lhs=new_iv, rhs=ahead_const.result())
        current = iteration(new_iv, shared + unlabeled)
        future_ops = iteration(ahead_iv.result(), shared + labeled)
        if self.guarded_tail:
            # Keep the drain in the common loop, suppressing only work past
            # the original bound. This avoids a duplicate output epilogue
            # and retains its original per-iteration guards and dependencies.
            limit = IndexConstantOp.build(ub)
            active = IndexLtOp.build(ahead_iv.result(), limit.result())
            future = IfOp.build(active.result(), Block(ops=future_ops))
            future.attrs['label'] = self.label_filter_op
            new_body_ops = [*current, ahead_const, ahead_iv, limit, active, future]
        elif guard is not None and guard_narrows(op, guard):
            *current_prefix, current_guard = current
            assert isinstance(current_guard, IfOp)
            current_guard.body.ops = [*current_guard.body.ops, future_ops[-1]]
            new_body_ops = [*current_prefix, ahead_const, ahead_iv, *future_ops[:-1], current_guard]
        else:
            new_body_ops = [*current, ahead_const, ahead_iv, *future_ops]
        new_body = Block(arguments=[new_iv], ops=new_body_ops)
        new_ub = ub if self.guarded_tail else ub - n * step
        new_for = ForOp.build(lb=lb, ub=new_ub, step=step, body=new_body,
                              attrs=op.extra_attrs)
        if self.guarded_tail:
            return [*prologue_ops, new_for], []

        # Epilogue: shared + unlabeled ops for iv = new_ub, ..., ub-step.
        epilogue_ops: list[Op] = []
        for i in range(n):
            iv_const = IndexConstantOp.build(value=new_ub + i * step)
            epilogue_ops.append(iv_const)
            epilogue_ops.extend(iteration(iv_const.result(), shared + unlabeled))

        return [*prologue_ops, new_for, *epilogue_ops], []

    def _pipeline_resources_static(self, module, context):
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
        schedule = RingSchedule(banks, steps, len(fragments), self.plan.output_overlap,
                                output_writeback_window=self.plan.writeback_window)
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
                    frame_free[slot], await_ready=self.plan.output_overlap,
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
            if not self.plan.output_overlap:
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
        compute(job1, job2, 1, job0 if self.plan.output_overlap else None, 0, False, True, prologue)
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
            prior1 if self.plan.output_overlap else None,
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
            current0 if self.plan.output_overlap else None,
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
                    previous if self.plan.output_overlap else None, (1 + offset) % 2,
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
                previous if self.plan.output_overlap else None,
                (j - 1) % 2,
                False,
                False,
                epilogue,
                pending_output,
            )
        if self.plan.output_overlap:
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
            kernel.name + "_shared_ring" + ("_overlap" if self.plan.output_overlap else "_serial")
        )
        return Canonicalize().run(
            PackTmemAllocs(
                track_column_regions=True, preserve_release_barriers=False
            ).run(module)
        )

    @staticmethod
    def assign_participants(result, op):
        kernel = find_ancestor_of_type(op, KernelOp)
        assert kernel is not None
        assert kernel.num_threads == 256
        group_warps = kernel.num_threads // 32 // 2
        warp = WarpIdsOp.build()
        coordinate = warp.x
        four = IndexConstantOp.build(group_warps)
        three = IndexConstantOp.build(group_warps - 1)
        active = [
            IndexLtOp.build(coordinate, four.result()),
            IndexLtOp.build(three.result(), coordinate),
        ]
        # The actor of job j-1 releases issue to the actor of job j. The event is
        # this CTA's own: its next phase needs the waiter's own release, so the
        # expected parity cannot alias. Waiting instead for the previous job's
        # MMA completion lets a paired peer, which only loads, fall two phases
        # behind once its loads run more than one job ahead.
        handoff = SplitBarrier.build(1, 1, "job_issue_handoff")
        additions = [warp, four, three, *active, handoff.allocation]
        remaps = {}

        def actor(slot):
            return group_warps * slot

        def participants(slot):
            return list(range(group_warps * slot, group_warps * (slot + 1)))

        def mapped(value):
            if value in remaps:
                return remaps[value]
            definition = value.definition
            assert isinstance(definition, (AllocOp, TileOp)), repr(definition)
            if isinstance(definition, AllocOp):
                clone = AllocOp.build_with_fresh_id(
                    value.type, attrs={"label": "rotating_output_slot1"}
                )
            else:
                values = {definition.arg: mapped(definition.arg)}
                clone = clone_op(definition, values)
            additions.append(clone)
            remaps[value] = clone.result()
            return clone.result()

        def remap_output(operation):
            for i, value in enumerate(list(operation.operands)):
                if isinstance(value.type, SmemTensorType):
                    operation.operands[i] = mapped(value)

        def walk(operation, compute_slot=None, output_slot=None):
            compute_slot = operation.attrs.get("compute_slot", compute_slot)
            if "compute_slot" in operation.attrs:
                job = operation.body.ops[0].rhs
                zero = IndexConstantOp.build(0)
                minus = IndexConstantOp.build(-1)
                two = IndexConstantOp.build(2)
                prev = IndexAddOp.build(job, minus.result())
                phase = IndexModOp.build(prev.result(), two.result())
                # A scalar view gives this consumer its own expected phase (one per
                # job); no extra hardware barrier or implicit parity mutation.
                view = LockViewOp.build(handoff.allocation.result(), zero.result(), phase.result())
                due = IndexLtOp.build(zero.result(), job)
                wait = AwaitOp.build(
                    view.result(),
                    {
                        "waiter_warps": [actor(compute_slot)],
                        "label": "previous_job_issued",
                    },
                )
                operation.body.ops = [
                    zero,
                    minus,
                    two,
                    prev,
                    phase,
                    view,
                    due,
                    IfOp.build(due.result(), Block(ops=[wait])),
                    *operation.body.ops,
                ]
            output_slot = operation.attrs.get("writeback_slot", output_slot)
            if "writeback_slot" in operation.attrs:
                region = IfOp.build(
                    active[output_slot].result(), Block(ops=list(operation.body.ops))
                )
                region.attrs["participant_warps"] = participants(output_slot)
                operation.body.ops = [region]
            if isinstance(operation, AllocOp) and isinstance(
                operation.result().type, LockType
            ):
                label = operation.attrs.get("label", "")
                if label.startswith("last_tmem_read_"):
                    slot = int(label[-1])
                    lock_type = operation.result().type
                    if lock_type.kind == "mbar":
                        # One local arrival from each warp that reads this slot.
                        lock_type.attrs["arrivals"] = len(participants(slot))
                    else:
                        lock_type.attrs["release_warps"] = participants(slot)
            if isinstance(operation, AsyncOp):
                if operation.lock.type.kind == "bulk":
                    assert output_slot is not None
                    operation.attrs["issuer_warp"] = actor(output_slot)
                else:
                    if compute_slot is not None:
                        operation.attrs["issuer_warp"] = actor(compute_slot)
            if isinstance(operation, (MatmulOp, TmaMoveOp)):
                if output_slot is not None and isinstance(operation, TmaMoveOp):
                    operation.attrs["issuer_warp"] = actor(output_slot)
                    if output_slot == 1:
                        remap_output(operation)
                elif compute_slot is not None:
                    operation.attrs["issuer_warp"] = actor(compute_slot)
            if isinstance(operation, AwaitOp):
                if output_slot is not None:
                    operation.attrs["issuer_warp"] = actor(output_slot)
                    if operation.lock.type.kind != "bulk":
                        operation.attrs["waiter_warps"] = participants(output_slot)
                elif compute_slot is not None:
                    operation.attrs["waiter_warps"] = [actor(compute_slot)]
            if isinstance(operation, (MoveOp, ElementwiseOp)) and output_slot is not None:
                operation.attrs["participant_warps"] = participants(output_slot)
                if output_slot == 1 and isinstance(operation, MoveOp):
                    remap_output(operation)
            if isinstance(operation, BarrierOp):
                label = operation.attrs.get("label", "")
                if label == "last_tmem_read":
                    assert output_slot is not None
                    operation.attrs["participant_warps"] = participants(output_slot)
                elif label == "tmem_slot_free":
                    assert compute_slot is not None
                    operation.attrs.update(
                        participant_warps=[actor(compute_slot)],
                        issuer_warp=actor(compute_slot),
                    )
                elif output_slot is not None:
                    assert operation.attrs["scope"] == "block"
                    operation.attrs.update(
                        participant_warps=participants(output_slot), id=10 + output_slot
                    )
            for block in operation.blocks:
                for child in block.ops:
                    walk(child, compute_slot, output_slot)

        for root in result[0]:
            walk(root)
        limit0 = IndexConstantOp.build(1)
        limit1 = IndexConstantOp.build(5)
        issuing = [
            IndexLtOp.build(coordinate, limit0.result()),
            IndexLtOp.build(coordinate, limit1.result()),
        ]
        additions.extend([limit0, limit1, *issuing])
        for root in result[0]:
            for operation in list(iter_ops(root)):
                if "compute_slot" not in operation.attrs:
                    continue
                slot = operation.attrs["compute_slot"]
                for i, child in enumerate(list(operation.body.ops)):
                    if (
                        isinstance(child, ForOp)
                        and child.attrs.get("label") == "shared_ring_rolled_k"
                    ):
                        release = handoff.barrier("arrive", "job_issue_handoff")
                        release.attrs["participant_warps"] = [actor(slot)]
                        region = IfOp.build(issuing[slot].result(), Block(ops=[child, release]))
                        if slot == 0:
                            region.attrs["participant_warps"] = [0]
                        else:
                            region.attrs["participant_warps"] = [4]
                            region = IfOp.build(active[slot].result(), Block(ops=[region]))
                            region.attrs["participant_warps"] = participants(slot)
                        operation.body.ops[i] = region
        # The terminal whole-cluster rendezvous includes all output participants
        # in each CTA before either owner can deallocate the paired TMEM arena.
        kernel.attrs["name"] += "_rotating"
        return [*additions, *result[0]], result[1]


def pipeline_loop(
    module: ModuleOp,
    label_filter_loop: str,
    label_filter_op: str,
    pipelining_offset: int,
    *,
    guarded_tail: bool = False,
) -> ModuleOp:
    """Convenience entry point."""
    return PipelineLoop(label_filter_loop, label_filter_op, pipelining_offset,
                        guarded_tail=guarded_tail).run(module)
