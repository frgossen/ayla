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

Legality: see ``_check_legality``.
"""
from __future__ import annotations

from ..gpu_dialect import (
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexType,
    ModuleOp,
    effective_operands,
    has_ordering_effect,
    is_side_effect_free,
    memory_effects,
)
from ..base.ir import Block, Op, Value, clone_op
from ..base.lowering_pass import LoweringPass, lowers
from ._utils import label_matches


class PipelineLoop(LoweringPass):
    """Software-pipeline labeled ops in ForOp bodies."""

    label_filter_loop: str
    label_filter_op: str
    pipelining_offset: int

    def __init__(
        self,
        label_filter_loop: str,
        label_filter_op: str,
        pipelining_offset: int,
    ) -> None:
        super().__init__(
            label_filter_loop=label_filter_loop,
            label_filter_op=label_filter_op,
            pipelining_offset=pipelining_offset,
        )

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
                if effect_a.roots & effect_b.roots and (
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
        labeled_set, shared_set = self._compute_labeled_and_shared(body_ops)
        labeled = [o for o in body_ops if o in labeled_set]
        shared = [o for o in body_ops if o in shared_set]
        unlabeled = [o for o in body_ops if o not in labeled_set and o not in shared_set]
        if not labeled:
            return [op], []

        self._check_legality(body_ops, labeled_set, shared_set)

        # Prologue: shared + labeled ops for iv = lb, lb+step, ..., lb+(n-1)*step.
        prologue_ops: list[Op] = []
        for i in range(n):
            iv_const = IndexConstantOp.build(value=lb + i * step)
            prologue_ops.append(iv_const)
            vmap: dict[Value, Value] = {op.iv: iv_const.result()}
            prologue_ops.extend(
                clone_op(body_op, vmap) for body_op in shared + labeled
            )

        # Main loop body: shared + unlabeled at iv, then shared + labeled at iv + n*step.
        new_iv = Value(name=op.iv.name, type=IndexType())
        ahead_const = IndexConstantOp.build(value=n * step)
        ahead_iv = IndexAddOp.build(lhs=new_iv, rhs=ahead_const.result())
        new_body_ops: list[Op] = []
        vmap = {op.iv: new_iv}
        for body_op in shared + unlabeled:
            new_body_ops.append(clone_op(body_op, vmap))
        new_body_ops.extend([ahead_const, ahead_iv])
        vmap = {op.iv: ahead_iv.result()}
        for body_op in shared + labeled:
            new_body_ops.append(clone_op(body_op, vmap))
        new_body = Block(arguments=[new_iv], ops=new_body_ops)
        new_ub = ub - n * step
        new_for = ForOp.build(lb=lb, ub=new_ub, step=step, body=new_body,
                              attrs=op.extra_attrs)

        # Epilogue: shared + unlabeled ops for iv = new_ub, ..., ub-step.
        epilogue_ops: list[Op] = []
        for i in range(n):
            iv_const = IndexConstantOp.build(value=new_ub + i * step)
            epilogue_ops.append(iv_const)
            vmap = {op.iv: iv_const.result()}
            epilogue_ops.extend(
                clone_op(body_op, vmap) for body_op in shared + unlabeled
            )

        return [*prologue_ops, new_for, *epilogue_ops], []


def pipeline_loop(
    module: ModuleOp,
    label_filter_loop: str,
    label_filter_op: str,
    pipelining_offset: int,
) -> ModuleOp:
    """Convenience entry point."""
    return PipelineLoop(label_filter_loop, label_filter_op, pipelining_offset).run(module)
