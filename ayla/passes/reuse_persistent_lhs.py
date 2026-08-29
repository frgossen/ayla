"""Reuse the final LHS banks across a two-output persistent GEMM pair.

The marked direct-TMA K128 pipeline computes its second output with descending
K coordinates.  The first output leaves K[-2:] in two of the three LHS shared
buffers; rotating only the second phase's buffer views lets it consume those
resident banks and omit two 16 KiB TMA loads.  The terminal first-half
completion also releases B0 early enough to prefetch the next output's first
RHS bank across the remaining tensor work. All warps keep the same software
pipeline. This pass runs after synchronization insertion, then unrolls only the
two-iteration persistent loop so the existing loop-carried barriers remain
explicit while the two phases can have different K order and buffer mapping.
"""
from __future__ import annotations

from ..base.ir import Value, clone_op, iter_ops
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    AsyncCommitOp,
    AsyncOp,
    AwaitOp,
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    MatmulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TmemTensorType,
    TmaMoveOp,
    TileOp,
    root_buffer,
)
from .canonicalize import Canonicalize
from .unroll_loop import unroll_loop


_MARKER = "reuse_lhs_reverse_k"


def _walk_ops(ops):
    for op in ops:
        yield op
        for block in op.blocks:
            yield from _walk_ops(block.ops)


def _anchor(op):
    """Return the op whose parent block can hold setup for ``op``."""
    anchor = op
    while (
        anchor.parent_block is not None
        and isinstance(anchor.parent_block.parent_op, AsyncOp)
    ):
        anchor = anchor.parent_block.parent_op
    if anchor.parent_block is None:
        raise ValueError("persistent LHS reuse found a detached operation")
    return anchor


class ReusePersistentLhs(Pass):
    """Apply reverse-K LHS reuse to a structurally validated persistent pair."""

    def run(self, module: ModuleOp) -> ModuleOp:
        marked = [
            op for op in iter_ops(module)
            if isinstance(op, ForOp) and op.attrs.get(_MARKER)
        ]
        if not marked:
            return module
        if len(marked) != 1 or marked[0].trip_count != 2:
            raise ValueError(
                "persistent LHS reuse requires one two-iteration marked loop"
            )
        marked[0].attrs["label"] = "persistent_reverse_k_pair"
        module = unroll_loop(
            module, unroll_factor=3,
            label_filter="persistent_reverse_k_pair",
        )
        module = Canonicalize().run(module)

        kernel = module.body.ops[0]
        body = kernel.body
        outputs = [
            op for op in body.ops
            if isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType)
        ]
        if len(outputs) != 2:
            raise ValueError(
                "persistent LHS reuse requires exactly two tensor-memory outputs"
            )
        phase_start = body.ops.index(outputs[0]) + 1
        phase_end = body.ops.index(outputs[1])
        phase_ops = list(body.ops[phase_start:phase_end])
        middle = next(
            (
                op for op in phase_ops
                if isinstance(op, ForOp)
                and op.attrs.get("label") == "prefetch_lhs_middle_tail1"
            ),
            None,
        )
        if middle is None or middle.step != 3:
            raise ValueError(
                "persistent LHS reuse requires the grouped K128 middle loop"
            )
        iteration_count = middle.ub + 1
        if middle.lb != 1 or iteration_count < 5 or iteration_count % 3 != 2:
            raise ValueError(
                "persistent LHS reuse requires K128 iteration count congruent "
                "to two modulo three"
            )

        lhs_allocs = [
            op.result() for op in body.ops
            if isinstance(op, AllocOp)
            and isinstance(op.result().type, SmemTensorType)
            and tuple(op.result().type.sizes) == (128, 64)
        ]
        if len(lhs_allocs) != 3:
            raise ValueError(
                "persistent LHS reuse requires three M128xK64 shared buffers"
            )

        def smem_operand(op: TmaMoveOp) -> Value:
            return op.dst if isinstance(op.dst.type, SmemTensorType) else op.src

        tma_moves = [
            op for op in _walk_ops(phase_ops) if isinstance(op, TmaMoveOp)
        ]
        a_tmas = [
            op for op in tma_moves
            if root_buffer(smem_operand(op)) in lhs_allocs
        ]
        rhs_allocs = {
            root_buffer(smem_operand(op))
            for op in tma_moves
            if tuple(smem_operand(op).type.sizes) != (128, 64)
        }
        b_tmas = [
            op for op in tma_moves
            if root_buffer(smem_operand(op)) in rhs_allocs
        ]
        if len(a_tmas) != 10 or len(b_tmas) != 20 or len(rhs_allocs) != 2:
            raise ValueError(
                "persistent LHS reuse requires the three-A/two-split-B K128 "
                "prefetch pipeline"
            )

        middle_ops = set(_walk_ops(middle.body.ops))

        def partition(tmas):
            prefix = [
                op for op in tmas
                if op not in middle_ops
                and body.ops.index(_anchor(op)) < body.ops.index(middle)
            ]
            within = [op for op in tmas if op in middle_ops]
            suffix = [
                op for op in tmas
                if op not in middle_ops
                and body.ops.index(_anchor(op)) > body.ops.index(middle)
            ]
            return prefix, within, suffix

        prefix_a, middle_a, suffix_a = partition(a_tmas)
        prefix_b, middle_b, suffix_b = partition(b_tmas)
        if [len(values) for values in (
            prefix_a, middle_a, suffix_a, prefix_b, middle_b, suffix_b,
        )] != [3, 6, 1, 6, 12, 2]:
            raise ValueError("persistent LHS reuse found an unexpected TMA schedule")

        def set_coordinate(op: TmaMoveOp, coordinate: int) -> None:
            anchor = _anchor(op)
            constant = IndexConstantOp.build(coordinate * 64)
            anchor.parent_block.ops.insert(
                anchor.parent_block.ops.index(anchor), constant,
            )
            if len(op.indices) != 2:
                raise ValueError("persistent LHS reuse requires rank-two TMA")
            op.operands[3] = constant.result()

        last_iteration = iteration_count - 1
        for op, coordinate in zip(
            prefix_a,
            [2 * last_iteration, 2 * last_iteration + 1,
             2 * last_iteration - 2],
            strict=True,
        ):
            set_coordinate(op, coordinate)
        for op, coordinate in zip(suffix_a, [1], strict=True):
            set_coordinate(op, coordinate)
        for op, coordinate in zip(
            prefix_b,
            [2 * last_iteration, 2 * last_iteration,
             2 * last_iteration + 1, 2 * last_iteration + 1,
             2 * last_iteration - 2, 2 * last_iteration - 2],
            strict=True,
        ):
            set_coordinate(op, coordinate)
        for op, coordinate in zip(suffix_b, [1, 1], strict=True):
            set_coordinate(op, coordinate)

        middle.attrs.update({
            "lb": iteration_count - 2,
            "ub": 0,
            "step": -3,
        })
        scale = IndexConstantOp.build(128)
        base = IndexMulOp.build(middle.iv, scale.result())
        middle.body.ops[0:0] = [scale, base]

        def set_middle_offset(op: TmaMoveOp, offset: int) -> None:
            anchor = _anchor(op)
            if anchor.parent_block is not middle.body:
                raise ValueError("persistent LHS reuse lost middle-loop dominance")
            constant = IndexConstantOp.build(offset * 64)
            coordinate = IndexAddOp.build(base.result(), constant.result())
            index = middle.body.ops.index(anchor)
            middle.body.ops[index:index] = [constant, coordinate]
            op.operands[3] = coordinate.result()

        a_offsets = [1, -2, -1, -4, -3, -6]
        for op, offset in zip(middle_a, a_offsets, strict=True):
            set_middle_offset(op, offset)
        for op, offset in zip(
            middle_b,
            [offset for offset in a_offsets for _ in range(2)],
            strict=True,
        ):
            set_middle_offset(op, offset)

        replacement: dict[Value, Value] = {
            lhs_allocs[0]: lhs_allocs[2],
            lhs_allocs[1]: lhs_allocs[0],
            lhs_allocs[2]: lhs_allocs[1],
        }
        rotated_views: dict[Value, Value] = {}

        def rotate_view(value: Value, anchor) -> Value:
            if value in replacement:
                return replacement[value]
            cached = rotated_views.get(value)
            if cached is not None:
                return cached
            definition = value.definition
            if not isinstance(definition, TileOp):
                return value
            new_arg = rotate_view(definition.arg, anchor)
            if new_arg is definition.arg:
                return value
            cloned = clone_op(definition, {definition.arg: new_arg})
            anchor.parent_block.ops.insert(
                anchor.parent_block.ops.index(anchor), cloned,
            )
            rotated_views[value] = cloned.result()
            return cloned.result()

        # CSE may share pure TileOps between the two cloned phases. Copy views
        # at phase-two uses so rotating the ring cannot mutate phase one.
        for op in list(_walk_ops(phase_ops)):
            if isinstance(op, TileOp):
                continue
            anchor = _anchor(op)
            op.operands = [rotate_view(value, anchor) for value in op.operands]

        # Phase one left its final two K64 A banks in the rotated phase-two A0
        # and A1 slots. Keep the B transfers but remove those redundant loads.
        for op in a_tmas[:2]:
            if op.parent_block is None:
                raise ValueError("persistent LHS reuse found a detached TMA")
            op.parent_block.ops.remove(op)

        # The final K128 group did not previously need a partial completion.
        # Add one after its first four MMAs, when B0 is dead, and use it to
        # launch phase two's B0 while the final four MMAs and output-zero
        # epilogue proceed. Every warp observes the same half-completion and
        # continues through the same pipeline; only TMA's required thread-zero
        # issuer performs the transfer.
        output_zero_index = body.ops.index(outputs[0])
        terminal_mma = next(
            op for op in reversed(body.ops[:output_zero_index])
            if isinstance(op, AsyncOp)
            and sum(isinstance(inner, MatmulOp) for inner in op.body.ops) == 8
        )
        if any(
            isinstance(inner, AsyncCommitOp) for inner in terminal_mma.body.ops
        ):
            raise ValueError(
                "persistent LHS reuse expected an unsplit terminal MMA group"
            )
        phase_two_b0 = next(
            op for op in phase_ops
            if isinstance(op, AsyncOp)
            and len([
                inner for inner in op.body.ops
                if isinstance(inner, TmaMoveOp)
            ]) == 2
        )
        phase_two_mma = next(
            op for op in phase_ops
            if isinstance(op, AsyncOp)
            and any(
                isinstance(inner, AsyncCommitOp) for inner in op.body.ops
            )
        )
        half_lock = next(
            inner.lock for inner in phase_two_mma.body.ops
            if isinstance(inner, AsyncCommitOp)
        )
        terminal_matmuls = [
            inner for inner in terminal_mma.body.ops
            if isinstance(inner, MatmulOp)
        ]
        fourth_mma = terminal_matmuls[3]
        terminal_mma.body.ops.insert(
            terminal_mma.body.ops.index(fourth_mma) + 1,
            AsyncCommitOp.build(half_lock),
        )

        # Hoist the phase-two address/view setup with the transfer so every
        # operand dominates its new issue point. These are pure CTA-uniform
        # calculations and do not interact with output-zero's tensor data.
        b0_index = body.ops.index(phase_two_b0)
        setup = list(body.ops[output_zero_index + 1:b0_index])
        del body.ops[output_zero_index + 1:b0_index]
        terminal_index = body.ops.index(terminal_mma)
        body.ops[terminal_index:terminal_index] = setup
        body.ops.remove(phase_two_b0)
        terminal_index = body.ops.index(terminal_mma)
        body.ops[terminal_index + 1:terminal_index + 1] = [
            AwaitOp.build(half_lock),
            phase_two_b0,
        ]

        return Canonicalize().run(module)


def reuse_persistent_lhs(module: ModuleOp) -> ModuleOp:
    return ReusePersistentLhs().run(module)
