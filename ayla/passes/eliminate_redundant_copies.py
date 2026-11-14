"""Reuse completed copies while their source and destination remain unchanged."""
from ..base.ir import Block, Value
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AsyncOp, AwaitOp, BarrierOp, GmemTensorType, MoveOp, SmemTensorType,
    memory_effects, root_buffers,
)


def _is_operand_copy(op):
    return (isinstance(op, MoveOp) and isinstance(op.src.type, GmemTensorType)
            and isinstance(op.dst.type, SmemTensorType) and op.attrs.get('tma')
            and all(getattr(root.definition, 'attrs', {}).get('reuse_group') is None
                    for root in root_buffers(op.dst) | root_buffers(op.src)))


class EliminateRedundantCopies(Pass):
    """Remove repeated TMA operands after observing the earlier completion.

    Facts are local to one straight-line block. A write invalidates every copy
    whose source or destination may share a physical root with it. Pending
    copies become available only at their await; control-flow boundaries discard
    facts. At least one move remains in each event, preserving its completion.
    """

    def __init__(self, *, preserve_barrier_facts: bool = False):
        super().__init__(preserve_barrier_facts=preserve_barrier_facts)

    def run_block(self, block: Block) -> None:
        copies: dict[Value, Value] = {}
        pending: dict[Value, list[tuple[Value, Value]]] = {}

        def invalidate(writes):
            def preserved(dst, src):
                return not writes & (root_buffers(dst) | root_buffers(src))
            for dst, src in list(copies.items()):
                if not preserved(dst, src):
                    del copies[dst]
            for lock in pending:
                pending[lock] = [(dst, src) for dst, src in pending[lock]
                                 if preserved(dst, src)]

        for op in block.ops:
            if self.preserve_barrier_facts and isinstance(op, BarrierOp):
                continue
            if isinstance(op, AwaitOp):
                if op.attrs.get('pending', 0) or op.waiter_warps is not None:
                    pending.pop(op.lock, None)
                    continue
                for dst, src in pending.pop(op.lock, []):
                    copies[dst] = src
                continue
            if isinstance(op, AsyncOp) and all(isinstance(inner, MoveOp) for inner in op.body.ops):
                moves = list(op.body.ops)
                remaining = [move for move in moves if not (
                    _is_operand_copy(move) and copies.get(move.dst) is move.src)]
                if not remaining and moves:
                    remaining = [moves[-1]]
                op.body.ops = remaining
                effects = memory_effects(op)
                if effects is None:
                    copies.clear(); pending.clear()
                    continue
                invalidate({root for e in effects if e.kind == 'W' for root in e.roots})
                pending[op.lock] = [(move.dst, move.src) for move in remaining
                                    if _is_operand_copy(move)]
                continue
            if op.blocks and not isinstance(op, AsyncOp):
                # Loops and guards can change facts or execute zero times.
                copies.clear(); pending.clear()
                self.run_op(op)
                continue
            effects = memory_effects(op)
            if effects is None:
                copies.clear(); pending.clear()
            else:
                invalidate({root for e in effects if e.kind == 'W' for root in e.roots})
