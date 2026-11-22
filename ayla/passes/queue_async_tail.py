"""Keep one MMA group queued through peeled, straight-line pipeline stages.

PipelineLoop queues completion inside a software-pipelined loop, but its
prologue and epilogue stages fully drain the WGMMA queue after every issue.
Within a block, delay each full await, and the operand refills that require
it, until after the next issue as a ``pending=1`` await. The last issue keeps
its full drain before accumulator reads.
"""

from __future__ import annotations

from ..base.ir import Block, find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AsyncOp,
    AwaitOp,
    ForOp,
    LockType,
    ModuleOp,
    MoveOp,
    TmaMoveOp,
    is_side_effect_free,
)
from ._utils import can_reorder


def _wgmma(op) -> bool:
    return (
        isinstance(op, AsyncOp)
        and isinstance(op.lock.type, LockType)
        and op.lock.type.kind == "wgmma"
    )


def _refill(op) -> bool:
    return is_side_effect_free(op) or (
        isinstance(op, AsyncOp)
        and op.lock.type.kind == "mbar"
        and all(isinstance(inner, (MoveOp, TmaMoveOp)) for inner in op.body.ops)
    )


def queue_block(block: Block) -> bool:
    changed = False
    i = 0
    while i < len(block.ops):
        wait = block.ops[i]
        if not (
            isinstance(wait, AwaitOp)
            and wait.lock.type.kind == "wgmma"
            and not wait.attrs.get("pending")
        ):
            i += 1
            continue
        end = i + 1
        while (
            end < len(block.ops)
            and _refill(block.ops[end])
            and not _wgmma(block.ops[end])
        ):
            end += 1
        # The next issue must follow its operand await, with no other WGMMA
        # await or accumulator use in between.
        nxt = end
        while nxt < len(block.ops) and (
            is_side_effect_free(block.ops[nxt])
            or isinstance(block.ops[nxt], AwaitOp)
            and block.ops[nxt].lock.type.kind == "mbar"
        ):
            nxt += 1
        if nxt >= len(block.ops) or not _wgmma(block.ops[nxt]):
            i += 1
            continue
        # Pure definitions stay in place; only the await and its refills move.
        refills = [op for op in block.ops[i + 1 : end] if not is_side_effect_free(op)]
        # A refill may move only past independent input waits and MMA reads.
        # In particular, moving the copy behind its own input await deadlocks,
        # and moving it behind an MMA that reads its destination uses stale data.
        if any(
            not can_reorder(refill, crossed)
            for refill in refills
            for crossed in block.ops[end : nxt + 1]
        ):
            i += 1
            continue
        queued = AwaitOp.build(wait.lock, attrs={**wait.attrs, "pending": 1})
        block.ops[nxt + 1 : nxt + 1] = [queued, *refills]
        for op in [wait, *refills]:
            block.ops.remove(op)
        changed = True
    return changed


class QueueAsyncTail(Pass):
    """Queue WGMMA completion in straight-line pipeline stages."""

    def run(self, module: ModuleOp) -> ModuleOp:
        for kernel in module.body.ops:
            for block in [
                kernel.body,
                *(loop.body for loop in find_ops_of_type(kernel, ForOp)),
            ]:
                queue_block(block)
        return module
