"""Dead code elimination.

Removes side-effect-free ops whose results are unused, along with empty async
groups and their awaits.
"""
from __future__ import annotations

from ..gpu_dialect import AsyncOp, AwaitOp, ModuleOp, is_side_effect_free
from ..base.ir import Block, Op
from ..base.pipeline import Pass


def _collect_uses(block: Block, used: set[int]) -> None:
    """Add every value `block` and its descendants read to `used`."""
    for op in block.ops:
        for v in op.operands:
            used.add(id(v))
        for inner in op.blocks:
            _collect_uses(inner, used)


def _in_empty_async_group(op: Op) -> bool:
    """Whether ``op`` is an async without work, or an await on only those.

    One makes no sense without the other, so a lock's asyncs and awaits go
    together. Deciding from the lock alone gives every member the same answer,
    whichever block it sits in.
    """
    if not isinstance(op, (AsyncOp, AwaitOp)):
        return False
    return all(
        isinstance(user, AwaitOp)
        or isinstance(user, AsyncOp) and not user.body.ops
        for user in op.lock.users()
    )


class DCE(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        for block in module.blocks:
            self.run_block(block)
        return module

    def run_block(self, block: Block) -> None:
        # Innermost blocks first, so the ops dropped there stop counting as
        # uses of the values this block defines.
        for op in block.ops:
            for inner in op.blocks:
                self.run_block(inner)

        # Backwards, so an op is judged only against the ops that survive
        # after it. That clears a whole dead chain in one sweep: dropping the
        # last of them takes the operands it was the only reader of with it.
        used: set[int] = set()
        kept: list[Op] = []
        for op in reversed(block.ops):
            if _in_empty_async_group(op) or is_side_effect_free(op) and not any(
                id(r) in used for r in op.results
            ):
                continue
            kept.append(op)
            for v in op.operands:
                used.add(id(v))
            for inner in op.blocks:
                _collect_uses(inner, used)
        block.ops = list(reversed(kept))


def dce(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return DCE().run(module)
