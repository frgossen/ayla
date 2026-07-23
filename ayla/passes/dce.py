"""Dead code elimination.

Removes side-effect-free ops whose results are unused.

TODO: Also remove AllocOp and AwaitOp when their results/locks are
unused — a dead alloc is harmless, and an await on a lock with no
pending async ops is a no-op.
"""
from __future__ import annotations

from ..gpu_dialect import ModuleOp, is_side_effect_free
from ..base.ir import Block, Op, Region
from ..base.pipeline import Pass


class DCE(Pass):

    def run(self, module: ModuleOp) -> ModuleOp:
        for region in module.regions:
            self.run_region(region)
        return module

    def run_region(self, region: Region) -> None:
        for block in region.blocks:
            self.run_block(block)

    def run_block(self, block: Block) -> None:
        for op in block.ops:
            for region in op.regions:
                self.run_region(region)

        used: set[int] = set()
        for op in block.ops:
            for v in op.operands:
                used.add(id(v))
            for region in op.regions:
                self._collect_uses(region, used)

    def _collect_uses(self, region: Region, used: set[int]) -> None:
        for block in region.blocks:
            for op in block.ops:
                for v in op.operands:
                    used.add(id(v))
                for region in op.regions:
                    self._collect_uses(region, used)

        new_ops: list[Op] = []
        for op in block.ops:
            if not is_side_effect_free(op) or any(
                id(r) in used for r in op.results
            ):
                new_ops.append(op)
        block.ops = new_ops


def dce(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return DCE().run(module)
