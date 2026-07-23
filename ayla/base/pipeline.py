"""Base class for IR passes."""
from __future__ import annotations

from ..gpu_dialect import ModuleOp
from .ir import Block, Op, Region


class Pass:
    """Base class for all IR passes.

    Provides a default top-down walk: ``run`` → ``run_op`` →
    ``run_region`` → ``run_block`` → ``run_op`` (recursive).
    Subclasses override whichever level they need.
    """

    def run(self, module: ModuleOp) -> ModuleOp:
        self.run_op(module)
        return module

    def run_op(self, op: Op) -> None:
        for region in op.regions:
            self.run_region(region)

    def run_region(self, region: Region) -> None:
        for block in region.blocks:
            self.run_block(block)

    def run_block(self, block: Block) -> None:
        for op in block.ops:
            self.run_op(op)


class Pipeline:
    """A sequence of passes invoked one after the other.

    Subclasses register passes via `add_pass` (typically in `__init__`).
    """

    def __init__(self) -> None:
        self.passes: list[Pass] = []

    def add_pass(self, p: Pass) -> None:
        self.passes.append(p)

    def run(self, module: ModuleOp) -> ModuleOp:
        for p in self.passes:
            module = p.run(module)
        return module
