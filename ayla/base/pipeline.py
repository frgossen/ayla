"""Base class for IR passes."""
from __future__ import annotations

from ..gpu_dialect import ModuleOp
from .ir import Block, Op


class Pass:
    """Base class for all IR passes.

    Provides a default top-down walk: ``run`` → ``run_op`` →
    ``run_block`` → ``run_op`` (recursive).
    Subclasses override whichever level they need.
    """

    def run(self, module: ModuleOp) -> ModuleOp:
        self.run_op(module)
        return module

    def run_op(self, op: Op) -> None:
        for block in op.blocks:
            self.run_block(block)

    def run_block(self, block: Block) -> None:
        for op in block.ops:
            self.run_op(op)


class Pipeline(Pass):
    """A sequence of passes invoked one after the other.

    Subclasses register passes via `add_pass` (typically in `__init__`).

    A pipeline is itself a pass, so one nests inside another: a group of passes
    that always run together can be named once and added wherever a single pass
    goes. `run` is overridden outright, so the inherited walk never runs.
    """

    def __init__(self) -> None:
        self.passes: list[Pass] = []

    def add_pass(self, p: Pass) -> None:
        self.passes.append(p)

    def run(self, module: ModuleOp) -> ModuleOp:
        for p in self.passes:
            module = p.run(module)
        return module
