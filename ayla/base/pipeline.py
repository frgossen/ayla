"""Base class for IR passes."""
from __future__ import annotations

import re

from ..gpu_dialect import ModuleOp
from .ir import Block, Op


def op_label_matches(op: Op, label_filter: str | None) -> bool:
    """Whether an op's string label fully matches a filter.

    A ``None`` filter selects every op.
    """
    # TODO: Make all other label-selective passes use this helper too.
    if label_filter is None:
        return True
    label = op.attrs.get("label")
    return isinstance(label, str) and re.fullmatch(label_filter, label) is not None


class Pass:
    """Base class for all IR passes.

    Provides a default top-down walk: ``run`` → ``run_op`` →
    ``run_block`` → ``run_op`` (recursive).
    Subclasses override whichever level they need.
    """

    def __init__(self, **options: object) -> None:
        self._option_names = tuple(options)
        for name, value in options.items():
            if hasattr(type(self), name):
                raise ValueError(f"option shadows Pass attribute: {name}")
            setattr(self, name, value)

    def __repr__(self) -> str:
        args = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in self._option_names
        )
        return f"{type(self).__name__}({args})"

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
    With `dump_ir_on_change=True`, the complete module is printed after each
    direct child pass that changes its printed representation. Passes registered
    via `add_instrumentation_pass` run after each direct child pass, but do not
    themselves trigger instrumentation.

    A pipeline is itself a pass, so one nests inside another: a group of passes
    that always run together can be named once and added wherever a single pass
    goes. `run` is overridden outright, so the inherited walk never runs.
    """

    def __init__(self, *, dump_ir_on_change: bool = False) -> None:
        super().__init__(dump_ir_on_change=dump_ir_on_change)
        self.dump_ir_on_change = dump_ir_on_change
        self.passes: list[Pass] = []
        self.instrumentation_passes: list[Pass] = []

    def add_pass(self, p: Pass) -> None:
        self.passes.append(p)

    def add_instrumentation_pass(self, p: Pass) -> None:
        # TODO: Allow instrumentation passes, including verifiers, to be
        # enabled or disabled with an environment variable.
        self.instrumentation_passes.append(p)

    def run(self, module: ModuleOp) -> ModuleOp:
        for p in self.passes:
            before = repr(module) if self.dump_ir_on_change else None
            module = p.run(module)
            after = repr(module) if self.dump_ir_on_change else None
            if after != before:
                print(f"// ----- IR after {p!r} -----")
                print(after)
                print()
            for instrumentation_pass in self.instrumentation_passes:
                module = instrumentation_pass.run(module)
        return module
