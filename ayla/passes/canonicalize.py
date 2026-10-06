"""Canonicalize: folding + CSE + DCE."""
from __future__ import annotations

from .cse import CSE
from .dce import DCE
from .folding import Folding
from ..base.pipeline import Pipeline
from ..gpu_dialect import ModuleOp


class Canonicalize(Pipeline):
    """Fold what is constant, share what is duplicated, drop what is dead.

    The three run together often enough to be worth a name: folding exposes
    duplicates for CSE, and both leave values behind for DCE. Being a pipeline
    rather than a function, it also adds to another pipeline as a single pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_pass(Folding())
        self.add_pass(CSE())
        self.add_pass(DCE())


def canonicalize(module: ModuleOp) -> ModuleOp:
    return Canonicalize().run(module)
