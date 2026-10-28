"""Canonicalize: folding + CSE + DCE."""
from __future__ import annotations

from ..base.pipeline import Pipeline
from ..passes.cse import CSE
from ..passes.dce import DCE
from ..passes.folding import Folding


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
