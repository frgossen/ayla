"""Canonicalize: folding + CSE + DCE."""
from __future__ import annotations

from .cse import cse
from .dce import dce
from .folding import folding
from ..gpu_dialect import ModuleOp


def canonicalize(module: ModuleOp) -> ModuleOp:
    module = folding(module)
    module = cse(module)
    module = dce(module)
    return module
