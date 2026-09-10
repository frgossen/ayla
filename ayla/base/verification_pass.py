"""Base class for type-directed IR verification passes."""
from __future__ import annotations

from typing import Callable, ClassVar

from .ir import Op
from .pipeline import Pass


def verifies(*op_types: type[Op], stage: object | None = None):
    """Mark a method as a verifier for the given op types and stage."""
    for op_type in op_types:
        assert isinstance(op_type, type) and issubclass(op_type, Op), (
            f"verifies: expected an Op subclass, got {op_type!r}"
        )

    def deco(fn):
        registrations = getattr(fn, "_verifies", ())
        fn._verifies = (*registrations, (op_types, stage))
        return fn

    return deco


class VerificationPass(Pass):
    """Walk the IR and dispatch registered verification methods."""

    _verifiers: ClassVar[
        dict[
            tuple[type[Op], object | None],
            Callable[["VerificationPass", Op], None],
        ]
    ] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        table: dict[tuple[type[Op], object | None], Callable] = {}
        for klass in reversed(cls.__mro__):
            claimed: dict[tuple[type[Op], object | None], str] = {}
            for name, member in vars(klass).items():
                for op_types, stage in getattr(member, "_verifies", ()):
                    for op_type in op_types:
                        key = op_type, stage
                        assert key not in claimed, (
                            f"{klass.__name__}: {op_type.__name__} at "
                            f"{stage} claimed by both {claimed[key]} and {name}"
                        )
                        claimed[key] = name
                        table[key] = member
        cls._verifiers = table

    def __init__(self, stage: object | None = None) -> None:
        self.stage = stage

    def run_op(self, op: Op) -> None:
        for (op_type, stage), fn in self._verifiers.items():
            if isinstance(op, op_type) and (
                stage is None or stage == self.stage
            ):
                fn(self, op)
        super().run_op(op)
