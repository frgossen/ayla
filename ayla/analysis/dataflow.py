"""Generic dataflow analysis over structured GPU IR."""

from __future__ import annotations

import enum
from collections.abc import Callable
from typing import ClassVar, Protocol, Self

from ..base.ir import Block, Op
from ..gpu_dialect import ForOp, IfOp, KernelOp, ModuleOp


def transfers[FnT: Callable[..., object]](
    *op_types: type[Op],
) -> Callable[[FnT], FnT]:
    """Register a method as the transfer function for ``op_types``."""
    for op_type in op_types:
        assert issubclass(op_type, Op), (
            f"transfers: expected an Op subclass, got {op_type!r}"
        )

    def decorator(fn: FnT) -> FnT:
        setattr(fn, "_transfers", op_types)
        return fn

    return decorator


class DataflowState(Protocol):
    """State propagated by a :class:`DataflowAnalysis`."""

    @classmethod
    def empty(cls) -> Self:
        """Return the state at the entry to a fresh analysis scope."""
        ...

    def clone(self) -> Self:
        """Return a copy that may be mutated independently."""
        ...

    def join(self, other: Self) -> Self:
        """Return the least upper bound of two control-flow paths."""
        ...


class DataflowAnalysis[StateT: DataflowState]:
    """Forward or backward analysis with structured control flow.

    Module and kernel bodies start with a fresh state and do not propagate
    their state into the enclosing scope. An ``if`` body joins with the path
    that does not execute it. A ``for`` body is analyzed to a fixpoint over its
    backedge.
    """

    class Direction(enum.Enum):
        FORWARD = enum.auto()
        BACKWARD = enum.auto()

    _transfers: ClassVar[dict[type[Op], Callable]] = {}

    def __init__(
        self,
        state_type: type[StateT],
        direction: Direction,
    ) -> None:
        self._state_type = state_type
        self._direction = direction
        self._live_in: dict[Op, StateT] | None = None
        self._live_out: dict[Op, StateT] | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        table: dict[type[Op], Callable] = {}
        for klass in reversed(cls.__mro__):
            claimed: dict[type[Op], str] = {}
            for name, member in vars(klass).items():
                for op_type in getattr(member, "_transfers", ()):
                    assert op_type not in claimed, (
                        f"{klass.__name__}: {op_type.__name__} claimed by both "
                        f"{claimed[op_type]} and {name}"
                    )
                    claimed[op_type] = name
                    table[op_type] = member
        cls._transfers = table

    def boundary_state(self, op: Op) -> StateT:
        """Return the state at a forward entry or backward exit boundary."""
        return self._state_type.empty()

    def _transfer_op(self, op: Op, state: StateT) -> StateT:
        """Transfer ``state`` across one operation."""
        assert self._live_in is not None and self._live_out is not None
        if self._direction == self.Direction.FORWARD:
            self._live_in[op] = state.clone()
        # Container ops with isolated bodies are hard-coded by the framework.
        if isinstance(op, (ModuleOp, KernelOp)):
            self._transfer_block(op.body, self.boundary_state(op))
            result = state
        # Loop ops and their fixpoint behavior are hard-coded by the framework.
        elif isinstance(op, ForOp):
            header = state
            while True:
                live_out = self._transfer_block(op.body, header)
                joined = header.join(live_out)
                if joined == header:
                    result = header
                    break
                header = joined
        # If ops and their join behavior are hard-coded by the framework.
        elif isinstance(op, IfOp):
            result = state.join(self._transfer_block(op.body, state))
        else:
            transfer = self._transfers.get(type(op))
            if transfer is None:
                raise NotImplementedError(
                    f"no transfer function registered for {type(op).__name__}"
                )
            result = transfer(self, op, state)
        if self._direction == self.Direction.BACKWARD:
            self._live_in[op] = result.clone()
        return result

    def analyze(self, op: Op) -> None:
        """Analyze ``op`` starting from a fresh state."""
        if self._live_in is not None:
            return
        self._live_in = {}
        self._live_out = {}
        initial = self._state_type.empty()
        result = self._transfer_op(op, initial)
        self._live_out[op] = (
            result.clone()
            if self._direction == self.Direction.FORWARD
            else initial.clone()
        )

    def live_in(self, op: Op) -> StateT:
        """Return the state immediately before ``op`` in execution order."""
        assert self._live_in is not None, "analysis has not run"
        return self._live_in[op].clone()

    def has_liveness(self, op: Op) -> bool:
        """Whether live-in and live-out are available for ``op``."""
        assert self._live_in is not None, "analysis has not run"
        return op in self._live_in

    def live_out(self, op: Op) -> StateT:
        """Return the state immediately after ``op`` in execution order."""
        assert self._live_in is not None and self._live_out is not None, (
            "analysis has not run"
        )
        if op in self._live_out:
            return self._live_out[op].clone()
        block = op.parent_block
        assert block is not None
        index = block.ops.index(op)
        return self._live_in[block.ops[index + 1]].clone()

    def _transfer_block(self, block: Block, state: StateT) -> StateT:
        """Propagate ``state`` through the operations in ``block``."""
        live = state.clone()
        if block.ops and self._direction == self.Direction.BACKWARD:
            assert self._live_out is not None
            self._live_out[block.ops[-1]] = live.clone()
        ops = (
            block.ops
            if self._direction == self.Direction.FORWARD
            else reversed(block.ops)
        )
        for op in ops:
            live = self._transfer_op(op, live)
        if block.ops and self._direction == self.Direction.FORWARD:
            assert self._live_out is not None
            self._live_out[block.ops[-1]] = live.clone()
        return live
