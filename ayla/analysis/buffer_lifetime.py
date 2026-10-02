"""Analysis of buffer lifetimes from first through last memory access."""

from __future__ import annotations

from dataclasses import dataclass

from ..base.ir import Op, Value, iter_ops
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdsOp,
    ClusterBlockIdsOp,
    ElementwiseOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LaneIdOp,
    MatmulOp,
    MoveOp,
    PTXTemplateOp,
    PtrAddOp,
    ReduceOp,
    ReinterpretOp,
    ReshapeOp,
    ConstantOp,
    SelectOp,
    ThreadIdsOp,
    TileOp,
    TmaMoveOp,
    ToPtrOp,
    WarpGroupIdOp,
    WarpIdsOp,
    memory_effects,
)
from .dataflow import DataflowAnalysis, DataflowState, transfers


@dataclass(frozen=True)
class BufferLivenessState(DataflowState):
    buffers: frozenset[Value] = frozenset()

    @classmethod
    def empty(cls) -> BufferLivenessState:
        return cls()

    def clone(self) -> BufferLivenessState:
        return self

    def join(self, other: BufferLivenessState) -> BufferLivenessState:
        return BufferLivenessState(self.buffers | other.buffers)


class _BufferAccessAnalysis(DataflowAnalysis[BufferLivenessState]):
    def __init__(self, direction: DataflowAnalysis.Direction) -> None:
        super().__init__(
            BufferLivenessState,
            direction=direction,
        )

    def boundary_state(self, op: Op) -> BufferLivenessState:
        if not isinstance(op, KernelOp):
            return super().boundary_state(op)
        effects = memory_effects(op)
        if effects is None:
            raise NotImplementedError(
                f"unmodelled memory effects for {type(op).__name__}"
            )
        kind = "R" if self._direction == self.Direction.FORWARD else "W"
        arguments = set(op.body.arguments)
        return BufferLivenessState(frozenset(
            buffer
            for effect in effects
            if effect.kind == kind
            for buffer in effect.roots
            if buffer in arguments
        ))

    @transfers(
        AllocOp,
        AsyncOp,
        AwaitOp,
        ElementwiseOp,
        MatmulOp,
        MoveOp,
        PTXTemplateOp,
        ReduceOp,
        TmaMoveOp,
    )
    def transfer_access(
        self, op: Op, state: BufferLivenessState,
    ) -> BufferLivenessState:
        effects = memory_effects(op)
        if effects is None:
            raise NotImplementedError(
                f"unmodelled memory effects for {type(op).__name__}"
            )
        return BufferLivenessState(
            state.buffers | {
                buffer for effect in effects for buffer in effect.roots
            }
        )

    @transfers(
        BarrierOp,
        BlockIdsOp,
        ClusterBlockIdsOp,
        IndexAddOp,
        IndexConstantOp,
        IndexDivOp,
        IndexLtOp,
        IndexModOp,
        IndexMulOp,
        LaneIdOp,
        PtrAddOp,
        ReinterpretOp,
        ReshapeOp,
        ConstantOp,
        SelectOp,
        ThreadIdsOp,
        TileOp,
        ToPtrOp,
        WarpGroupIdOp,
        WarpIdsOp,
    )
    def transfer_no_memory_effect(
        self, op: Op, state: BufferLivenessState,
    ) -> BufferLivenessState:
        return state


class BufferLifetimeAnalysis:
    def __init__(self) -> None:
        self._past = _BufferAccessAnalysis(DataflowAnalysis.Direction.FORWARD)
        self._future = _BufferAccessAnalysis(DataflowAnalysis.Direction.BACKWARD)
        self._lifetimes: dict[Value, int] | None = None

    def analyze(self, op: Op) -> None:
        if self._lifetimes is not None:
            return
        self._past.analyze(op)
        self._future.analyze(op)
        self._lifetimes = {}
        operations = iter_ops(op, self.has_liveness)
        for index, current in enumerate(operations):
            for buffer in self.live_in(current).buffers:
                self._lifetimes[buffer] = (
                    self._lifetimes.get(buffer, 0) | 1 << (2 * index)
                )
            for buffer in self.live_out(current).buffers:
                self._lifetimes[buffer] = (
                    self._lifetimes.get(buffer, 0) | 1 << (2 * index + 1)
                )

    def has_liveness(self, op: Op) -> bool:
        return self._past.has_liveness(op) and self._future.has_liveness(op)

    def live_in(self, op: Op) -> BufferLivenessState:
        past = self._past.live_in(op)
        future = self._future.live_in(op)
        return BufferLivenessState(past.buffers & future.buffers)

    def live_out(self, op: Op) -> BufferLivenessState:
        past = self._past.live_out(op)
        future = self._future.live_out(op)
        return BufferLivenessState(past.buffers & future.buffers)

    def lifetime(self, buffer: Value) -> int:
        assert self._lifetimes is not None, "analysis has not run"
        return self._lifetimes.get(buffer, 0)
