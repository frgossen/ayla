"""Tests for the generic dataflow analysis framework."""

from dataclasses import dataclass

import pytest

from ayla.base.ir import Block, Op
from ayla.gpu_dialect import ForOp, IfOp, ModuleOp
from ayla.analysis.dataflow import (
    DataflowAnalysis,
    DataflowState,
    transfers,
)


@dataclass(frozen=True)
class SeenState(DataflowState):
    names: frozenset[str] = frozenset()

    @classmethod
    def empty(cls) -> "SeenState":
        return cls()

    def clone(self) -> "SeenState":
        return self

    def join(self, other: "SeenState") -> "SeenState":
        return SeenState(self.names | other.names)


class SeenAnalysis(DataflowAnalysis[SeenState]):
    def __init__(
        self,
        direction: DataflowAnalysis.Direction = DataflowAnalysis.Direction.FORWARD,
    ) -> None:
        super().__init__(SeenState, direction=direction)
        self.visits: dict[Op, int] = {}

    @transfers(Op)
    def transfer_named(self, op: Op, state: SeenState) -> SeenState:
        name = op.attrs.get("name")
        assert isinstance(name, str)
        self.visits[op] = self.visits.get(op, 0) + 1
        if name.startswith("-"):
            return SeenState(state.names - {name.removeprefix("-")})
        return SeenState(state.names | {name})


def named_op(name: str) -> Op:
    return Op(operands=[], results=[], attrs={"name": name})


def analyze_ops(analysis: SeenAnalysis, ops: list[Op]) -> None:
    analysis.analyze(ModuleOp.build(Block(ops=ops)))


def test_live_in_requires_a_completed_analysis():
    op = named_op("op")
    analysis = SeenAnalysis()

    with pytest.raises(AssertionError, match="analysis has not run"):
        analysis.live_in(op)
    with pytest.raises(AssertionError, match="analysis has not run"):
        analysis.live_out(op)


def test_unregistered_operation_fails_loudly():
    class UnregisteredOp(Op):
        pass

    analysis = SeenAnalysis()
    with pytest.raises(
        NotImplementedError,
        match="no transfer function registered for UnregisteredOp",
    ):
        analyze_ops(analysis, [UnregisteredOp(operands=[], results=[])])


def test_forward_dataflow_joins_optional_regions():
    before = named_op("before")
    conditional = named_op("conditional")
    after = named_op("after")
    if_op = IfOp(operands=[], results=[], blocks=[Block(ops=[conditional])])

    analysis = SeenAnalysis()
    analyze_ops(analysis, [before, if_op, after])

    assert analysis.live_in(conditional).names == frozenset({"before"})
    assert analysis.live_in(after).names == frozenset({"before", "conditional"})
    assert analysis.live_out(before).names == frozenset({"before"})


def test_forward_dataflow_iterates_loops_to_a_fixpoint():
    body_op = named_op("loop")
    after = named_op("after")
    loop = ForOp.build(0, 4)
    loop.body.ops = [body_op]

    analysis = SeenAnalysis()
    analyze_ops(analysis, [loop, after])

    assert analysis.visits[body_op] == 2
    assert analysis.live_in(after).names == frozenset({"loop"})


def test_loop_preserves_the_zero_iteration_path():
    loop = ForOp.build(0, 4)
    loop.body.ops = [named_op("-before")]
    before = named_op("before")
    after = named_op("after")

    analysis = SeenAnalysis()
    analyze_ops(analysis, [before, loop, after])

    assert analysis.live_in(after).names == frozenset({"before"})


def test_backward_dataflow_visits_operations_in_reverse():
    first = named_op("first")
    second = named_op("second")

    analysis = SeenAnalysis(direction=DataflowAnalysis.Direction.BACKWARD)
    analyze_ops(analysis, [first, second])

    assert analysis.live_in(second).names == frozenset({"second"})
    assert analysis.live_in(first).names == frozenset({"first", "second"})
    assert analysis.live_out(second).names == frozenset()
    assert analysis.live_out(first).names == frozenset({"second"})
