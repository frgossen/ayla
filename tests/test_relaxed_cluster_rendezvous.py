"""A cluster rendezvous that orders only reads before writes has no release."""

import pytest

from ayla import Block, Op
from ayla.analysis.sync import (
    BarrierScope,
    BarrierTarget,
    ConcurrentMemoryEffect,
    SyncState,
    _after_barrier,
)
from ayla.gpu_dialect import AllocOp, BarrierOp, CollectiveScope, LockType
from ayla.passes.insert_syncs import _insert_barriers


def _settled(kind):
    return ConcurrentMemoryEffect(kind, CollectiveScope.CLUSTER_LEADER, None, None)


@pytest.mark.parametrize("reads_only", [False, True])
def test_a_reads_only_rendezvous_publishes_no_write(reads_only):
    root = AllocOp.build_with_fresh_id(LockType.build(kind="mbar")).result()
    state = SyncState(concurrent_memory_effects={root: {_settled("R"), _settled("W")}})
    after = _after_barrier(
        state, BarrierScope.CLUSTER, Op(operands=[], results=[]), reads_only=reads_only
    )
    kinds = {effect.kind for effect in after.concurrent_memory_effects[root]}
    assert kinds == ({"W"} if reads_only else set())


@pytest.mark.parametrize(
    ("scope", "publishes", "marked"),
    [
        (BarrierScope.CLUSTER, False, True),
        (BarrierScope.LEADER_TO_CLUSTER, False, True),
        (BarrierScope.CLUSTER, True, False),
        # A CTA barrier orders memory itself.
        (BarrierScope.BLOCK, False, False),
    ],
)
def test_insert_syncs_marks_rendezvous_that_publish_no_write(scope, publishes, marked):
    op = Op(operands=[], results=[])
    block = Block(ops=[op])
    _insert_barriers(
        block, {op: BarrierTarget(scope, publishes=publishes)}, num_threads=128
    )
    barrier = block.ops[0]
    assert isinstance(barrier, BarrierOp)
    assert ("publishes" in barrier.attrs) == marked
    assert barrier.attrs.get("publishes", True) is (not marked)
