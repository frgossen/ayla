from __future__ import annotations

import pytest

from ayla.analysis.interval_set import IntervalSet
from ayla.analysis.tile_dependency import (
    TileAccess,
    TileDependencyAnalyzer,
    TileTask,
    build_tile_dependency_graph,
)
from ayla.analysis.cross_loop_scheduler import (
    reduce_tile_dependencies,
    schedule_tiles,
    static_u32_expression,
    specialize_static_waits,
)
from ayla.analysis.tile_dependency import TileDependencyGraph
from ayla.base.ir import Block, Value
from ayla.gpu_dialect import (
    BF16Type,
    F32Type,
    GmemTensorType,
    IndexType,
    MoveOp,
    TileOp,
    ReshapeOp,
)


def region(*intervals):
    return IntervalSet.build(intervals)


def test_subwarp_head_rows_preserve_column_slice_dependencies():
    from ayla.gpu_dialect import (CollectiveAxis, ThreadIdOp, IndexConstantOp, IndexDivOp,
                                 IndexModOp, IndexMulOp, IndexAddOp, RmemTensorType)
    source = Value("projection", GmemTensorType.build((16, 4608), BF16Type()))
    tile = Value("tile", IndexType())
    tid = ThreadIdOp.build(CollectiveAxis.X)
    block = Block(ops=[tid])

    def constant(value):
        op = IndexConstantOp.build(value)
        block.append(op)
        return op.result()

    def binary(kind, lhs, rhs):
        op = kind.build(lhs, rhs)
        block.append(op)
        return op.result()

    head = binary(IndexDivOp, tid.value, constant(16))
    row = binary(IndexDivOp, head, constant(2))
    part = binary(IndexModOp, head, constant(2))
    column = binary(IndexMulOp, part, constant(128))
    column = binary(IndexAddOp, column, constant(4096))
    lane = binary(IndexModOp, tid.value, constant(16))
    column = binary(IndexAddOp, column, lane)
    for offset in range(0, 128, 16):
        at = binary(IndexAddOp, column, constant(offset))
        view = TileOp.build(source, [row, at], GmemTensorType.build((1, 1), BF16Type(), strides=(4608, 1)))
        block.append(view)
        block.append(MoveOp.build(Value(None, RmemTensorType.build((1, 1), BF16Type())), view.result(), collective={}))
    analyzer = TileDependencyAnalyzer(threads=256)
    [read] = analyzer.accesses(block, tile, 0, thread_group_size=16)
    expected = region(*[((row * 4608 + 4096) * 2, (row * 4608 + 4352) * 2) for row in range(8)])
    assert read.region == expected
    [coarse] = analyzer.accesses(block, tile, 0)
    assert not expected.difference(coarse.region).intervals
    assert coarse.region.difference(expected).intervals


def test_interval_algebra_preserves_holes_and_boundaries():
    a = region((5, 8), (0, 2), (2, 4), (8, 10), (7, 7))
    b = region((1, 3), (6, 9))
    assert a.intervals == ((0, 4), (5, 10))
    assert a.intersection(b).intervals == ((1, 3), (6, 9))
    assert a.difference(b).intervals == ((0, 1), (3, 4), (5, 6), (9, 10))
    assert a.union(b) == a
    assert not region((0, 4)).overlaps(region((4, 8)))
    assert region((1, 2)).shifted(3) == region((4, 5))
    with pytest.raises(ValueError):
        region((4, 2))


def test_counter_fan_in_deduplicates_accesses_and_shared_consumers():
    buffer = Value("h", GmemTensorType.build([8], F32Type()))

    def access(lo, hi, write):
        return TileAccess(buffer, region((lo, hi)), write)

    tasks = (
        TileTask(0, 0, (access(0, 4, True), access(0, 4, True))),
        TileTask(0, 1, (access(4, 8, True),)),
        TileTask(1, 0, (access(0, 8, False),)),
        TileTask(1, 1, (access(0, 8, False),)),
    )
    graph = build_tile_dependency_graph(tasks)
    assert graph.predecessors == (
        frozenset(),
        frozenset(),
        frozenset({0, 1}),
        frozenset({0, 1}),
    )
    plan = schedule_tiles(graph)
    assert len(plan.events) == 1
    assert plan.events[0].producers == (0, 1)
    assert plan.events[0].consumers == (2, 3)
    assert plan.scratch_words == 64
    assert plan.publications == (1, 1)


def test_snapshot_and_scratch_overwrite_dependencies():
    buffer = Value("scratch", GmemTensorType.build([8], F32Type()))
    read = TileAccess(buffer, region((0, 8)), False)
    write = TileAccess(buffer, region((0, 8)), True)
    graph = build_tile_dependency_graph(
        tuple(TileTask(i, 0, (a,)) for i, a in enumerate([write, read, write]))
    )
    assert graph.predecessors == (frozenset(), frozenset({0}), frozenset({0, 1}))


def test_residual_ancestors_do_not_duplicate_gemm_readiness_counters():
    # Two residual rows feed a projection, followed by two down column tiles.
    # Each final row reads its residual and both down outputs. The down tiles
    # already order both residual producers through the projection.
    graph = TileDependencyGraph(
        tuple(
            TileTask(root, tile, ())
            for root, tile in [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]
        ),
        tuple(
            frozenset(deps)
            for deps in [(), (), (0, 1), (2,), (2,), (0, 3, 4), (1, 3, 4)]
        ),
    )
    reduced = reduce_tile_dependencies(graph)
    assert reduced.predecessors[-2:] == (frozenset({3, 4}), frozenset({3, 4}))
    plan = schedule_tiles(graph)
    assert plan.records[-2][2:4] == plan.records[-1][2:4]
    assert plan.records[-1][3] == 2
    assert plan.records[3][5] == plan.records[4][5] == 1
    # Analysis still records every actual access edge for diagnostics.
    assert graph.predecessors[-1] == frozenset({1, 3, 4})


def test_dependency_reduction_preserves_all_reachable_ordering():
    import random

    rng = random.Random(29)
    for _ in range(20):
        graph = TileDependencyGraph(
            tuple(TileTask(i, 0, ()) for i in range(24)),
            tuple(
                frozenset(p for p in range(i) if rng.random() < 0.25) for i in range(24)
            ),
        )
        reduced = reduce_tile_dependencies(graph)

        def reachable(g, task):
            seen, pending = set(), list(g.predecessors[task])
            while pending:
                producer = pending.pop()
                if producer not in seen:
                    seen.add(producer)
                    pending.extend(g.predecessors[producer])
            return seen

        for task in range(len(graph.tasks)):
            assert reachable(graph, task) == reachable(reduced, task)
        assert schedule_tiles(graph).order == schedule_tiles(reduced).order


def test_static_workers_acquire_each_event_once_per_launch():
    graph = TileDependencyGraph(
        tuple(TileTask(i, 0, ()) for i in range(8)),
        tuple(
            frozenset(deps)
            for deps in [(), (), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        ),
    )
    plan = schedule_tiles(graph)
    specialized = specialize_static_waits(plan, 2)
    assert [record[2] for record in specialized.records] == [0, 0, 1, 1, 0, 0, 0, 0]
    assert all(record[3] == 0 for record in specialized.records if record[2] == 0)
    assert all(
        before[:2] == after[:2] and before[4:] == after[4:]
        for before, after in zip(plan.records, specialized.records, strict=True)
    )
    assert specialized.order == plan.order
    assert specialized.publications == plan.publications
    with pytest.raises(ValueError, match="worker count"):
        specialize_static_waits(plan, 0)


def test_static_local_dependencies_use_proxy_fences_without_global_events():
    graph = TileDependencyGraph(
        tuple(TileTask(i // 2, i % 2, ()) for i in range(6)),
        tuple(frozenset(deps) for deps in [(), (), (0,), (1,), (2,), (3,)]),
    )
    plan = specialize_static_waits(schedule_tiles(graph), 2)
    assert not plan.events
    assert not plan.publications
    assert [record[2:4] for record in plan.records] == [
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]


def test_static_acquire_covers_transitive_producer_frontier():
    # Worker zero acquires 1 -> 2, then consumes 1 again via a different
    # readiness event. That earlier producer is already ordered by the acquire.
    graph = TileDependencyGraph(
        tuple(TileTask(i, 0, ()) for i in range(7)),
        tuple(frozenset(deps) for deps in [(), (), (1,), (), (2,), (), (1,)]),
    )
    original = schedule_tiles(graph)
    plan = specialize_static_waits(original, 2)
    assert plan.records[2][2] != 0
    assert plan.records[4][2:4] == (0, 1)
    assert plan.records[6][2:4] == (0, 0)


def test_static_proof_uses_configured_task_owners():
    graph = TileDependencyGraph(
        tuple(TileTask(i // 3, i % 3, ()) for i in range(6)),
        tuple(frozenset(deps) for deps in [(), (), (), (0,), (1,), (2,)]),
    )
    plan = specialize_static_waits(schedule_tiles(graph), 2, workers=(1, 0, 1, 1, 0, 1))
    assert not plan.events and not plan.publications
    assert [record[2:4] for record in plan.records[3:]] == [(0, 1), (0, 1), (0, 0)]
    with pytest.raises(ValueError, match="ownership"):
        specialize_static_waits(schedule_tiles(graph), 2, workers=(0,))


def test_scalar_acquire_does_not_elide_a_later_tma_proxy_fence():
    graph = TileDependencyGraph(
        tuple(TileTask(i, 0, ()) for i in range(3)),
        (frozenset(), frozenset({0}), frozenset({0})),
    )
    plan = specialize_static_waits(
        schedule_tiles(graph), 2, workers=(0, 1, 1),
        async_tasks=frozenset({0, 2}),
    )
    assert plan.records[1][2] != 0
    # The scalar acquire orders generic loads, and the later TMA reader
    # crosses the proxy even though it needs no additional counter acquire.
    assert plan.records[2][2:4] == (0, 1)


@pytest.mark.parametrize(
    "values",
    [
        tuple(1 + tile % 4 for tile in range(128)),
        tuple(9 + tile // 128 for tile in range(512)),
        tuple(29 + tile % 4 + 4 * (tile // 32) for tile in range(128)),
        (32,) * 104,
    ],
)
def test_static_counter_arithmetic_matches_every_tile(values):
    expression = static_u32_expression(values)
    assert expression is not None
    base, terms = expression
    assert (
        tuple(
            base + sum(c * ((tile // stride) % period) for c, stride, period in terms)
            for tile in range(len(values))
        )
        == values
    )


def test_irregular_static_counter_keys_keep_table_fallback():
    assert static_u32_expression((3, 4, 3, 8, 2, 7)) is None


def test_ready_consumers_pipeline_before_unrelated_producer_tiles():
    buffer = Value("h", GmemTensorType.build([8], F32Type()))
    tasks = tuple(
        TileTask(r, t, (TileAccess(buffer, region((t * 4, t * 4 + 4)), r == 0),))
        for r in range(2)
        for t in range(2)
    )
    graph = build_tile_dependency_graph(tasks)
    assert schedule_tiles(graph).order == (0, 1, 2, 3)
    assert schedule_tiles(graph, tile_order="memory").order == (0, 2, 1, 3)


def test_gate_up_views_have_disjoint_row_intervals():
    buffer = Value("packed", GmemTensorType.build([4, 8], BF16Type()))
    tile_id = Value("tile", IndexType())
    gate = TileOp.build(
        buffer, [0, 0], GmemTensorType.build([4, 4], BF16Type(), strides=[8, 1])
    )
    up = TileOp.build(
        buffer, [0, 1], GmemTensorType.build([4, 4], BF16Type(), strides=[8, 1])
    )
    dst = Value("dst", GmemTensorType.build([4, 4], BF16Type()))

    def reads(view):
        body = Block(ops=[view, MoveOp.build(dst, view.result(), collective={})])
        return next(
            a.region
            for a in TileDependencyAnalyzer().accesses(body, tile_id, 0)
            if a.root is buffer
        )

    assert reads(gate).intervals == ((0, 8), (16, 24), (32, 40), (48, 56))
    assert not reads(gate).overlaps(reads(up))


def test_dynamic_tile_ids_and_reshape_share_root_byte_coordinates():
    buffer = Value("input", GmemTensorType.build([4, 4], F32Type()))
    tile_id = Value("tile", IndexType())
    reshaped = ReshapeOp.build(buffer, GmemTensorType.build([16], F32Type()))
    tile = TileOp.build(
        reshaped.result(), [tile_id], GmemTensorType.build([4], F32Type())
    )
    dst = Value("dst", GmemTensorType.build([4], F32Type()))
    body = Block(ops=[reshaped, tile, MoveOp.build(dst, tile.result(), collective={})])
    accesses = TileDependencyAnalyzer().accesses(body, tile_id, 2)
    assert next(a.region for a in accesses if a.root is buffer).intervals == ((32, 48),)


def test_unknown_indices_are_conservative_and_dynamic_shapes_require_specialization():
    buffer = Value("input", GmemTensorType.build([16], F32Type()))
    tile_id = Value("tile", IndexType())
    unknown = Value("unknown", IndexType())
    tile = TileOp.build(buffer, [unknown], GmemTensorType.build([4], F32Type()))
    dst = Value("dst", GmemTensorType.build([4], F32Type()))
    body = Block(ops=[tile, MoveOp.build(dst, tile.result(), collective={})])
    accesses = TileDependencyAnalyzer().accesses(body, tile_id, 0)
    assert next(a.region for a in accesses if a.root is buffer).intervals == ((0, 64),)
    dynamic = Value("dynamic", GmemTensorType.build([None], F32Type()))
    with pytest.raises(ValueError, match="specialize"):
        TileDependencyAnalyzer().accesses(
            Block(ops=[MoveOp.build(dst, dynamic, collective={})]), tile_id, 0
        )
    dynamic_stride = Value(
        "dynamic_stride", GmemTensorType.build([4], F32Type(), strides=[None])
    )
    with pytest.raises(ValueError, match="specialize"):
        TileDependencyAnalyzer().accesses(
            Block(ops=[MoveOp.build(dst, dynamic_stride, collective={})]), tile_id, 0
        )


def test_partial_columns_do_not_spill_into_next_row():
    buffer = Value("input", GmemTensorType.build([2, 5], F32Type()))
    tile_id = Value("tile", IndexType())
    tile = TileOp.build(
        buffer,
        [0, 1],
        GmemTensorType.build([2, 3], F32Type(), strides=[5, 1]),
        maybe_partial=True,
    )
    dst = Value("dst", GmemTensorType.build([2, 3], F32Type()))
    accesses = TileDependencyAnalyzer().accesses(
        Block(ops=[tile, MoveOp.build(dst, tile.result(), collective={})]), tile_id, 0
    )
    assert next(a.region for a in accesses if a.root is buffer).intervals == (
        (12, 20),
        (32, 40),
    )


def test_interval_algebra_matches_integer_set_reference():
    import random

    rng = random.Random(27)
    for _ in range(30):

        def make_set():
            intervals = [
                (min(a, b), max(a, b))
                for _ in range(4)
                for a, b in [(rng.randrange(-8, 9), rng.randrange(-8, 9))]
            ]
            reference = {i for lo, hi in intervals for i in range(lo, hi)}
            return IntervalSet.build(intervals), reference

        a, aa = make_set()
        b, bb = make_set()

        def integers(intervals):
            return {i for lo, hi in intervals.intervals for i in range(lo, hi)}

        assert integers(a.union(b)) == aa | bb
        assert integers(a.intersection(b)) == aa & bb
        assert integers(a.difference(b)) == aa - bb
        assert a.overlaps(b) == bool(aa & bb)
    with pytest.raises(ValueError, match="normalized"):
        IntervalSet(((0, 2), (1, 3)))
