"""Source-ordered roots with tile-level pipelining and shared readiness events.

Both static and dynamic device schedulers consume this same topological stream.
The source traversal retains written root order and the library's tile order.
Memory traversal prioritizes ready consumers and increasing destination offsets.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .tile_dependency import TileDependencyGraph

StaticU32Expression = tuple[int, tuple[tuple[int, int, int], ...]]


def static_u32_expression(values: tuple[int, ...]) -> StaticU32Expression | None:
    """Prove a small mixed-radix expression against every configured tile.

    A term is coefficient * ((tile // stride) % period). No approximation is
    allowed: irregular maps use the original immutable table instead.
    """
    if not values:
        return None
    residual = list(values)
    base = values[0]
    terms = []
    for _ in range(4):
        stride = next((i for i, value in enumerate(residual) if value != base), None)
        if stride is None:
            expression = (base, tuple(terms))
            assert all(
                base + sum(c * ((i // s) % p) for c, s, p in terms) == value
                for i, value in enumerate(values)
            )
            return expression
        coefficient = residual[stride] - base
        if coefficient <= 0:
            return None
        period = next(
            (
                j
                for j in range(2, -(-len(values) // stride))
                if residual[j * stride] != base + j * coefficient
            ),
            -(-len(values) // stride),
        )
        terms.append((coefficient, stride, period))
        residual = [
            value - coefficient * ((i // stride) % period)
            for i, value in enumerate(residual)
        ]
    return None


def reduce_tile_dependencies(graph: TileDependencyGraph) -> TileDependencyGraph:
    """Keep only edges needed to preserve the task graph's ordering.

    A consumer may read a residual from an earlier stage as well as a GEMM
    result. When the GEMM already depends on that residual's producer, its
    acquire/release chain orders both reads. Keeping the earlier edge makes
    every residual row's fan-in distinct and multiplies GEMM publications.
    Integer bitsets retain each task's ancestors in source order.
    """
    ancestors: list[int] = []
    predecessors = []
    for deps in graph.predecessors:
        covered = 0
        direct = set()
        # Later predecessors can cover earlier ones, since all edges point
        # backwards. An earlier predecessor cannot cover a later one.
        for producer in sorted(deps, reverse=True):
            bit = 1 << producer
            if covered & bit:
                continue
            direct.add(producer)
            covered |= bit | ancestors[producer]
        ancestors.append(covered)
        predecessors.append(frozenset(direct))
    return TileDependencyGraph(graph.tasks, tuple(predecessors))


@dataclass(frozen=True)
class ReadinessEvent:
    producers: tuple[int, ...]
    consumers: tuple[int, ...]


@dataclass(frozen=True)
class TileSchedule:
    order: tuple[int, ...]
    events: tuple[ReadinessEvent, ...]
    # Records are root, tile, wait event (+1, zero means none), expected count,
    # publication offset and publication length.
    records: tuple[tuple[int, ...], ...]
    publications: tuple[int, ...]

    @property
    def scratch_words(self) -> int:
        # Isolate hot fan-in counters (and the work queue) on cache lines.
        return 32 * (1 + len(self.events))


def specialize_static_waits(
    plan: TileSchedule, blocks: int, *, workers: tuple[int, ...] | None = None,
    async_tasks: frozenset[int] | None = None,
) -> TileSchedule:
    """Prove ordering from CTA program order and preceding acquire frontiers.

    Event zero with expected one requests only an async proxy fence: producer
    completion is already ordered on this CTA, but its ordinary global writes
    must be visible to later TMA loads. Event/expected both zero need no fence.
    Unused global events and their publications are removed after the proof.
    """
    if type(blocks) is not int or blocks < 1:
        raise ValueError("static waits require a positive worker count")
    if workers is not None and (
        len(workers) != len(plan.records)
        or any(worker < 0 or worker >= blocks for worker in workers)
    ):
        raise ValueError("static ownership must specify one valid worker per task")
    incoming = {
        consumer: event.producers
        for event in plan.events
        for consumer in event.consumers
    }
    ancestors: dict[int, int] = {}
    for task in plan.order:
        ancestors[task] = 0
        for producer in incoming.get(task, ()):
            ancestors[task] |= (1 << producer) | ancestors[producer]
    completed = [0] * blocks
    visible = [0] * blocks
    records = []
    for position, record in enumerate(plan.records):
        event = record[2]
        worker = position % blocks if workers is None else workers[position]
        async_proxy = async_tasks is None or plan.order[position] in async_tasks
        if event:
            required = 0
            for producer in plan.events[event - 1].producers:
                required |= (1 << producer) | ancestors[producer]
            if required & ~(visible[worker] if async_proxy else completed[worker]) == 0:
                record = (*record[:2], 0, 0, *record[4:])
            elif required & ~completed[worker] == 0:
                record = (*record[:2], 0, 1, *record[4:])
                if async_proxy:
                    visible[worker] = completed[worker]
            else:
                completed[worker] |= required
                if async_proxy:
                    visible[worker] = completed[worker]
        records.append(record)
        completed[worker] |= 1 << plan.order[position]
    needed = sorted({record[2] for record in records if record[2]})
    remap = {event: i + 1 for i, event in enumerate(needed)}
    publications = []
    compact = []
    for record in records:
        arrivals = [
            remap[event]
            for event in plan.publications[record[4] : record[4] + record[5]]
            if event in remap
        ]
        compact.append(
            (
                *record[:2],
                remap.get(record[2], 0),
                record[3],
                len(publications),
                len(arrivals),
            )
        )
        publications.extend(arrivals)
    return TileSchedule(
        plan.order,
        tuple(plan.events[event - 1] for event in needed),
        tuple(compact),
        tuple(publications),
    )


def schedule_tiles(
    graph: TileDependencyGraph, *, tile_order: str = "source"
) -> TileSchedule:
    if tile_order not in ("source", "memory"):
        raise ValueError("tile_order must be 'source' or 'memory'")

    graph = reduce_tile_dependencies(graph)

    def key(i: int) -> tuple[int, int, int]:
        task = graph.tasks[i]
        priority = task.tile_id
        if tile_order == "memory":
            priority = min(
                (
                    a.region.intervals[0][0]
                    for a in task.accesses
                    if a.write and a.region.intervals
                ),
                default=priority,
            )
        return task.root_id if tile_order == "source" else -task.root_id, priority, i

    successors: list[list[int]] = [[] for _ in graph.tasks]
    remaining = [len(deps) for deps in graph.predecessors]
    ready: list[tuple[int, int, int]] = []
    for i, (task, deps) in enumerate(zip(graph.tasks, graph.predecessors, strict=True)):
        for p in deps:
            successors[p].append(i)
        if not deps:
            heapq.heappush(ready, key(i))
    order = []
    while ready:
        _, _, i = heapq.heappop(ready)
        order.append(i)
        for s in successors[i]:
            remaining[s] -= 1
            if remaining[s] == 0:
                heapq.heappush(ready, key(s))
    if len(order) != len(graph.tasks):
        raise ValueError("tile dependency graph contains a cycle")
    # Consumers with identical fan-in share one counter. A producer signals
    # once per event, even if it contributes multiple accesses to the consumer.
    by_producers: dict[tuple[int, ...], list[int]] = {}
    for i, deps in enumerate(graph.predecessors):
        if deps:
            by_producers.setdefault(tuple(sorted(deps)), []).append(i)
    events = tuple(ReadinessEvent(p, tuple(c)) for p, c in by_producers.items())
    waits: dict[int, int] = {}
    signals: list[list[int]] = [[] for _ in graph.tasks]
    for e, event in enumerate(events, 1):
        for c in event.consumers:
            waits[c] = e
        for p in event.producers:
            signals[p].append(e)
    publications = []
    records = []
    for i in order:
        task = graph.tasks[i]
        e = waits.get(i, 0)
        records.append(
            (
                task.root_id,
                task.tile_id,
                e,
                len(events[e - 1].producers) if e else 0,
                len(publications),
                len(signals[i]),
            )
        )
        publications.extend(signals[i])
    return TileSchedule(tuple(order), events, tuple(records), tuple(publications))
