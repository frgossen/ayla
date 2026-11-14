"""Replace phase rendezvous with a persistent tile task stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Literal, cast

from ..base.ir import Block, Op, Value, clone_op, find_ops_of_type
from ..base.pipeline import Pass
from ..analysis.tile_dependency import (
    TileDependencyAnalyzer,
    TileDependencyGraph,
    TileTask,
    build_tile_dependency_graph,
)
from ..analysis.cross_loop_scheduler import (
    ReadinessEvent,
    TileSchedule,
    schedule_tiles,
    static_u32_expression,
    specialize_static_waits,
)
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    BarrierOp,
    BlockIdOp,
    CollectiveAxis,
    F32Type,
    BF16Type,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexConstantOp,
    IndexAddOp,
    IndexDivOp,
    IndexEqOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    IndexType,
    KernelOp,
    ModuleOp,
    MatmulOp,
    MoveOp,
    PredicateType,
    PTXTemplateOp,
    SmemTensorType,
    RmemTensorType,
    TileSyncOp,
    root_buffer,
    is_side_effect_free,
    memory_effects,
)


@dataclass
class _TileRoot:
    setup: list[Op]
    body: Block
    tile: Value
    count: int
    grid: int
    preamble: list[Op] = field(default_factory=list)
    n_fast: bool = False
    row_ranges: list[tuple[int, int]] | None = None
    worker_lanes: list[int] | None = None
    early_workers: tuple[int, ...] = ()
    row_batch: int = 1

    @property
    def uses_async_proxy(self) -> bool:
        for op in self.body.ops:
            if find_ops_of_type(op, MatmulOp):
                return True
            for move in find_ops_of_type(op, MoveOp):
                if (isinstance(move.src.type, GmemTensorType)
                    and isinstance(move.dst.type, SmemTensorType)) or (
                    isinstance(move.src.type, SmemTensorType)
                    and isinstance(move.dst.type, GmemTensorType)
                ):
                    return True
        return False


def _extract_root(
    ops: list[Op], blocks: int, shared_allocations: dict[Value, Value] | None = None
) -> _TileRoot:
    """Specialize the output wave to one tile, retaining its inner K pipeline."""
    if len(ops) == 4 and isinstance(ops[-1], IfOp) and ops[-1].else_body is None:
        guard = ops[-1]
        predicate = guard.condition.definition
        if isinstance(predicate, IndexLtOp):
            worker = predicate.operands[0].definition
            limit = predicate.operands[1].definition
            if (
                isinstance(worker, BlockIdOp)
                and worker.axis == CollectiveAxis.X
                and isinstance(limit, IndexConstantOp)
                and 0 < limit.value <= blocks
            ):
                return _extract_root(
                    list(guard.body.ops), limit.value, shared_allocations
                )
    tile = Value(None, IndexType())
    zero = IndexConstantOp.build(0)
    body: list[Op] = [zero]
    mapping: dict[Value, Value] = dict(shared_allocations or {})
    # A scheduled root is one fixed-grid output loop, or one direct grid tile.
    loops = [
        op
        for op in ops
        if isinstance(op, ForOp) and op.attrs.get("tile_output_loop") is True
    ]
    if len(loops) > 1:
        raise ValueError(
            "tile scheduling currently requires one output loop per graph phase"
        )
    # Fixed-successor lookahead carries operand buffers across output waves.
    # Retain both the physical lane and wave coordinate for its first/last
    # predicates; flattening every wave to zero would reload initial buffers
    # and issue a spurious prefetch after the final output.
    lookahead = any(str(nested.attrs.get("label", "")).startswith("output_prefetch_")
                    for op in ops for nested in find_ops_of_type(op, Op))
    lane = tile
    wave: Value | None = None
    if lookahead:
        stride = IndexConstantOp.build(blocks)
        physical_lane = IndexModOp.build(tile, stride.result())
        wave_id = IndexDivOp.build(tile, stride.result())
        wave_offset = IndexMulOp.build(wave_id.result(), stride.result())
        body.extend([stride, physical_lane, wave_id, wave_offset])
        lane = physical_lane.result()
        wave = wave_offset.result()
    count = blocks
    setup: list[Op] = []
    for op in ops:
        if isinstance(op, BarrierOp) and op.scope == "grid":
            raise ValueError(
                "tile scheduling requires library phases without internal grid barriers"
            )
        if isinstance(op, BlockIdOp):
            if op.axis != CollectiveAxis.X:
                raise ValueError(
                    "tile scheduling requires a one-dimensional fixed CTA grid"
                )
            mapping[op.value] = lane
            continue
        if isinstance(op, ForOp) and op.attrs.get("tile_output_loop") is True:
            # GEMM output loops use wave * grid + block; SwiGLU uses iv + block.
            # Setting the wave to zero and block to logical tile handles both.
            # A retiled SwiGLU tail starts at a nonzero column strip.
            # Metadata and assignment use a zero-based local tile index;
            # the library's flat coordinate retains the range origin.
            origin = IndexConstantOp.build(op.lb) if op.lb else zero
            if op.lb:
                body.append(origin)
            mapping[op.iv] = origin.result()
            if wave is not None:
                coordinate = IndexAddOp.build(origin.result(), wave)
                body.append(coordinate)
                mapping[op.iv] = coordinate.result()
            count = cast(
                int,
                op.attrs.get(
                    "tile_task_count",
                    op.ub if op.step == blocks else op.trip_count * blocks,
                ),
            ) - (op.lb if op.step == blocks else 0)
            body.extend(clone_op(inner, mapping) for inner in op.body.ops)
        elif isinstance(op, AllocOp):
            if op.result() not in mapping:
                setup.append(clone_op(op, mapping))
                if shared_allocations is not None:
                    shared_allocations[op.result()] = mapping[op.result()]
        else:
            body.append(clone_op(op, mapping))
    if any(
        barrier.scope == "grid"
        for op in body
        for barrier in find_ops_of_type(op, BarrierOp)
    ):
        raise ValueError(
            "tile scheduling requires library phases without internal grid barriers"
        )
    if any(find_ops_of_type(op, BlockIdOp) for op in body):
        raise ValueError("tile scheduling requires CTA IDs in the phase preamble")
    return _TileRoot(setup, Block(ops=body), tile, count, blocks,
                     row_batch=cast(int, loops[0].attrs.get("tile_rows_per_task", 1)) if loops else 1)


def _sparse_tile_guard(tile: Value, positions: list[int], wave: Value | None = None) -> PTXTemplateOp | None:
    """Test immutable sparse metadata without loading it on empty iterations."""
    positions = sorted(set(positions))
    operands, names = [tile], ["tile"]
    code = "mov.pred ${pred:result.needed.val}, 0;\n"
    if wave is not None:
        operands.append(wave)
        names.append("wave")
        code = "setp.eq.u32 ${pred:result.needed.val}, ${b32:operand.wave.val}, 0;\n"
    step = gcd(*(b - a for a, b in zip(positions, positions[1:]))) if len(positions) > 1 else 0
    # A residue/range guard may include holes: those still load the original
    # zero metadata. It must never exclude a configured nonzero entry.
    if len(positions) > 2 and (step > 1 or positions == list(range(positions[0], positions[-1] + 1))):
        code += f"setp.ge.u32 ${{pred:lo}}, ${{b32:operand.tile.val}}, {positions[0]};\n"
        code += f"setp.le.u32 ${{pred:hi}}, ${{b32:operand.tile.val}}, {positions[-1]};\n"
        code += "and.pred ${pred:match}, ${pred:lo}, ${pred:hi};\n"
        if step > 1:
            instruction = "and.b32" if step & (step - 1) == 0 else "rem.u32"
            divisor = step - 1 if instruction == "and.b32" else step
            code += f"{instruction} ${{b32:residue}}, ${{b32:operand.tile.val}}, {divisor};\n"
            code += f"setp.eq.u32 ${{pred:stride}}, ${{b32:residue}}, {positions[0] % step};\n"
            code += "and.pred ${pred:match}, ${pred:match}, ${pred:stride};\n"
        code += "or.pred ${pred:result.needed.val}, ${pred:result.needed.val}, ${pred:match};\n"
    elif len(positions) <= 32:
        for position in positions:
            code += f"setp.eq.u32 ${{pred:match}}, ${{b32:operand.tile.val}}, {position};\n"
            code += "or.pred ${pred:result.needed.val}, ${pred:result.needed.val}, ${pred:match};\n"
    else:
        return None
    return PTXTemplateOp.build(operands, [PredicateType()], code, names, ["needed"],
                               description="static sparse metadata guard")


def _row_range(count: int, blocks: int, worker: int) -> tuple[int, int]:
    """Balanced consecutive rows, including lanes beyond complete GEMM grids."""
    quotient, remainder = divmod(count, blocks)
    first = worker * quotient + min(worker, remainder)
    return first, first + quotient + (worker < remainder)


def _row_owner(tile: int, count: int, blocks: int) -> int:
    quotient, remainder = divmod(count, blocks)
    boundary = remainder * (quotient + 1)
    return (
        tile // (quotient + 1)
        if tile < boundary
        else remainder + (tile - boundary) // quotient
    )


def _root_row_ranges(root: _TileRoot, blocks: int) -> list[tuple[int, int]]:
    return root.row_ranges or [_row_range(root.count, blocks, worker) for worker in range(blocks)]


def _normalization_rows(rows: int, row_tile: int, producer_grid: int, blocks: int,
                        ready_rows: int | None = None) -> list[tuple[int, int]]:
    """Balance row work while giving idle lanes a ready prefix.

    Idle lanes get an ordinary CTA's row budget, with bounded extra work when
    it completes a downstream input tile. Assigning an entire producer M tile
    to them scales their critical path with the producer's M tile size.
    Keep all chunks inside one producer row group for one entry acquire.
    """
    idle = blocks - producer_grid
    groups = rows // row_tile
    assert 0 < idle <= row_tile and 1 < groups <= producer_grid + 1
    prefix = min(row_tile, -(-rows // blocks) * idle)
    if ready_rows is not None:
        # Extra idle-lane work must complete a downstream GEMM input tile,
        # and must stay within twice the balanced budget. A larger producer
        # M tile alone does not justify taking more normalization rows.
        if ready_rows <= min(row_tile, 2 * -(-rows // blocks) * idle):
            prefix = max(prefix, ready_rows)
    sizes = [row_tile - prefix, *([row_tile] * (groups - 1))]
    low, high = 1, max(sizes)
    while low < high:
        middle = (low + high) // 2
        if sum(-(-size // middle) for size in sizes) <= producer_grid:
            high = middle
        else:
            low = middle + 1
    allocations = [-(-size // low) for size in sizes]
    while sum(allocations) < producer_grid:
        group = max((g for g, size in enumerate(sizes) if size),
                    key=lambda g: sizes[g] / (allocations[g] + 1))
        allocations[group] += 1
    ranges = []
    for group, (size, count) in enumerate(zip(sizes, allocations, strict=True)):
        origin = group * row_tile + (prefix if group == 0 else 0)
        ranges.extend((origin + first, origin + stop)
                      for worker in range(count)
                      for first, stop in [_row_range(size, count, worker)])
    ranges.extend(_row_range(prefix, idle, worker) for worker in range(idle))
    return ranges


def _current_tile_body(root: _TileRoot) -> Block:
    """Analyze current computation independently of next-output lookahead."""
    mapping: dict[Value, Value] = {}
    body = Block(ops=[clone_op(op, mapping) for op in root.body.ops])

    def remove(block: Block) -> None:
        block.ops = [op for op in block.ops if not (
            isinstance(op, AsyncOp) and str(op.attrs.get("label", "")).startswith("output_prefetch_"))]
        for op in block.ops:
            for child in op.blocks:
                remove(child)

    remove(body)
    return body


def _assign_ready_gemms(roots: list[_TileRoot], tasks: tuple[TileTask, ...], blocks: int, threads: int) -> None:
    """Assign first ready consumer tiles to the CTAs that normalize them early."""
    for i, (producer, consumer) in enumerate(zip(roots, roots[1:])):
        if producer.row_ranges is None or consumer.count < consumer.grid or not any(
            find_ops_of_type(op, MatmulOp) for op in consumer.body.ops
        ):
            continue
        early_end = producer.row_ranges[-1][1]
        early_workers = [worker for worker, (first, stop) in enumerate(producer.row_ranges)
                         if first < early_end and stop <= early_end]
        written = {access.root for task in tasks if task.root_id == i for access in task.accesses if access.write}
        current = _current_tile_body(consumer)
        # Preserve the logical stride: changing it can change the row group on
        # every wave and defeat acquire reuse and fixed-successor lookahead.
        grid = consumer.grid
        ready = []
        analyzer = TileDependencyAnalyzer(threads=threads)
        for tile in range(min(grid, consumer.count)):
            reads = [access for access in analyzer.accesses(current, consumer.tile, tile)
                     if not access.write and access.root in written]
            if reads and all(isinstance(access.root.type, GmemTensorType)
                             and access.root.type.static_shape[0] == producer.count * producer.row_batch
                             and access.root.type.strides[0] is not None
                             and all(0 <= first < stop <= early_end * producer.row_batch * access.root.type.strides[0]
                                     * cast(BF16Type | F32Type, access.root.type.dtype).byte_size
                                     for first, stop in access.region.intervals)
                             for access in reads):
                ready.append(tile)
        if len(ready) < len(early_workers):
            continue
        if consumer.n_fast and ready == list(range(len(ready))) and i:
            # A long preceding GEMM leaves enough time for idle CTAs to
            # compute a complete output row, rather than three unrelated
            # column strips. This makes consecutive K/Q reductions ready.
            def tile_work(root_id: int) -> int:
                tile = next(t for t in tasks if t.root_id == root_id and t.tile_id == 0)
                outputs = [a for a in tile.accesses if a.write and isinstance(a.root.type, GmemTensorType)]
                if len({a.root for a in outputs}) != 1:
                    return 0
                output = cast(GmemTensorType, outputs[0].root.type)
                if len(output.static_shape) != 2:
                    return 0
                area = sum(stop - first for a in outputs for first, stop in a.region.intervals)
                area //= cast(BF16Type | F32Type, output.dtype).byte_size
                depths = [a.root.type.static_shape[1] for a in tile.accesses
                          if not a.write and isinstance(a.root.type, GmemTensorType)
                          and a.root not in {out.root for out in outputs}
                          and len(a.root.type.static_shape) == 2
                          and a.root.type.static_shape[0] == output.static_shape[0]]
                return area * max(depths, default=0)

            extra = max(0, -(-len(ready) // len(early_workers)) - -(-consumer.count // grid))
            previous_work, current_work = tile_work(i - 1), tile_work(i + 1)
            if (previous_work and current_work and 4 * extra * current_work <= 3 * previous_work
                and roots[i - 1].uses_async_proxy
                and roots[i - 1].count > roots[i - 1].grid
                and consumer.count - len(ready) >= blocks - len(early_workers)):
                other = [worker for worker in range(blocks) if worker not in early_workers]
                ranges = [(0, 0)] * blocks
                for rank, worker in enumerate(early_workers):
                    ranges[worker] = _row_range(len(ready), len(early_workers), rank)
                for rank, worker in enumerate(other):
                    first, stop = _row_range(consumer.count - len(ready), len(other), rank)
                    ranges[worker] = (len(ready) + first, len(ready) + stop)
                consumer.row_ranges = ranges
                consumer.early_workers = tuple(early_workers)
                continue
        # Favor lanes whose later waves finish the next scalar consumer's
        # needed columns on the early row groups (for example, K heads).
        if i + 2 < len(roots) and not roots[i + 2].uses_async_proxy:
            output_roots = {a.root for task in tasks if task.root_id == i + 1
                            for a in task.accesses if a.write}
            # When the early workers can collectively complete a whole output
            # row, consider subsequent scalar consumers too (K and Q heads).
            # Otherwise retain the immediate consumer's column priority.
            consumer_roots = {i + 2}
            if len(output_roots) == 1:
                from ..analysis.interval_set import IntervalSet
                [output] = output_roots
                typ = output.type
                if isinstance(typ, GmemTensorType) and len(typ.static_shape) == 2 and typ.strides[1] == 1:
                    row = IntervalSet.build([(0, typ.static_shape[1] * cast(BF16Type | F32Type, typ.dtype).byte_size)])
                    lanes_for_row = {task.tile_id % grid for task in tasks if task.root_id == i + 1
                                     and any(a.write and a.root is output and a.region.overlaps(row) for a in task.accesses)}
                    if lanes_for_row and lanes_for_row <= set(ready) and len(lanes_for_row) <= len(early_workers):
                        for following in range(i + 3, len(roots)):
                            if roots[following].uses_async_proxy:
                                break
                            consumer_roots.add(following)
            needed = []
            for task in tasks:
                if task.root_id not in consumer_roots:
                    continue
                reads = [a for a in task.accesses if not a.write and a.root in output_roots]
                if reads and all(isinstance(a.root.type, GmemTensorType)
                                 and a.root.type.strides[0] is not None
                                 and all(stop <= early_end * producer.row_batch * a.root.type.strides[0]
                                         * cast(BF16Type | F32Type, a.root.type.dtype).byte_size
                                         for _, stop in a.region.intervals) for a in reads):
                    needed.extend(reads)
            scores = {lane: 0 for lane in ready}
            for task in tasks:
                lane = task.tile_id % grid
                if task.root_id == i + 1 and lane in scores:
                    scores[lane] += any(a.write and a.root is read.root and a.region.overlaps(read.region)
                                        for a in task.accesses for read in needed)
            first_outputs = {lane: min((first for task in tasks if task.root_id == i + 1 and task.tile_id == lane
                                       for a in task.accesses if a.write for first, _ in a.region.intervals), default=0)
                             for lane in ready}
            # Complete one consumer row group before spreading its required
            # columns over several row groups that would each remain partial.
            ready.sort(key=lambda lane: (-scores[lane], first_outputs[lane], lane))
        # The sentinel lane has no work. Moving a ready lane to an early CTA
        # transfers its sentinel to the displaced CTA, retaining every tile.
        lanes = [*range(grid), *([grid] * (blocks - grid))]
        used = set()
        for worker in early_workers:
            lane = lanes[worker] if lanes[worker] in ready and lanes[worker] not in used else next(
                tile for tile in ready if tile not in used)
            other = lanes.index(lane)
            lanes[worker], lanes[other] = lanes[other], lanes[worker]
            used.add(lane)
        consumer.worker_lanes = lanes
        consumer.early_workers = tuple(early_workers)




def _assign_early_scalar_rows(roots: list[_TileRoot], tasks: tuple[TileTask, ...],
                             graph: TileDependencyGraph, blocks: int) -> None:
    """Assign a bounded ready scalar prefix to early GEMM CTAs."""
    for consumer_id, consumer in enumerate(roots):
        if consumer.row_ranges is not None or consumer.uses_async_proxy:
            continue
        producer_id = next((i for i in range(consumer_id - 1, -1, -1)
                            if roots[i].worker_lanes is not None or roots[i].uses_async_proxy), None)
        if producer_id is None:
            continue
        producer = roots[producer_id]
        if producer.worker_lanes is None and not producer.early_workers:
            continue
        early = list(producer.early_workers) if producer.row_ranges is not None else [
            worker for worker, lane in enumerate(producer.worker_lanes or [])
            if worker >= producer.grid and lane < producer.grid]
        if not early:
            continue
        if producer.row_ranges is not None:
            task_owners = {tile: worker for worker, (first, stop) in enumerate(producer.row_ranges)
                           for tile in range(first, stop)}
        else:
            lane_owners = {lane: worker for worker, lane in enumerate(producer.worker_lanes or [])
                           if lane < producer.grid}
            task_owners = {tile: lane_owners[tile % producer.grid] for tile in range(producer.count)}
        ready = []
        for task_id, task in enumerate(tasks):
            if task.root_id != consumer_id:
                continue
            deps = graph.predecessors[task_id]
            if deps and all(tasks[p].root_id == producer_id
                            and task_owners[tasks[p].tile_id] in early for p in deps):
                ready.append(task.tile_id)
        if not ready or ready != list(range(len(ready))) or len(ready) < len(early):
            continue
        budget = 2 * -(-consumer.count // blocks)
        ready = ready[:min(len(ready), len(early) * budget)]
        other = [worker for worker in range(blocks) if worker not in early]
        if consumer.count - len(ready) < len(other):
            continue
        ranges = [(0, 0)] * blocks
        for rank, worker in enumerate(early):
            ranges[worker] = _row_range(len(ready), len(early), rank)
        for rank, worker in enumerate(other):
            first, stop = _row_range(consumer.count - len(ready), len(other), rank)
            ranges[worker] = (len(ready) + first, len(ready) + stop)
        consumer.row_ranges = ranges


def _coalesce_row_entry_waits(plan: TileSchedule, roots: list[_TileRoot], owners: list[list[int]],
                            *, eligible_async: frozenset[int] = frozenset()) -> TileSchedule:
    """Acquire a CTA's remaining input frontier once before its root chunk.

    Async roots qualify only when their tasks are mutually independent. Never
    hoist an acquire past an earlier task in this root that must produce it.
    """
    chunks: dict[tuple[int, int], list[int]] = {}
    for position, record in enumerate(plan.records):
        if not roots[record[0]].uses_async_proxy or record[0] in eligible_async:
            chunks.setdefault((record[0], owners[record[0]][record[1]]), []).append(position)
    replacements, keys = {}, {}
    task_positions = {task: p for p, task in enumerate(plan.order)}
    for positions in chunks.values():
        root_id, tile = plan.records[positions[0]][:2]
        if owners[root_id][tile] in roots[root_id].early_workers:
            # These CTAs intentionally start with a ready input tile before
            # later input row groups finish. Hoisting their later acquires
            # would destroy the overlap established by the ownership plan.
            continue
        positive = [p for p in positions if plan.records[p][2]]
        if not positive or positive == [positions[0]]:
            continue
        key = tuple(sorted({producer for p in positive
                            for producer in plan.events[plan.records[p][2] - 1].producers}))
        # Hoisting must not wait on a producer at/after this chunk's entry.
        if any(task_positions[producer] >= positions[0] for producer in key):
            continue
        event = keys.setdefault(key, len(plan.events) + len(keys) + 1)
        for p in positions:
            replacements[p] = (event, len(key)) if p == positions[0] else (0, 0)
    if not keys:
        return plan
    extra: dict[int, list[int]] = {}
    positions_by_task = {task: p for p, task in enumerate(plan.order)}
    for producers, event in keys.items():
        for producer in producers:
            extra.setdefault(positions_by_task[producer], []).append(event)
    records, publications = [], []
    for p, record in enumerate(plan.records):
        arrivals = [*plan.publications[record[4]:record[4] + record[5]], *extra.get(p, [])]
        records.append((*record[:2], *replacements.get(p, record[2:4]), len(publications), len(arrivals)))
        publications.extend(arrivals)
    events = tuple(ReadinessEvent(producers, tuple(plan.order[p] for p, pair in replacements.items()
                                                   if pair[0] == event)) for producers, event in keys.items())
    return TileSchedule(plan.order, (*plan.events, *events), tuple(records), tuple(publications))


def _order_normalization_producers(roots: list[_TileRoot], tasks: tuple[TileTask, ...], blocks: int) -> tuple[TileTask, ...]:
    """Produce complete N rows before unrelated rows for normalization CTAs.

    Head reductions may read only part of a row. Completing the whole row
    also makes consecutive reductions of other head columns ready.
    """
    from ..analysis.interval_set import IntervalSet

    by_root = {i: [task for task in tasks if task.root_id == i] for i in range(len(roots))}
    remapped: dict[int, list[int]] = {}
    for i, (producer, consumer) in enumerate(zip(roots, roots[1:])):
        if (producer.grid >= blocks or producer.count <= producer.grid
            or len({root_buffer(mm.dst) for op in producer.body.ops for mm in find_ops_of_type(op, MatmulOp)}) != 1
            or any(find_ops_of_type(op, MatmulOp) for op in consumer.body.ops)):
            continue
        produced = by_root[i]
        outputs = {access.root for task in produced for access in task.accesses if access.write}
        if len(outputs) != 1 or not produced or not by_root[i + 1]:
            continue
        [hidden] = outputs
        if any(not access.write and access.root is hidden for task in produced for access in task.accesses):
            continue
        typ = cast(GmemTensorType, hidden.type)
        if len(typ.static_shape) != 2 or tuple(typ.strides) != (typ.static_shape[1], 1):
            continue
        rows, width = typ.static_shape
        whole_row = consumer.count * consumer.row_batch == rows
        dtype_bytes = cast(BF16Type | F32Type, typ.dtype).byte_size
        first = IntervalSet()
        for access in produced[0].accesses:
            if access.write:
                first = first.union(access.region)
        if not first.intervals:
            continue
        n_tile = (first.intervals[0][1] - first.intervals[0][0]) // dtype_bytes
        m_tile = sum(end - start for start, end in first.intervals) // (n_tile * dtype_bytes)
        if (not m_tile or not n_tile or rows % m_tile or width % n_tile
            or rows // m_tile < 2 or rows // m_tile - 1 > producer.grid
            or producer.grid > rows - m_tile
            or width // n_tile > producer.grid
            or blocks - producer.grid > m_tile // consumer.row_batch
            or m_tile % consumer.row_batch
            or producer.count != (rows // m_tile) * (width // n_tile)
            or [task.tile_id for task in produced] != list(range(producer.count))):
            continue
        first_read = IntervalSet()
        for access in by_root[i + 1][0].accesses:
            if not access.write and access.root is hidden:
                first_read = first_read.union(access.region)
        if (not first_read.intervals or (whole_row and
                first_read != IntervalSet.build([(0, consumer.row_batch * width * dtype_bytes)]))):
            continue
        m_tiles, n_tiles = rows // m_tile, width // n_tile
        rectangular = True
        for task in produced:
            actual = IntervalSet()
            for access in task.accesses:
                if access.write:
                    actual = actual.union(access.region)
            m, n = task.tile_id % m_tiles, task.tile_id // m_tiles
            expected = IntervalSet.build(((row * width + n * n_tile) * dtype_bytes,
                                         (row * width + (n + 1) * n_tile) * dtype_bytes)
                                        for row in range(m * m_tile, (m + 1) * m_tile))
            if actual != expected:
                rectangular = False
                break
        if not rectangular:
            continue
        mt, nt = IndexConstantOp.build(m_tiles), IndexConstantOp.build(n_tiles)
        n = IndexModOp.build(producer.tile, nt.result())
        m = IndexDivOp.build(producer.tile, nt.result())
        column = IndexMulOp.build(n.result(), mt.result())
        flat = IndexAddOp.build(column.result(), m.result())
        mapping = {producer.tile: flat.result()}
        producer.body.ops = [mt, nt, n, m, column, flat,
                             *(clone_op(op, mapping) for op in producer.body.ops)]
        producer.n_fast = True
        ready_rows = None
        if i + 2 < len(roots):
            written = {a.root for task in by_root[i + 1] for a in task.accesses if a.write}
            following = by_root[i + 2]
            if following and roots[i + 2].uses_async_proxy:
                for access in following[0].accesses:
                    source = access.root.type
                    if (not access.write and access.root in written
                        and isinstance(source, GmemTensorType)
                        and source.static_shape == (rows, width)
                        and tuple(source.strides) == (width, 1)
                        and len(access.region.intervals) == 1
                        and access.region.intervals[0][0] == 0):
                        row_bytes = width * cast(BF16Type | F32Type, source.dtype).byte_size
                        extent, remainder = divmod(access.region.intervals[0][1], row_bytes)
                        if not remainder and 0 < extent <= rows and extent % consumer.row_batch == 0:
                            ready_rows = extent // consumer.row_batch
        if whole_row:
            consumer.row_ranges = _normalization_rows(rows // consumer.row_batch, m_tile // consumer.row_batch,
                                                       producer.grid, blocks, ready_rows)
        remapped[i] = [(tile % n_tiles) * m_tiles + tile // n_tiles for tile in range(producer.count)]
    return tuple(TileTask(task.root_id, task.tile_id,
                          by_root[task.root_id][remapped[task.root_id][task.tile_id]].accesses)
                 if task.root_id in remapped else task for task in tasks)


class ScheduleTileGraph(Pass):
    """Use the library's CTA tiles as tasks on a bounded resident grid.

    Native static roots retain aligned GEMM lanes and consecutive row chunks.
    Table fallback workers consume strided positions in a topological stream;
    dynamic workers claim its next position atomically. Only resident CTAs may
    wait, so the cooperative launch occupancy check remains mandatory. No CTA
    can wait on an unclaimed earlier task once all workers are blocked: the
    earliest unfinished task's worker always has runnable work.
    """

    _coalesce_publications = True

    def __init__(
        self,
        mode: Literal["static", "dynamic"] = "static",
        *,
        tile_order: str = "source",
    ):
        super().__init__()
        if mode not in ("static", "dynamic"):
            raise ValueError("tile scheduler must be 'static' or 'dynamic'")
        self.mode = mode
        if tile_order not in ("source", "memory"):
            raise ValueError("tile_order must be 'source' or 'memory'")
        self.tile_order = tile_order

    def run(self, module: ModuleOp) -> ModuleOp:
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        blocks = cast(int, kernel.grid_dim[0])
        threads = cast(int, kernel.block_dim[0])
        if (
            type(blocks) is not int
            or blocks < 1
            or type(threads) is not int
            or threads < 128
            or threads % 128
        ):
            raise ValueError("tile scheduling requires a specialized resident CTA grid")
        if kernel.grid_dim[1:] != (1, 1) or any(
            d not in (None, 1) for d in kernel.cluster_dim
        ):
            raise ValueError(
                "tile scheduling requires a fixed, unclustered one-dimensional grid"
            )
        if kernel.attrs.get("cooperative") is not True:
            raise ValueError("tile scheduling requires a resident cooperative launch")
        phases: list[list[Op]] = [[]]
        for op in kernel.body.ops:
            if isinstance(op, BarrierOp) and op.scope == "grid":
                if phases[-1]:
                    phases.append([])
            else:
                phases[-1].append(op)
        phases = [p for p in phases if p]
        roots = []
        for phase in phases:
            loops = [
                op
                for op in phase
                if isinstance(op, ForOp) and op.attrs.get("tile_output_loop") is True
            ]
            if len(loops) <= 1:
                roots.append(_extract_root(phase, blocks))
                continue
            prefix = [op for op in phase if op not in loops]
            if any(
                not is_side_effect_free(op) and not isinstance(op, BarrierOp)
                for op in prefix
            ):
                raise ValueError("multiple tile loops require a pure phase preamble")
            allocations: dict[Value, Value] = {}
            for loop in loops:
                roots.append(_extract_root([*prefix, loop], blocks, allocations))
        analyzer = TileDependencyAnalyzer(threads=threads)
        tasks = tuple(
            TileTask(i, t, accesses)
            for i, root in enumerate(roots)
            for t in range(root.count)
            if (accesses := analyzer.accesses(root.body, root.tile, t,
                                             thread_group_size=min(128, threads // root.row_batch)))
        )
        if not tasks:
            raise ValueError("tile scheduling requires global memory tasks")
        # Row loops round their launch extent up to whole resident waves.
        # Analysis removes inactive tail iterations; use their dense logical
        # prefix, not the padded loop capacity, for static root eligibility.
        for i, root in enumerate(roots):
            tile_ids = [task.tile_id for task in tasks if task.root_id == i]
            if tile_ids and tile_ids == list(range(len(tile_ids))):
                root.count = len(tile_ids)
        if self.mode == "static" and self.tile_order == "source" and self._coalesce_publications:
            tasks = _order_normalization_producers(roots, tasks, blocks)
            _assign_ready_gemms(roots, tasks, blocks, threads)
        graph = build_tile_dependency_graph(tasks)
        if self.mode == "static" and self.tile_order == "source" and self._coalesce_publications:
            _assign_early_scalar_rows(roots, tasks, graph, blocks)
        plan = schedule_tiles(graph, tile_order=self.tile_order)
        native = (
            self.mode == "static"
            and self.tile_order == "source"
            and all(
                [(record[0], record[1]) for record in plan.records if record[0] == i]
                == [(i, tile) for tile in range(root.count)]
                for i, root in enumerate(roots)
            )
        )
        if native:
            written = {access.root for task in tasks for access in task.accesses if access.write}
            for root in roots:
                # Preserve scalar coefficients/column coordinates before the
                # row loop. Reads may move only when their global roots are
                # never written by this graph, and no logical-tile value feeds
                # the computation. Keep this preamble at its original phase.
                blocked = {root.tile}
                remaining = []
                for op in root.body.ops:
                    effects = [effect for nested in find_ops_of_type(op, Op)
                               for effect in memory_effects(nested) or []]
                    legal = op.attrs.get("tile_row_preamble") and not any(
                        operand in blocked or root_buffer(operand) in blocked
                        for nested in find_ops_of_type(op, Op) for operand in nested.operands
                    ) and all(
                        isinstance(effect.value.type, RmemTensorType)
                        or (effect.kind == "R" and isinstance(effect.value.type, GmemTensorType)
                            and not (effect.roots & written))
                        for effect in effects
                    ) and all(memory_effects(nested) is not None
                              for nested in find_ops_of_type(op, Op) if not nested.blocks)
                    if legal:
                        root.preamble.append(op)
                    else:
                        remaining.append(op)
                        blocked.update(op.results)
                        blocked.update(root_buffer(effect.value) for effect in effects if effect.kind != "R")
                root.body.ops = remaining
            kernel.attrs["tile_scalar_preamble_ops"] = [len(root.preamble) for root in roots]
            kernel.attrs["tile_scalar_preamble_global_reads"] = [
                sum(effect.kind == "R" and isinstance(effect.value.type, GmemTensorType)
                    for op in root.preamble for nested in find_ops_of_type(op, Op)
                    for effect in memory_effects(nested) or [])
                for root in roots
            ]
        independent = {
            i
            for i in range(len(roots))
            if not any(
                tasks[producer].root_id == i
                for task, deps in zip(tasks, graph.predecessors, strict=True)
                if task.root_id == i
                for producer in deps
            )
        }
        affinity = {
            i: max(stop - first for first, stop in _root_row_ranges(root, blocks))
            for i, root in enumerate(roots)
            if native
            and self._coalesce_publications
            and i in independent
            and (root.row_ranges is not None or not any(find_ops_of_type(op, MatmulOp) for op in root.body.ops))
        }
        row_owners = {}
        for i in affinity:
            row_owners[i] = [0] * roots[i].count
            for worker, (first, stop) in enumerate(_root_row_ranges(roots[i], blocks)):
                row_owners[i][first:stop] = [worker] * (stop - first)
        lane_owners = {i: {lane: worker for worker, lane in enumerate(root.worker_lanes) if lane < root.grid}
                       for i, root in enumerate(roots) if root.worker_lanes is not None}
        owners = [
            [
                row_owners[i][tile]
                if i in affinity
                else lane_owners[i][tile % root.grid] if i in lane_owners else tile % root.grid
                for tile in range(root.count)
            ]
            for i, root in enumerate(roots)
        ]
        workers = tuple(
            owners[record[0]][record[1]] if native else position % blocks
            for position, record in enumerate(plan.records)
        )
        original_waits = sum(bool(record[2]) for record in plan.records)
        if self.mode == "static":
            async_roots = {
                i for i, root in enumerate(roots)
                if root.uses_async_proxy
            }
            plan = specialize_static_waits(
                plan, blocks, workers=workers,
                async_tasks=frozenset(i for i, task in enumerate(tasks)
                                      if task.root_id in async_roots) if native else None,
            )
            if native:
                coalesced = _coalesce_row_entry_waits(plan, roots, owners,
                                                     eligible_async=frozenset(independent))
                if coalesced is not plan:
                    plan = specialize_static_waits(coalesced, blocks, workers=workers,
                                                  async_tasks=frozenset(i for i, task in enumerate(tasks)
                                                                        if task.root_id in async_roots))
        deferred: dict[str, list[tuple[int, int]]] = {}
        if native and self._coalesce_publications:
            # Acquire the inputs of speculative output lookahead before its
            # loads; scalar reductions already acquire complete rows at entry.
            plan, lookahead, lookahead_tables = self._split_lookahead_readiness(
                roots, tasks, graph, plan, threads, kernel.body.arguments[-1]
            )
            deferred.update(lookahead_tables)
            kernel.attrs["tile_lookahead_readiness"] = lookahead
        cohorts: dict[int, list[set[int]]] = {}
        frontier_cohorts: dict[int, list[set[int]]] = {}
        arrivals: list[set[tuple[int, int]]] = []
        if (
            native
            and self._coalesce_publications
            and all(
                [(record[0], record[1]) for record in plan.records if record[0] == i]
                == [(i, tile) for tile in range(root.count)]
                for i, root in enumerate(roots)
            )
        ):
            # One arrival per CTA/key, published at its exact last contributing
            # tile. A key must not wait for unrelated subsequent output tiles.
            eligible = independent
            cohorts = {root: [set() for _ in range(blocks)] for root in eligible}
            frontier_cohorts = {
                root: [set() for _ in range(roots[root].count)] for root in eligible
            }
            last = {}
            for position, record in enumerate(plan.records):
                last[record[0], workers[position]] = record[1]
            contributing = {}
            for position, record in enumerate(plan.records):
                for event in plan.publications[record[4] : record[4] + record[5]]:
                    key = (record[0], workers[position], event)
                    contributing[key] = record[1]
            arrivals = [set() for _ in plan.events]
            for position, record in enumerate(plan.records):
                worker = workers[position]
                for event in plan.publications[record[4] : record[4] + record[5]]:
                    if record[0] in cohorts:
                        final_tile = contributing[record[0], worker, event]
                        if final_tile < last[record[0], worker]:
                            frontier_cohorts[record[0]][final_tile].add(event)
                        else:
                            cohorts[record[0]][worker].add(event)
                        arrivals[event - 1].add((record[0], worker))
                    else:
                        arrivals[event - 1].add((-1, position))
            plan = TileSchedule(
                plan.order,
                plan.events,
                tuple(
                    (
                        *record[:3],
                        len(arrivals[record[2] - 1]) if record[2] else record[3],
                        *record[4:],
                    )
                    for record in plan.records
                ),
                plan.publications,
            )
            frontier_cohorts = {
                root: publishers
                for root, publishers in frontier_cohorts.items()
                if any(publishers)
            }
        sync = kernel.body.arguments[-1]
        if sync.name != "forward_sync":
            raise ValueError(
                "tile scheduling requires the forward scheduler's scratch argument"
            )
        sync.type = GmemTensorType.build([plan.scratch_words], F32Type())
        # Immutable schedule tables live in PTX global storage. The data are
        # read-only; global rather than constant memory avoids the 64KiB limit.
        tables = {
            "ayla_tile_tasks": [word for record in plan.records for word in record],
            "ayla_tile_publications": list(plan.publications) or [0],
        }
        kernel.attrs["ptx_global_u32_tables"] = tables
        for name, readiness in deferred.items():
            tables[name] = [
                word
                for event, expected in readiness
                for word in (event, len(arrivals[event - 1]) if event else expected)
            ]
        for root, workers in cohorts.items():
            entries = []
            for events in workers:
                entries.extend([len(tables["ayla_tile_publications"]), len(events)])
                tables["ayla_tile_publications"].extend(sorted(events))
            tables[f"ayla_root_{root}_publications"] = entries
        for root, publishers in frontier_cohorts.items():
            entries = []
            for events in publishers:
                entries.extend([len(tables["ayla_tile_publications"]), len(events)])
                tables["ayla_tile_publications"].extend(sorted(events))
            tables[f"ayla_root_{root}_frontier_publications"] = entries
        kernel.attrs["tile_coalesced_roots"] = sorted(cohorts)
        kernel.attrs["tile_frontier_publication_roots"] = sorted(frontier_cohorts)
        kernel.attrs["tile_publication_scope"] = "last_contributing_tile"
        kernel.attrs["tile_row_affinity"] = affinity
        kernel.attrs["tile_root_grids"] = [root.grid for root in roots]
        kernel.attrs["tile_row_ranges"] = {
            i: _root_row_ranges(roots[i], blocks)
            for i in affinity
        }
        kernel.attrs["tile_n_fast_roots"] = [i for i, root in enumerate(roots) if root.n_fast]
        kernel.attrs["tile_rows_per_task"] = [root.row_batch for root in roots]
        kernel.attrs["tile_worker_lanes"] = {i: root.worker_lanes for i, root in enumerate(roots)
                                             if root.worker_lanes is not None}
        for i, root in enumerate(roots):
            if root.worker_lanes is not None:
                tables[f"ayla_root_{i}_worker_lanes"] = root.worker_lanes
            if root.row_ranges is not None:
                tables[f"ayla_root_{i}_row_bounds"] = [value for first, stop in root.row_ranges
                                                        for value in (first, stop - first)]
        kernel.attrs["tile_actual_publication_arrivals"] = sum(
            record[5] for record in plan.records if record[0] not in cohorts
        ) + sum(
            len(events)
            for publishers in (*cohorts.values(), *frontier_cohorts.values())
            for events in publishers
        )
        kernel.attrs["tile_scheduler"] = self.mode
        kernel.attrs["tile_order"] = self.tile_order
        kernel.attrs["tile_task_count"] = len(tasks)
        kernel.attrs["tile_readiness_events"] = len(plan.events)
        kernel.attrs["tile_readiness_manifest"] = [
            dict(
                event=event_id,
                producers=[
                    (tasks[task].root_id, tasks[task].tile_id)
                    for task in event.producers
                ],
                expected=len(arrivals[event_id - 1])
                if arrivals
                else len(event.producers),
            )
            for event_id, event in enumerate(plan.events, 1)
        ]
        kernel.attrs["tile_wait_count"] = sum(
            bool(record[2]) for record in plan.records
        )
        kernel.attrs["tile_waits_by_root"] = {
            str(root): sum(
                bool(record[2]) for record in plan.records if record[0] == root
            )
            for root in range(len(roots))
        }
        kernel.attrs["tile_local_proxy_fence_count"] = sum(
            record[2] == 0 and record[3] != 0 for record in plan.records
        )
        kernel.attrs["tile_redundant_waits_removed"] = original_waits - cast(
            int, kernel.attrs["tile_wait_count"]
        )
        if (
            self.mode == "static"
            and self.tile_order == "source"
            and all(
                [(record[0], record[1]) for record in plan.records if record[0] == i]
                == [(i, tile) for tile in range(root.count)]
                for i, root in enumerate(roots)
            )
        ):
            self._static_root_loops(kernel, roots, plan, blocks)
            return module
        setup = [op for root in roots for op in root.setup]
        broadcast = AllocOp.build_with_fresh_id(SmemTensorType.build([1], F32Type()))
        setup.append(broadcast)
        worker = BlockIdOp.build(CollectiveAxis.X)
        setup.append(worker)
        loop = ForOp.build(
            0, len(tasks) if self.mode == "dynamic" else -(-len(tasks) // blocks)
        )
        if self.mode == "dynamic":
            claim_template = """
                bar.sync 0;
                mov.u32 ${b32:tid}, %tid.x;
                setp.ne.u32 ${pred:other}, ${b32:tid}, 0;
                @${pred:other} bra ${label:broadcast};
                atom.relaxed.gpu.global.add.u32 ${b32:claimed}, [${b64:operand.sync.ptr}], 1;
                st.shared.u32 [${b32:operand.broadcast.ptr}], ${b32:claimed};
                ${label:broadcast}:
                bar.sync 0;
                ld.shared.u32 ${b32:result.position.val}, [${b32:operand.broadcast.ptr}];
            """
        else:
            claim_template = f"mad.lo.u32 ${{b32:result.position.val}}, ${{b32:operand.wave.val}}, {blocks}, ${{b32:operand.worker.val}};"
        claim = PTXTemplateOp.build(
            [sync, broadcast.result(), loop.iv, worker.value],
            [IndexType()],
            claim_template,
            operand_names=["sync", "broadcast", "wave", "worker"],
            result_names=["position"],
            description="claim tile task",
        )
        # Uniform early return is safe after the preceding tile's full drain.
        # It lets dynamic workers terminate without O(tasks * CTAs) empty polls.
        publication_fields = (
            """
            ld.global.u32 ${b32:result.offset.val}, [${b64:record}+16];
            ld.global.u32 ${b32:result.length.val}, [${b64:record}+20];
        """
            if self.mode == "dynamic"
            else ""
        )
        load = PTXTemplateOp.build(
            [claim.result()],
            [IndexType() for _ in range(6 if self.mode == "dynamic" else 4)],
            f"""
            setp.ge.u32 ${{pred:done}}, ${{b32:operand.position.val}}, {len(tasks)};
            @${{pred:done}} ret;
            mov.u64 ${{b64:table}}, ayla_tile_tasks;
            mul.wide.u32 ${{b64:offset}}, ${{b32:operand.position.val}}, 24;
            add.u64 ${{b64:record}}, ${{b64:table}}, ${{b64:offset}};
            ld.global.u32 ${{b32:result.root.val}}, [${{b64:record}}];
            ld.global.u32 ${{b32:result.tile.val}}, [${{b64:record}}+4];
            ld.global.u32 ${{b32:result.event.val}}, [${{b64:record}}+8];
            ld.global.u32 ${{b32:result.expected.val}}, [${{b64:record}}+12];
        """
            + publication_fields,
            operand_names=["position"],
            result_names=["root", "tile", "event", "expected"]
            + (["offset", "length"] if self.mode == "dynamic" else []),
            description="load tile task",
        )
        wait = TileSyncOp.build(sync, load.results[2], load.results[3], action="wait")
        body: list[Op] = [claim, load, wait]
        dispatch = Block()
        for i, root in enumerate(roots):
            index = IndexConstantOp.build(i)
            condition = IndexEqOp.build(load.results[0], index.result())
            mapping = {root.tile: load.results[1]}
            tile_body = [clone_op(op, mapping) for op in root.body.ops]
            if self.mode == "static":
                # Keeping publication offset/length live across a large GEMM
                # competes with its accumulators. Recompute the static record
                # position after work, using the already-live wave IV.
                publication = PTXTemplateOp.build(
                    [loop.iv],
                    [IndexType(), IndexType()],
                    f"""
                    mov.u32 ${{b32:worker}}, %ctaid.x;
                    mad.lo.u32 ${{b32:position}}, ${{b32:operand.wave.val}}, {blocks}, ${{b32:worker}};
                    mov.u64 ${{b64:table}}, ayla_tile_tasks;
                    mul.wide.u32 ${{b64:offset}}, ${{b32:position}}, 24;
                    add.u64 ${{b64:record}}, ${{b64:table}}, ${{b64:offset}};
                    ld.global.u32 ${{b32:result.offset.val}}, [${{b64:record}}+16];
                    ld.global.u32 ${{b32:result.length.val}}, [${{b64:record}}+20];
                    """,
                    operand_names=["wave"],
                    result_names=["offset", "length"],
                    description="load tile publications",
                )
                tile_body.append(publication)
                offset, length = publication.results
            else:
                offset, length = load.results[4:]
            tile_body.append(TileSyncOp.build(sync, offset, length, action="publish"))
            alternative = Block() if i + 1 < len(roots) else None
            branch = IfOp.build(condition.result(), Block(ops=tile_body), alternative)
            branch.attrs["tile_root"] = i
            dispatch.ops.extend(
                [
                    index,
                    condition,
                    branch,
                ]
            )
            if i == 0:
                body.extend(dispatch.ops)
            if alternative is not None:
                dispatch = alternative
        loop.body.ops = body
        kernel.body.ops = [*setup, loop]
        return module


    @staticmethod
    def _split_lookahead_readiness(
        roots: list[_TileRoot], tasks: tuple[TileTask, ...], graph: TileDependencyGraph,
        plan: TileSchedule, threads: int, sync: Value,
    ) -> tuple[TileSchedule, list[dict[str, int]], dict[str, list[tuple[int, int]]]]:
        """Acquire current inputs at entry, next-output inputs at their first load.

        Entry and guarded prefetch acquires cover the original full producer
        frontier before completion, preserving subsequent dependency elision.
        """
        info, tables = [], {}
        analyzer = TileDependencyAnalyzer(threads=threads)
        for root_id, root in enumerate(roots):
            if root_id == 0 or roots[root_id - 1].row_ranges is None:
                continue
            future = [op for top in root.body.ops for op in find_ops_of_type(top, AsyncOp)
                      if str(op.attrs.get("label", "")).startswith("output_prefetch_")]
            if not future:
                continue
            written = {access.root for task in tasks if task.root_id == root_id - 1
                       for access in task.accesses if access.write}
            sites = [op for op in future if any(root_buffer(move.src) in written
                                               for move in find_ops_of_type(op, MoveOp))]
            if not sites or len({id(op.parent_block) for op in sites}) != 1:
                continue
            active = [position for position, record in enumerate(plan.records)
                      if record[0] == root_id and record[2]]
            current = _current_tile_body(root)
            prefixes = {}
            for position in active:
                task_id = plan.order[position]
                deps = graph.predecessors[task_id]
                if any(tasks[p].root_id != root_id - 1 for p in deps):
                    break
                reads = [a for a in analyzer.accesses(current, root.tile, tasks[task_id].tile_id)
                         if not a.write and a.root in written]
                prefix = tuple(sorted(p for p in deps if any(
                    access.write and access.root is read.root and access.region.intersection(read.region).intervals
                    for access in tasks[p].accesses for read in reads)))
                if not prefix:
                    break
                prefixes[position] = prefix
            if len(prefixes) != len(active) or not prefixes:
                continue
            # A prior deferred acquire also proves the next tile's current
            # inputs. Do not re-acquire those rows at its entry.
            lane_owners = {lane: worker for worker, lane in enumerate(root.worker_lanes or range(root.grid))
                           if lane < root.grid}
            acquired: dict[int, set[int]] = {}
            entries = {}
            for position, prefix in prefixes.items():
                task = tasks[plan.order[position]]
                worker = lane_owners[task.tile_id % root.grid]
                seen = acquired.setdefault(worker, set())
                if not set(prefix).issubset(seen):
                    entries[position] = prefix
                seen.update(graph.predecessors[plan.order[position]])
            keys = list(dict.fromkeys(entries.values()))
            event_ids = {key: len(plan.events) + index + 1 for index, key in enumerate(keys)}
            extra = {}
            positions = {task: position for position, task in enumerate(plan.order)}
            for key, event in event_ids.items():
                for task in key:
                    extra.setdefault(positions[task], []).append(event)
            readiness = [(record[2], record[3]) if position in prefixes else (0, 0)
                         for position, record in enumerate(plan.records) if record[0] == root_id]
            records, publications = [], []
            for position, record in enumerate(plan.records):
                arrivals = [*plan.publications[record[4]:record[4] + record[5]], *extra.get(position, [])]
                event, expected = record[2:4]
                if position in prefixes:
                    key = entries.get(position)
                    event, expected = (event_ids[key], len(key)) if key is not None else (0, 0)
                records.append((*record[:2], event, expected, len(publications), len(arrivals)))
                publications.extend(arrivals)
            plan = TileSchedule(plan.order, (*plan.events, *(ReadinessEvent(key, ()) for key in keys)),
                                tuple(records), tuple(publications))
            name = f"ayla_root_{root_id}_lookahead_readiness"
            tables[name] = readiness
            load = PTXTemplateOp.build(
                [root.tile], [IndexType(), IndexType()],
                f"mov.u64 ${{b64:table}}, {name};\n"
                "mul.wide.u32 ${b64:offset}, ${b32:operand.tile.val}, 8;\n"
                "add.u64 ${b64:record}, ${b64:table}, ${b64:offset};\n"
                "ld.global.u32 ${b32:result.event.val}, [${b64:record}];\n"
                "ld.global.u32 ${b32:result.expected.val}, [${b64:record}+4];",
                ["tile"], ["event", "expected"], description="load deferred lookahead readiness",
            )
            wait = TileSyncOp.build(sync, *load.results, action="wait")
            wait.attrs.update(readiness_root=root_id, async_proxy=True)
            checkpoint: list[Op] = [load, wait]
            sparse = _sparse_tile_guard(root.tile, [tile for tile, entry in enumerate(readiness) if entry[0]])
            if sparse is not None:
                checkpoint = [sparse, IfOp.build(sparse.result(), Block(ops=checkpoint))]
            block = sites[0].parent_block
            assert block is not None
            at = block.ops.index(sites[0])
            block.ops[at:at] = checkpoint
            info.append(dict(root=root_id, entry_keys=len(keys), entry_tiles=len(entries), deferred_tiles=len(prefixes)))
        return plan, info, tables


    @staticmethod
    def _static_root_loops(
        kernel: KernelOp, roots: list[_TileRoot], plan: TileSchedule, blocks: int
    ) -> None:
        """Emit the statically assigned stream as root loops, without a dispatcher.

        Each ordinary root begins at a worker-aligned ownership base, as in
        Helion. The packed record base is only an address into metadata tables.
        Roots can overlap because there is no inter-CTA rendezvous between them.
        """
        kernel.attrs["tile_static_lowering"] = "root_loops"
        kernel.attrs["tile_static_readiness_table_roots"] = 0
        kernel.attrs["tile_static_publication_table_roots"] = 0
        sync = kernel.body.arguments[-1]
        worker = BlockIdOp.build(CollectiveAxis.X)
        body: list[Op] = [op for root in roots for op in root.setup]
        body.append(worker)

        base = 0
        for i, root in enumerate(roots):
            async_proxy = root.uses_async_proxy
            body.extend(root.preamble)
            logical_worker = worker.value
            if root.worker_lanes is not None:
                lane = PTXTemplateOp.build(
                    [worker.value], [IndexType()],
                    f"mov.u64 ${{b64:table}}, ayla_root_{i}_worker_lanes;\n"
                    "mul.wide.u32 ${b64:offset}, ${b32:operand.worker.val}, 4;\n"
                    "add.u64 ${b64:record}, ${b64:table}, ${b64:offset};\n"
                    "ld.global.u32 ${b32:result.lane.val}, [${b64:record}];",
                    ["worker"], ["lane"], description="load static worker lane",
                )
                body.append(lane)
                logical_worker = lane.result()
            records = plan.records[base : base + root.count]
            assert all(record[:2] == (i, tile) for tile, record in enumerate(records))
            grain = cast(dict[int, int], kernel.attrs["tile_row_affinity"]).get(i)
            row_bounds = None
            if root.row_ranges is not None:
                row_bounds = PTXTemplateOp.build(
                    [worker.value], [IndexType(), IndexType()],
                    f"mov.u64 ${{b64:table}}, ayla_root_{i}_row_bounds;\n"
                    "mul.wide.u32 ${b64:offset}, ${b32:operand.worker.val}, 8;\n"
                    "add.u64 ${b64:record}, ${b64:table}, ${b64:offset};\n"
                    "ld.global.u32 ${b32:result.first.val}, [${b64:record}];\n"
                    "ld.global.u32 ${b32:result.length.val}, [${b64:record}+4];",
                    ["worker"], ["first", "length"], description="load static row bounds",
                )
                body.append(row_bounds)
            loop = ForOp.build(
                0, grain if grain is not None else -(-root.count // root.grid)
            )
            tile = PTXTemplateOp.build(
                [logical_worker, loop.iv],
                [IndexType()],
                f"""
                mad.lo.u32 ${{b32:result.tile.val}}, ${{b32:operand.wave.val}}, {root.grid}, ${{b32:operand.worker.val}};
                """,
                operand_names=["worker", "wave"],
                result_names=["tile"],
                description="static root tile",
            )
            if grain is not None:
                quotient, remainder = divmod(root.count, blocks)
                tile.attrs["template"] = f"""
                    min.u32 ${{b32:extra}}, ${{b32:operand.worker.val}}, {remainder};
                    mad.lo.u32 ${{b32:result.tile.val}}, ${{b32:operand.worker.val}}, {quotient}, ${{b32:extra}};
                    add.u32 ${{b32:result.tile.val}}, ${{b32:result.tile.val}}, ${{b32:operand.wave.val}};
                """
                if row_bounds is not None:
                    tile.operands = [row_bounds.results[0], loop.iv]
                    tile.attrs["operand_names"] = ["first", "wave"]
                    tile.attrs["template"] = "add.u32 ${b32:result.tile.val}, ${b32:operand.first.val}, ${b32:operand.wave.val};"
            limit = IndexConstantOp.build(root.count)
            valid = IndexLtOp.build(tile.result(), limit.result())
            if grain is not None:
                quotient, remainder = divmod(root.count, blocks)
                valid = PTXTemplateOp.build(
                    [worker.value, loop.iv],
                    [PredicateType()],
                    f"setp.lt.u32 ${{pred:base}}, ${{b32:operand.wave.val}}, {quotient};\n"
                    f"setp.lt.u32 ${{pred:extra}}, ${{b32:operand.worker.val}}, {remainder};\n"
                    "or.pred ${pred:result.valid.val}, ${pred:base}, ${pred:extra};",
                    operand_names=["worker", "wave"],
                    result_names=["valid"],
                    description="balanced row iteration valid",
                )
                if row_bounds is not None:
                    valid = IndexLtOp.build(loop.iv, row_bounds.results[1])
            elif root.grid < blocks:
                valid = PTXTemplateOp.build(
                    [tile.result(), logical_worker],
                    [PredicateType()],
                    f"setp.lt.u32 ${{pred:tile_valid}}, ${{b32:operand.tile.val}}, {root.count};\n"
                    f"setp.lt.u32 ${{pred:worker_valid}}, ${{b32:operand.worker.val}}, {root.grid};\n"
                    "and.pred ${pred:result.valid.val}, ${pred:tile_valid}, ${pred:worker_valid};",
                    operand_names=["tile", "worker"],
                    result_names=["valid"],
                    description="static root active worker and tile",
                )
            mapping = {root.tile: tile.result()}
            work: list[Op] = []
            if any(record[2] or record[3] for record in records):
                readiness = PTXTemplateOp.build(
                    [tile.result()],
                    [IndexType(), IndexType()],
                    f"""
                    add.u32 ${{b32:position}}, ${{b32:operand.tile.val}}, {base};
                    mul.wide.u32 ${{b64:offset}}, ${{b32:position}}, 24;
                    mov.u64 ${{b64:table}}, ayla_tile_tasks;
                    add.u64 ${{b64:record}}, ${{b64:table}}, ${{b64:offset}};
                    ld.global.u32 ${{b32:result.event.val}}, [${{b64:record}}+8];
                    ld.global.u32 ${{b32:result.expected.val}}, [${{b64:record}}+12];
                    """,
                    operand_names=["tile"],
                    result_names=["event", "expected"],
                    description="load static readiness",
                )
                active = [bool(record[2] or record[3]) for record in records]
                prefix = next(
                    (index for index, needed in enumerate(active) if not needed),
                    len(records),
                )
                if any(active[prefix:]):
                    prefix = len(records)
                expressions = [
                    static_u32_expression(
                        tuple(record[field] for record in records[:prefix])
                    )
                    for field in (2, 3)
                ]
                if all(expression is not None for expression in expressions):
                    from ..tile_codegen import static_u32_ptx

                    computed = ""
                    if prefix < len(records):
                        computed += f"setp.ge.u32 ${{pred:inactive}}, ${{b32:operand.tile.val}}, {prefix};\n@${{pred:inactive}} bra ${{label:skip_acquire}};\n"
                    for name, expression in zip(
                        ("event", "expected"), expressions, strict=True
                    ):
                        assert expression is not None
                        computed += static_u32_ptx(
                            expression,
                            "${b32:operand.tile.val}",
                            f"${{b32:result.{name}.val}}",
                            f"readiness_{name}",
                        )
                    if prefix < len(records):
                        computed += "bra ${label:ready_key};\n${label:skip_acquire}:\nmov.u32 ${b32:result.event.val}, 0;\nmov.u32 ${b32:result.expected.val}, 0;\n${label:ready_key}:\n"
                    readiness.attrs["template"] = computed
                    readiness.attrs["description"] = "compute static readiness"
                else:
                    kernel.attrs["tile_static_readiness_table_roots"] = (
                        cast(int, kernel.attrs["tile_static_readiness_table_roots"]) + 1
                    )
                event = readiness.results[0]
                if not any(record[2] for record in records):
                    zero_event = IndexConstantOp.build(0)
                    work.append(zero_event)
                    event = zero_event.result()
                work.extend(
                    [
                        readiness,
                        TileSyncOp.build(
                            sync,
                            event,
                            readiness.results[1],
                            action="wait",
                        ),
                    ]
                )
            for op in work:
                if isinstance(op, TileSyncOp):
                    op.attrs["readiness_root"] = i
                    op.attrs["async_proxy"] = async_proxy
            entry_tiles = ({first for first, stop in _root_row_ranges(root, blocks) if first < stop}
                           if grain is not None else set(range(min(root.grid, root.count))))
            if (
                work
                and all(
                    not (record[2] or record[3])
                    or tile_id
                    in entry_tiles
                    for tile_id, record in enumerate(records)
                )
            ):
                entry_ops: list[Op] = []
                first_tile = logical_worker
                if grain is not None:
                    if row_bounds is not None:
                        first_tile = row_bounds.results[0]
                    else:
                        quotient, remainder = divmod(root.count, blocks)
                        first = PTXTemplateOp.build(
                            [worker.value], [IndexType()],
                            f"min.u32 ${{b32:extra}}, ${{b32:operand.worker.val}}, {remainder};\n"
                            f"mad.lo.u32 ${{b32:result.tile.val}}, ${{b32:operand.worker.val}}, {quotient}, ${{b32:extra}};",
                            ["worker"], ["tile"], description="static first balanced row",
                        )
                        entry_ops.append(first)
                        first_tile = first.result()
                first_limit = IndexConstantOp.build(root.count if grain is not None else min(root.count, root.grid))
                first_valid = IndexLtOp.build(first_tile, first_limit.result())
                remap = {tile.result(): first_tile}
                body.extend(
                    [
                        *entry_ops,
                        first_limit,
                        first_valid,
                        IfOp.build(
                            first_valid.result(),
                            Block(ops=[clone_op(op, remap) for op in work]),
                        ),
                    ]
                )
                work = []
            if work:
                first_tiles = {
                    first for first, _ in _root_row_ranges(root, blocks)
                }
                active_tiles = [tile_id for tile_id, record in enumerate(records) if record[2] or record[3]]
                needed = _sparse_tile_guard(
                    tile.result(), [tile_id for tile_id in active_tiles if tile_id not in first_tiles]
                    if grain is not None else active_tiles,
                    loop.iv if grain is not None else None,
                )
                if needed is None and active_tiles[-1] + 1 < root.count:
                    needed = PTXTemplateOp.build(
                        [tile.result()], [PredicateType()],
                        f"setp.le.u32 ${{pred:result.needed.val}}, ${{b32:operand.tile.val}}, {active_tiles[-1]};",
                        ["tile"], ["needed"], description="static last acquire guard",
                    )
                if needed is not None:
                    readiness = next(op for op in work if isinstance(op, PTXTemplateOp))
                    if len(needed.operands) == 2:
                        readiness.operands.append(loop.iv)
                        readiness.attrs["operand_names"] = ["tile", "wave"]
                    guard_code = needed.template.replace(
                        "${pred:result.needed.val}", "${pred:need_metadata}"
                    )
                    readiness.attrs["template"] = (
                        guard_code + "@!${pred:need_metadata} bra ${label:no_metadata};\n"
                        + readiness.template + "\nbra ${label:metadata_done};\n"
                        "${label:no_metadata}:\n"
                        "mov.u32 ${b32:result.event.val}, 0;\n"
                        "mov.u32 ${b32:result.expected.val}, 0;\n"
                        "${label:metadata_done}:\n"
                    )
            work.extend(clone_op(op, mapping) for op in root.body.ops)
            if i in cast(list[int], kernel.attrs.get("tile_coalesced_roots", [])):
                branch = IfOp.build(valid.result(), Block(ops=work))
                branch.attrs.update(tile_root=i, tile_completion_scope="cta_root")
                loop.body.ops = [tile, limit, valid, branch]
                if i in cast(
                    list[int], kernel.attrs.get("tile_frontier_publication_roots", [])
                ):
                    # Release each key at its exact last contributing tile;
                    # zero-length frontiers do not drain stores or join the CTA.
                    prefix = PTXTemplateOp.build(
                        [tile.result()],
                        [IndexType(), IndexType()],
                        f"""
                        mov.u64 ${{b64:table}}, ayla_root_{i}_frontier_publications;
                        mul.wide.u32 ${{b64:offset}}, ${{b32:operand.tile.val}}, 8;
                        add.u64 ${{b64:record}}, ${{b64:table}}, ${{b64:offset}};
                        ld.global.u32 ${{b32:result.offset.val}}, [${{b64:record}}];
                        ld.global.u32 ${{b32:result.length.val}}, [${{b64:record}}+4];
                        """,
                        operand_names=["tile"],
                        result_names=["offset", "length"],
                        description="load frontier publications",
                    )
                    zero = IndexConstantOp.build(0)
                    needed = IndexLtOp.build(zero.result(), prefix.results[1])
                    publish_prefix = TileSyncOp.build(
                        sync, *prefix.results, action="publish"
                    )
                    publish_prefix.attrs["join_after"] = False
                    publish_prefix.attrs["async_proxy"] = async_proxy
                    entries = cast(dict[str, list[int]], kernel.attrs["ptx_global_u32_tables"])[
                        f"ayla_root_{i}_frontier_publications"
                    ]
                    sparse = _sparse_tile_guard(
                        tile.result(), [tile_id for tile_id in range(root.count)
                                        if entries[2 * tile_id + 1]],
                    )
                    publication_work = [prefix, zero, needed, IfOp.build(
                        needed.result(), Block(ops=[publish_prefix])
                    )]
                    if sparse is not None:
                        publication_work = [sparse, IfOp.build(
                            sparse.result(), Block(ops=publication_work)
                        )]
                    loop.body.ops.extend(
                        [
                            IfOp.build(
                                valid.result(),
                                Block(
                                    ops=publication_work
                                ),
                            ),
                        ]
                    )
                body.append(loop)
                publications = PTXTemplateOp.build(
                    [worker.value],
                    [IndexType(), IndexType()],
                    f"""
                    mov.u64 ${{b64:table}}, ayla_root_{i}_publications;
                    mul.wide.u32 ${{b64:offset}}, ${{b32:operand.worker.val}}, 8;
                    add.u64 ${{b64:record}}, ${{b64:table}}, ${{b64:offset}};
                    ld.global.u32 ${{b32:result.offset.val}}, [${{b64:record}}];
                    ld.global.u32 ${{b32:result.length.val}}, [${{b64:record}}+4];
                    """,
                    operand_names=["worker"],
                    result_names=["offset", "length"],
                    description="load root publications",
                )
                if any(record[5] for record in records):
                    body.append(publications)
                    offset, length = publications.results
                else:
                    zero = IndexConstantOp.build(0)
                    body.append(zero)
                    offset = length = zero.result()
                publish = TileSyncOp.build(sync, offset, length, action="publish")
                publish.attrs["join_after"] = False
                publish.attrs["async_proxy"] = async_proxy
                # Lanes outside this root's grid issued no work or stores.
                # Every preceding root completed its own publication already.
                active_workers = min(root.count, blocks if grain is not None else root.grid)
                if active_workers < blocks and root.worker_lanes is None:
                    publish.attrs["active_workers"] = active_workers
                if root.worker_lanes is not None and root.grid < blocks:
                    # Keep the logical task boundary visible to sync analysis;
                    # lowering skips the CTAs proven to have issued no work.
                    publish.attrs["inactive_workers"] = tuple(
                        worker for worker, lane in enumerate(root.worker_lanes) if lane >= root.grid)
                body.append(publish)
                base += root.count
                continue
            lengths = {record[5] for record in records}
            static_events = None
            if len(lengths) == 1 and 0 < records[0][5] <= 4:
                candidates = [
                    static_u32_expression(
                        tuple(plan.publications[record[4] + slot] for record in records)
                    )
                    for slot in range(records[0][5])
                ]
                if all(candidate is not None for candidate in candidates):
                    static_events = tuple(candidates)
            if static_events is not None:
                count = IndexConstantOp.build(records[0][5])
                work.append(count)
                offset, length = tile.result(), count.result()
            elif any(record[5] for record in records):
                kernel.attrs["tile_static_publication_table_roots"] = (
                    cast(int, kernel.attrs["tile_static_publication_table_roots"]) + 1
                )
                publication = PTXTemplateOp.build(
                    [tile.result()],
                    [IndexType(), IndexType()],
                    f"""
                    add.u32 ${{b32:position}}, ${{b32:operand.tile.val}}, {base};
                    mul.wide.u32 ${{b64:offset}}, ${{b32:position}}, 24;
                    mov.u64 ${{b64:table}}, ayla_tile_tasks;
                    add.u64 ${{b64:record}}, ${{b64:table}}, ${{b64:offset}};
                    ld.global.u32 ${{b32:result.offset.val}}, [${{b64:record}}+16];
                    ld.global.u32 ${{b32:result.length.val}}, [${{b64:record}}+20];
                    """,
                    operand_names=["tile"],
                    result_names=["offset", "length"],
                    description="load tile publications",
                )
                work.append(publication)
                offset, length = publication.results
            else:
                zero = IndexConstantOp.build(0)
                work.append(zero)
                offset = length = zero.result()
            publish = TileSyncOp.build(sync, offset, length, action="publish")
            publish.attrs["async_proxy"] = async_proxy
            if static_events is not None:
                publish.attrs["static_events"] = static_events
            # After the full store drain and CTA join, only thread zero touches
            # immutable publication tables/counters. Other warps may proceed;
            # no payload or shared-memory use remains in this publication.
            publish.attrs["join_after"] = False
            work.append(publish)
            branch = IfOp.build(valid.result(), Block(ops=work))
            branch.attrs["tile_root"] = i
            loop.body.ops = [tile, limit, valid, branch]
            body.append(loop)
            base += root.count
        assert base == len(plan.records)
        kernel.body.ops = body
