"""Analyze specialized Ayla tile bodies across graph phases.

This adapts Helion's access-region/dependency model to Ayla's memory IR. Regions
are byte intervals on root allocations, so tiles, transposes and reshapes alias
correctly. Unsupported index expressions conservatively cover the whole root.
Read-after-write, write-after-read and write-after-write edges preserve snapshots
and scratch reuse. Bounds are overapproximations, never speculative disjointness.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import cast

from ..base.ir import Block, Value
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    ConstantOp,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexEqOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    PTXTemplateOp,
    ReshapeOp,
    SelectOp,
    ThreadIdOp,
    TileOp,
    TmaMoveOp,
    WarpGroupIdOp,
    WarpIdsOp,
    memory_effects,
    root_buffers,
)
from .interval_set import IntervalSet

Bounds = tuple[int, int]  # inclusive


@dataclass(frozen=True)
class TileAccess:
    root: Value
    region: IntervalSet
    write: bool


@dataclass(frozen=True)
class TileTask:
    root_id: int
    tile_id: int
    accesses: tuple[TileAccess, ...]


@dataclass(frozen=True)
class TileDependencyGraph:
    tasks: tuple[TileTask, ...]
    predecessors: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        if len(self.tasks) != len(self.predecessors):
            raise ValueError("each task must have a predecessor set")
        if any(
            p < 0 or p >= i for i, deps in enumerate(self.predecessors) for p in deps
        ):
            raise ValueError("tile dependencies must point backwards in source order")


def build_tile_dependency_graph(tasks: tuple[TileTask, ...]) -> TileDependencyGraph:
    # Index by allocation and access kind: read-only weights never cause pairs.
    readers: dict[Value, list[tuple[int, IntervalSet]]] = {}
    writers: dict[Value, list[tuple[int, IntervalSet]]] = {}
    predecessors: list[frozenset[int]] = []
    for i, task in enumerate(tasks):
        deps: set[int] = set()
        for access in task.accesses:
            previous = writers.get(access.root, [])
            if access.write:
                previous = [*previous, *readers.get(access.root, [])]
            for p, region in previous:
                if p != i and access.region.overlaps(region):
                    deps.add(p)
        predecessors.append(frozenset(deps))
        for access in task.accesses:
            target = writers if access.write else readers
            target.setdefault(access.root, []).append((i, access.region))
    return TileDependencyGraph(tasks, tuple(predecessors))


class TileDependencyAnalyzer:
    """Conservatively summarize one CTA tile with a specialized logical ID."""

    def __init__(self, *, threads: int = 256, max_intervals: int = 4096):
        if type(threads) is not int or threads < 128 or threads > 1024 or threads % 128:
            raise ValueError("tile analysis requires complete warpgroups")
        if type(max_intervals) is not int or max_intervals < 1:
            raise ValueError("tile interval budget must be positive")
        self.threads, self.max_intervals = threads, max_intervals

    def accesses(
        self, body: Block, tile: Value, tile_id: int, *, thread_group_size: int = 128
    ) -> tuple[TileAccess, ...]:
        combined: dict[tuple[Value, bool], IntervalSet] = {}
        # Bound each cooperating thread group separately. Smaller scalar row
        # groups avoid merging unrelated head slices across token strides;
        # tensor-core callers retain the default warpgroup partition.
        if thread_group_size not in (16, 32, 64, 128):
            raise ValueError("thread analysis groups must divide a warpgroup")
        for first_lane in range(0, self.threads, thread_group_size):
            wg = first_lane // 128
            env: dict[Value, Bounds | None] = {tile: (tile_id, tile_id)}
            memo: dict[Value, Bounds | None] = {}

            def bounds(v: Value) -> Bounds | None:
                if v in env:
                    return env[v]
                if v in memo:
                    return memo[v]
                op = v.definition
                result: Bounds | None = None
                if isinstance(op, IndexConstantOp):
                    result = (op.value, op.value)
                elif isinstance(op, BlockIdOp):
                    result = (0, 0) if op.axis.value else None
                elif isinstance(op, ThreadIdOp):
                    result = (
                        (first_lane, first_lane + thread_group_size - 1) if op.axis.value == 0 else (0, 0)
                    )
                elif isinstance(op, WarpGroupIdOp):
                    result = (wg, wg)
                elif isinstance(op, WarpIdsOp):
                    result = None
                elif isinstance(
                    op,
                    (
                        IndexAddOp,
                        IndexMulOp,
                        IndexDivOp,
                        IndexModOp,
                        IndexLtOp,
                        IndexEqOp,
                    ),
                ):
                    a, b = (bounds(x) for x in op.operands)
                    if a is not None and b is not None:
                        if isinstance(op, IndexAddOp):
                            result = (a[0] + b[0], a[1] + b[1])
                        elif isinstance(op, IndexMulOp):
                            products = [x * y for x in a for y in b]
                            result = (min(products), max(products))
                        elif isinstance(op, IndexDivOp) and b[0] > 0 and a[0] >= 0:
                            result = (a[0] // b[1], a[1] // b[0])
                        elif (
                            isinstance(op, IndexModOp)
                            and b[0] == b[1]
                            and b[0] > 0
                            and a[0] >= 0
                        ):
                            d = b[0]
                            result = (
                                (a[0] % d, a[1] % d)
                                if a[0] // d == a[1] // d
                                else (0, d - 1)
                            )
                        elif isinstance(op, IndexLtOp):
                            result = (
                                (1, 1)
                                if a[1] < b[0]
                                else (0, 0)
                                if a[0] >= b[1]
                                else (0, 1)
                            )
                        elif isinstance(op, IndexEqOp):
                            result = (
                                (0, 0)
                                if a[1] < b[0] or b[1] < a[0]
                                else (1, 1)
                                if a == b and a[0] == a[1]
                                else (0, 1)
                            )
                elif isinstance(op, SelectOp):
                    choices = [bounds(x) for x in op.operands[1:]]
                    if all(x is not None for x in choices):
                        result = (
                            min(cast(Bounds, x)[0] for x in choices),
                            max(cast(Bounds, x)[1] for x in choices),
                        )
                memo[v] = result
                return result

            def origin(v: Value) -> Bounds | None:
                op = v.definition
                if isinstance(op, ReshapeOp):
                    return origin(op.arg)
                if isinstance(op, TileOp):
                    base = origin(op.arg)
                    if base is None:
                        return None
                    typ = cast(GmemTensorType, op.result().type)
                    parent = cast(GmemTensorType, op.arg.type)
                    lo, hi = base
                    for index, size, stride in zip(
                        op.indices, typ.static_shape, parent.strides, strict=True
                    ):
                        at = (index, index) if isinstance(index, int) else bounds(index)
                        if at is None or stride is None or stride < 0:
                            return None
                        lo += at[0] * size * stride
                        hi += at[1] * size * stride
                    return lo, hi
                return (0, 0) if v in root_buffers(v) else None

            def region(v: Value, root: Value) -> IntervalSet:
                typ = cast(GmemTensorType, v.type)
                if not isinstance(root.type, GmemTensorType):
                    raise ValueError(
                        "global tile accesses require global root allocations"
                    )
                rt = cast(GmemTensorType, root.type)
                if (
                    typ.is_dynamic
                    or rt.is_dynamic
                    or None in typ.strides
                    or None in rt.strides
                ):
                    raise ValueError(
                        "specialize shapes and strides before tile dependency analysis"
                    )
                itemsize = cast(int, getattr(typ.dtype, "byte_size"))
                strides = cast(tuple[int, ...], typ.strides)
                if 0 in rt.sizes or 0 in typ.sizes:
                    return IntervalSet()
                if any(s < 0 for s in cast(tuple[int, ...], rt.strides)):
                    raise ValueError(
                        "tile analysis does not support negative root strides"
                    )
                root_end = 1 + sum(
                    (s - 1) * d
                    for s, d in zip(
                        rt.static_shape, cast(tuple[int, ...], rt.strides), strict=True
                    )
                )
                full = IntervalSet.build(
                    [(0, root_end * cast(int, getattr(rt.dtype, "byte_size")))]
                )
                if typ.dtype != rt.dtype:
                    return full
                base = origin(v)
                if base is None or any(s < 0 for s in strides):
                    return full
                shape = list(typ.static_shape)
                # Clip partial TMA tiles along each logical axis. Clipping just
                # the root's byte envelope would spill past a row's valid
                # columns into the following row and invent dependencies.
                current = v
                offsets = [0] * typ.rank
                while isinstance(current.definition, TileOp):
                    view = current.definition
                    vt = cast(GmemTensorType, current.type)
                    pt = cast(GmemTensorType, view.arg.type)
                    if (
                        vt.rank != typ.rank
                        or pt.rank != typ.rank
                        or pt.strides != strides
                    ):
                        break
                    known = True
                    for d, (index, size) in enumerate(
                        zip(view.indices, vt.static_shape, strict=True)
                    ):
                        at = (index, index) if isinstance(index, int) else bounds(index)
                        if at is None or at[0] < 0:
                            known = False
                            break
                        offsets[d] += at[0] * size
                    if not known:
                        break
                    for d, extent in enumerate(pt.static_shape):
                        shape[d] = min(shape[d], max(0, extent - offsets[d]))
                    current = view.arg
                if not all(shape):
                    return IntervalSet()
                # Collapse the innermost contiguous axes, preserving holes
                # between rows for gate/up slices and transposed tensors.
                dimensions = sorted(
                    zip(shape, strides, strict=True), key=lambda x: x[1]
                )
                span = 1
                remaining = []
                for size, stride in dimensions:
                    if stride <= span:
                        span += (size - 1) * stride
                    else:
                        remaining.append((size, stride))
                if math.prod(s for s, _ in remaining) > self.max_intervals:
                    end = (
                        base[1]
                        + 1
                        + sum((s - 1) * d for s, d in zip(shape, strides, strict=True))
                    )
                    return IntervalSet.build(
                        [(base[0] * itemsize, end * itemsize)]
                    ).intersection(full)
                intervals = []
                for coordinates in itertools.product(*(range(s) for s, _ in remaining)):
                    offset = sum(
                        c * d for c, (_, d) in zip(coordinates, remaining, strict=True)
                    )
                    intervals.append(
                        (
                            (base[0] + offset) * itemsize,
                            (base[1] + offset + span) * itemsize,
                        )
                    )
                return IntervalSet.build(intervals).intersection(full)

            def visit(block: Block) -> None:
                for op in block.ops:
                    if isinstance(op, ForOp):
                        if op.trip_count:
                            env[op.iv] = (op.lb, op.lb + (op.trip_count - 1) * op.step)
                            memo.clear()
                            visit(op.body)
                        continue
                    if isinstance(op, IfOp):
                        condition = bounds(op.condition)
                        if condition != (0, 0):
                            visit(op.body)
                        if op.else_body is not None and condition != (1, 1):
                            visit(op.else_body)
                        continue
                    if isinstance(op, AsyncOp):
                        visit(op.body)
                        continue
                    if isinstance(op, (AllocOp, AwaitOp, BarrierOp, ConstantOp)):
                        continue
                    if isinstance(op, TmaMoveOp):
                        raise ValueError(
                            "analyze tile dependencies before TMA descriptor lowering"
                        )
                    effects = memory_effects(op)
                    if effects is None:
                        # Unknown scalar memory templates may carry a precise
                        # address in an unsupported register expression.
                        values = [
                            x for x in op.operands if isinstance(x.type, GmemTensorType)
                        ]
                        if values:
                            raise ValueError(
                                f"unsupported global memory operation for tile analysis: {op.op_id}"
                            )
                        if isinstance(op, PTXTemplateOp):
                            raise ValueError(
                                "analyze tile dependencies before PTX template lowering"
                            )
                        continue
                    for effect in effects:
                        if not isinstance(effect.value.type, GmemTensorType):
                            continue
                        roots = root_buffers(effect.value)
                        if not roots:
                            raise ValueError(
                                "tile analysis requires known root allocations"
                            )
                        for root in roots:
                            key = (root, effect.kind == "W")
                            combined[key] = combined.get(key, IntervalSet()).union(
                                region(effect.value, root)
                            )

            visit(body)
        return tuple(
            TileAccess(root, region, write)
            for (root, write), region in combined.items()
            if region.intervals
        )
