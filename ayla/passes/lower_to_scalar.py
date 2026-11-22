"""Expose distributed tensor operations as scalar GPU IR.

Loop structure, partial partitions, ownership predicates, and shuffle trees
are transformations over IR. PTX lowering subsequently selects instructions
for scalar moves/arithmetic and explicit shuffles.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import cast

from ..base.ir import Block, Op, Type, Value, clone_type, find_ancestor_of_type, find_ops_of_type
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AddrMemTensorType, AllocOp, CollectiveAxis, CollectiveLevel, ConstantOp,
    ElementwiseOp, F32Type, ForOp, IfOp, IndexAddOp,
    IndexConstantOp, IndexDivOp, IndexEqOp, IndexLtOp, IndexModOp, IndexMulOp,
    KernelOp, MemTensorType, MemoryEffect, MoveOp, ReduceOp, RegisterViewOp,
    RegMemTensorType, ReshapeOp, RmemTensorType, ShuffleOp, SmemTensorType, ThreadIdOp, TileOp, WgrmemTensorType,
    WrmemTensorType,
)
from ..register_layout import broadcast_register, register_groups, register_layout


IndexBinaryOp = IndexAddOp | IndexDivOp | IndexEqOp | IndexLtOp | IndexModOp | IndexMulOp


def _has_partial_view(value: Value) -> bool:
    while isinstance(value.definition, (TileOp, ReshapeOp)):
        op = value.definition
        if isinstance(op, TileOp) and op.maybe_partial:
            return True
        value = op.arg
    return False


class _ScalarBuilder:
    """Build ordinary scalar operations with explicit mutable accumulators."""
    def __init__(self, anchor: Op, ops: list[Op], allocated: set[int]):
        self.anchor, self.ops, self.allocated = anchor, ops, allocated
        self.local = {axis: CollectiveLevel.THREAD for axis in CollectiveAxis}
        self.scheme = MemoryEffect.fresh_tiling_scheme_id()

    def emit(self, op: Op) -> None:
        self.ops.append(op)

    def value(self, op: Op) -> Value:
        self.emit(op)
        return op.result()

    def index(self, value: int | Value) -> Value:
        return value if isinstance(value, Value) else self.value(IndexConstantOp.build(value))

    def binary(self, cls: type[IndexBinaryOp], lhs: int | Value, rhs: int | Value) -> Value:
        return self.value(cls.build(self.index(lhs), self.index(rhs)))

    def allocate(self, typ: RegMemTensorType) -> Value:
        alloc = AllocOp.build_with_fresh_id(typ)
        while alloc.attrs['id'] in self.allocated:
            alloc = AllocOp.build_with_fresh_id(alloc.result().type)
        self.allocated.add(cast(int, alloc.attrs['id']))
        return self.value(alloc)

    def scalar(self, dtype: Type = F32Type()) -> Value:
        return self.allocate(RmemTensorType.build([], dtype))

    def constant(self, value: int | float, dtype: Type = F32Type()) -> Value:
        return self.value(ConstantOp.build(value, RmemTensorType.build([], dtype)))

    def view(self, value: Value, index: int) -> Value:
        return self.value(RegisterViewOp.build(value, index))

    def arithmetic(self, dst: Value, srcs: Sequence[Value], fn: str) -> None:
        self.emit(ElementwiseOp.build(dst, list(srcs), fn))

    def copy(self, dst: Value, src: Value, attrs: dict[str, object] | None = None) -> None:
        self.emit(MoveOp.build(dst, src, collective=self.local, attrs=attrs))

    def guard(self, predicate: Value, body: Callable[[], None]) -> None:
        previous = self.ops
        nested: list[Op] = []
        self.ops = nested
        body()
        self.ops = previous
        self.emit(IfOp.build(predicate, Block(ops=nested)))

    def branch(self, predicate: Value, body: Callable[[], None], alternative: Callable[[], None]) -> None:
        previous = self.ops
        self.ops = []
        body()
        yes = Block(ops=self.ops)
        self.ops = []
        alternative()
        no = Block(ops=self.ops)
        self.ops = previous
        self.emit(IfOp.build(predicate, yes, no))

    def element(self, value: Value, coordinates: Sequence[int | Value]) -> Value:
        typ = cast(AddrMemTensorType, value.type)
        if typ.attrs.get('physical_layout') is not None or isinstance(typ, SmemTensorType) and typ.swizzle is not None:
            raise NotImplementedError('scalar memory decomposition requires affine, unswizzled storage')
        return self.value(TileOp.build(value, list(coordinates), clone_type(typ, {'sizes': [1] * typ.rank})))

    def memory_guards(self, memory: Value, coordinates: Sequence[int | Value]) -> list[Value]:
        """Map logical coordinates through views to each partial tile's bounds."""
        if not _has_partial_view(memory):
            return []
        predicates = []
        coordinates = list(coordinates)
        while isinstance(memory.definition, (TileOp, ReshapeOp)):
            op = memory.definition
            shape = cast(AddrMemTensorType, memory.type).static_shape
            parent = cast(AddrMemTensorType, op.arg.type)
            if isinstance(op, TileOp):
                coordinates = [self.binary(IndexAddOp, c, self.binary(IndexMulOp, i, s))
                               for c, i, s in zip(coordinates, op.indices, op.index_steps, strict=True)]
                if op.maybe_partial:
                    predicates.extend(self.binary(IndexLtOp, c, s)
                                      for c, s in zip(coordinates, parent.static_shape, strict=True))
            else:
                flat = self.index(0)
                for c, s in zip(coordinates, shape, strict=True):
                    flat = self.binary(IndexAddOp, self.binary(IndexMulOp, flat, s), c)
                coordinates = [self.binary(IndexModOp, self.binary(IndexDivOp, flat, math.prod(parent.static_shape[d + 1:])), s)
                               for d, s in enumerate(parent.static_shape)]
            memory = op.arg
        return predicates

    def guarded_memory(self, memory: Value, coordinates: Sequence[int | Value], body: Callable[[], None]) -> None:
        predicates = self.memory_guards(memory, coordinates)
        def nest(index: int) -> None:
            if index == len(predicates):
                body()
            else:
                self.guard(predicates[index], lambda: nest(index + 1))
        nest(0)

    def load(self, dst: Value, memory: Value, coordinates: Sequence[int | Value],
             padding: int | float = 0.0) -> None:
        attrs = {'src_tiling_scheme': self.anchor.attrs.get('src_tiling_scheme', self.scheme)}
        if 'nc' in self.anchor.attrs:
            attrs['nc'] = self.anchor.attrs['nc']
        if _has_partial_view(memory):
            self.copy(dst, self.constant(padding, cast(RegMemTensorType, dst.type).dtype))
        self.guarded_memory(memory, coordinates, lambda: self.copy(dst, self.element(memory, coordinates), attrs))

    def store(self, memory: Value, coordinates: Sequence[int | Value], src: Value) -> None:
        attrs = {'dst_tiling_scheme': self.anchor.attrs.get('dst_tiling_scheme', self.scheme)}
        if 'atomic_add' in self.anchor.attrs:
            attrs['atomic_add'] = self.anchor.attrs['atomic_add']
        self.guarded_memory(memory, coordinates, lambda: self.copy(self.element(memory, coordinates), src, attrs))

    def lane(self) -> Value:
        tid = self.value(ThreadIdOp.build(CollectiveAxis.X))
        return self.binary(IndexModOp, tid, 32)

    def block_thread(self) -> tuple[Value, int]:
        kernel = find_ancestor_of_type(self.anchor, KernelOp)
        assert kernel is not None
        nx, ny, nz = (extent or 1 for extent in kernel.block_dim)
        tx, ty, tz = (self.value(ThreadIdOp.build(axis)) for axis in CollectiveAxis)
        linear = self.binary(IndexAddOp, self.binary(IndexMulOp, tz, ny), ty)
        return self.binary(IndexAddOp, self.binary(IndexMulOp, linear, nx), tx), nx * ny * nz

    def shuffle_fold(self, acc: Value, fn: str, lanes: int) -> None:
        if lanes < 1 or lanes > 32 or lanes & (lanes - 1):
            raise ValueError('shuffle group must be a power of two within a warp')
        for bit in reversed(range(lanes.bit_length() - 1)):
            shuffled = self.value(ShuffleOp.build(acc, 1 << bit, 'bfly'))
            self.arithmetic(acc, [acc, shuffled], fn)

    def memory_fold(self, source: Value, row: tuple[int, ...], start: int | Value,
                    step: int, fn: str, identity: int | float, unroll: int) -> Value:
        acc = self.scalar()
        self.copy(acc, self.constant(identity))
        accumulators = [acc] if unroll == 1 else [self.scalar() for _ in range(unroll)]
        if unroll != 1:
            for value in accumulators:
                self.copy(value, self.constant(identity))
        values = [self.scalar() for _ in range(unroll)]
        loop = ForOp.build(0, cast(MemTensorType, source.type).static_shape[-1], step * unroll,
                           attrs={'label': 'reduction_load'})
        self.emit(loop)
        previous, self.ops = self.ops, loop.body.ops
        active = []
        for u, value in enumerate(values):
            column = self.binary(IndexAddOp, loop.iv, self.binary(IndexAddOp, start, u * step))
            condition = self.binary(IndexLtOp, column, cast(MemTensorType, source.type).static_shape[-1])
            self.guard(condition, lambda value=value, column=column: self.load(value, source, (*row, column), padding=identity))
            active.append(condition)
        for value, accumulator, condition in zip(values, accumulators, active, strict=True):
            self.guard(condition, lambda value=value, accumulator=accumulator: self.arithmetic(accumulator, [accumulator, value], fn))
        self.ops = previous
        if unroll != 1:
            for accumulator in accumulators:
                self.arithmetic(acc, [acc, accumulator], fn)
        return acc


class LowerToScalar(LoweringPass):
    """Decompose reductions, distributed transfers, and register broadcasts.

    Sum/max are scalar elementwise combinations. Normalization functions do
    not participate: any graph using these primitives follows the same path.
    """
    def __init__(self):
        super().__init__()
        self.allocated: set[int] = set()

    def run(self, module):
        self.value_map.clear()
        self.allocated = {cast(int, op.attrs['id']) for op in find_ops_of_type(module, AllocOp) if 'id' in op.attrs}
        return super().run(module)

    def builder(self, op: Op) -> tuple[list[Op], _ScalarBuilder]:
        ops: list[Op] = []
        return ops, _ScalarBuilder(op, ops, self.allocated)

    @lowers(ReduceOp)
    def lower_reduce(self, op: ReduceOp) -> tuple[list[Op], list[object]]:
        dst, src = cast(MemTensorType, op.dst.type), cast(MemTensorType, op.src.type)
        if dst.dtype != F32Type() or src.dtype != F32Type():
            raise NotImplementedError('scalar reductions require F32 operands')
        unroll = op.attrs.get('load_unroll', 1)
        if type(unroll) is not int or unroll < 1:
            raise ValueError('reduction load_unroll must be a positive integer')
        ops, b = self.builder(op)
        fn = ReduceOp.COMBINATIONS[op.fn]
        identity = ReduceOp.NEUTRAL_ELEMENTS[op.fn]
        assert identity is not None

        def finish(index, acc, predicate=None):
            def write():
                out = b.view(op.dst, index)
                initial = b.constant(op.init) if op.init is not None else out
                b.arithmetic(out, [initial, acc], fn)
            if predicate is None:
                write()
            else:
                b.guard(predicate, write)

        if isinstance(src, AddrMemTensorType) and isinstance(dst, WrmemTensorType):
            layout = register_layout(op.dst)
            p, lanes = layout.shape[-1], layout.column_lanes
            lane = b.lane()
            rows = itertools.product(*(range(s) for s in layout.shape[:-1]))
            for row_index, row in enumerate(rows):
                for q in (range(p) if p < lanes else range(0, p, lanes)):
                    start = b.binary(IndexAddOp, b.binary(IndexMulOp, lane, p), q) if p < lanes else b.binary(IndexAddOp, lane, q)
                    acc = b.memory_fold(op.src, row, start, p * lanes if p < lanes else p, fn, identity, unroll)
                    if p < lanes:
                        b.shuffle_fold(acc, fn, lanes)
                        owner = b.binary(IndexEqOp, lane, q)
                    else:
                        owner = b.binary(IndexLtOp, start, p)
                    finish(row_index * ((p + lanes - 1) // lanes) + q // lanes, acc, owner)
            return ops, []

        if not isinstance(src, AddrMemTensorType) and not isinstance(dst, AddrMemTensorType):
            source, destination = register_layout(op.src), register_layout(op.dst)
            if isinstance(src, WgrmemTensorType) or isinstance(dst, WgrmemTensorType):
                if not isinstance(src, WgrmemTensorType) or not isinstance(dst, WgrmemTensorType):
                    raise NotImplementedError('WGMMA reductions require register operands with matching ownership')
            p = destination.shape[-1]
            lane = b.lane() if source.column_lanes > 1 or destination.column_lanes > 1 else None
            if p == 1 or p % source.column_period == 0 and source.column_period == destination.column_period:
                for group in register_groups(source, destination):
                    acc = b.scalar()
                    b.copy(acc, b.constant(identity))
                    for element in group.sources:
                        value = b.view(op.src, element.register)
                        def combine(value=value):
                            b.arithmetic(acc, [acc, value], fn)
                        if not source.distributed_rows and source.column_lanes > 1 and element.coordinates[-1] + source.column_period > source.shape[-1]:
                            b.guard(b.binary(IndexLtOp, cast(Value, lane), source.shape[-1] - element.coordinates[-1]), combine)
                        else:
                            combine()
                    b.shuffle_fold(acc, fn, group.shuffle_lanes)
                    owner = None
                    if not destination.distributed_rows and destination.column_lanes > 1:
                        owner = b.binary(IndexLtOp, cast(Value, lane), min(destination.column_lanes, p - group.destination.coordinates[-1]))
                    finish(group.destination.register, acc, owner)
            else:
                if source.distributed_rows or destination.distributed_rows:
                    raise NotImplementedError('WGMMA partial columns must preserve lane ownership')
                for row in itertools.product(*(range(s) for s in source.shape[:-1])):
                    for q in range(p):
                        acc = b.scalar()
                        b.copy(acc, b.constant(identity))
                        for column in range(q, source.shape[-1], p):
                            index, owner_lane = source.locate((*row, column))
                            value = b.view(op.src, index)
                            if owner_lane is not None:
                                value = b.value(ShuffleOp.build(cast(Value, value), owner_lane))
                            b.arithmetic(acc, [acc, value], fn)
                        index, owner_lane = destination.locate((*row, q))
                        owner = b.binary(IndexEqOp, cast(Value, lane), owner_lane) if owner_lane is not None else None
                        finish(index, acc, owner)
            return ops, []

        if isinstance(src, WrmemTensorType):
            raise NotImplementedError('promote the reduction destination to warp registers')
        if isinstance(dst, WgrmemTensorType) or isinstance(src, WgrmemTensorType):
            raise NotImplementedError('WGMMA reductions require register operands with matching ownership')
        p = dst.static_shape[-1]
        source_layout = register_layout(op.src) if not isinstance(src, AddrMemTensorType) else None
        leader = thread = None
        threads = 1
        level = op.collective[CollectiveAxis.X]
        if isinstance(dst, AddrMemTensorType):
            if level == CollectiveLevel.THREAD_32:
                leader = b.binary(IndexEqOp, b.lane(), 0)
            elif level == CollectiveLevel.BLOCK:
                thread, threads = b.block_thread()
            elif level != CollectiveLevel.THREAD:
                raise NotImplementedError('tile grid/cluster reductions before scalarization')
        for row_index, row in enumerate(itertools.product(*(range(s) for s in src.static_shape[:-1]))):
            for q in range(p):
                def compute(row=row, q=q, row_index=row_index):
                    if isinstance(src, AddrMemTensorType):
                        acc = b.memory_fold(op.src, row, q, p, fn, identity, unroll)
                    else:
                        assert source_layout is not None
                        acc = b.scalar()
                        b.copy(acc, b.constant(identity))
                        for column in range(q, src.static_shape[-1], p):
                            index, _ = source_layout.locate((*row, column))
                            b.arithmetic(acc, [acc, b.view(op.src, index)], fn)
                    if isinstance(dst, AddrMemTensorType):
                        initial = b.constant(op.init) if op.init is not None else b.scalar()
                        if op.init is None:
                            b.load(initial, op.dst, (*row, q))
                        b.arithmetic(acc, [initial, acc], fn)
                        b.store(op.dst, (*row, q), acc)
                    else:
                        finish(row_index * p + q, acc)
                owner = b.binary(IndexEqOp, thread, (row_index * p + q) % threads) if thread is not None else leader
                if owner is None:
                    compute()
                else:
                    b.guard(owner, compute)
        return ops, []

    @lowers(MoveOp)
    def lower_move(self, op: MoveOp) -> tuple[list[Op], list[object]] | None:
        dst, src = op.dst.type, op.src.type
        if not any(isinstance(t, AddrMemTensorType) for t in (dst, src)):
            return None
        load = not isinstance(dst, AddrMemTensorType)
        fragment, memory = (op.dst, op.src) if load else (op.src, op.dst)
        if not (isinstance(fragment.type, RmemTensorType) and _has_partial_view(memory)
                or isinstance(fragment.type, WrmemTensorType)
                or isinstance(fragment.type, WgrmemTensorType) and (
                    fragment.type.layout == 'wgmma_row'
                    or fragment.type.dtype == F32Type() and _has_partial_view(memory))):
            return None
        layout = register_layout(fragment)
        typ = cast(AddrMemTensorType, memory.type)
        scalar = typ.num_elements == 1 and cast(MemTensorType, fragment.type).num_elements == 1
        if (typ.static_shape != layout.shape and not scalar) or typ.dtype != cast(MemTensorType, fragment.type).dtype:
            raise NotImplementedError('distributed moves require equal shapes and float dtypes')
        ops, b = self.builder(op)
        bases = []
        writer = None
        if not load and 'writer_wg' in op.attrs:
            group = b.binary(IndexDivOp, b.value(ThreadIdOp.build(CollectiveAxis.X)), 128)
            if 'writer_wg_divisor' in op.attrs:
                group = b.binary(IndexDivOp, group, cast(int, op.attrs['writer_wg_divisor']))
            if 'writer_wg_modulus' in op.attrs:
                group = b.binary(IndexModOp, group, cast(int, op.attrs['writer_wg_modulus']))
            writer = b.binary(IndexEqOp, group, cast(int, op.attrs['writer_wg']))
        for terms in layout.thread_bases or ((),) * len(layout.shape):
            total = None
            for term in terms:
                value = b.value(ThreadIdOp.build(CollectiveAxis.X))
                if term.source == 'lane':
                    value = b.binary(IndexModOp, value, 32)
                if term.divisor != 1:
                    value = b.binary(IndexDivOp, value, term.divisor)
                if term.modulus is not None:
                    value = b.binary(IndexModOp, value, term.modulus)
                if term.scale != 1:
                    value = b.binary(IndexMulOp, value, term.scale)
                total = value if total is None else b.binary(IndexAddOp, total, value)
            bases.append(total)
        for element in layout.elements:
            slot = b.view(fragment, element.register)
            coordinates = tuple(c if base is None else b.binary(IndexAddOp, base, c)
                                for c, base in zip(element.coordinates, bases, strict=True))
            if scalar:
                coordinates = (0,) * typ.rank
            def move(slot=slot, coordinates=coordinates):
                if load:
                    b.load(slot, memory, coordinates)
                else:
                    if writer is None:
                        b.store(memory, coordinates, slot)
                    else:
                        b.guard(writer, lambda: b.store(memory, coordinates, slot))
            if bases and bases[-1] is not None:
                b.guard(b.binary(IndexLtOp, coordinates[-1], layout.shape[-1]), move)
            elif layout.replicas > 1 and not load:
                b.guard(b.binary(IndexEqOp, b.binary(IndexModOp, b.lane(), layout.replicas), 0), move)
            else:
                move()
        return ops, []

    @lowers(ElementwiseOp)
    def lower_elementwise(self, op: ElementwiseOp) -> tuple[list[Op], list[object]] | None:
        if op.fn in ('fill', 'convert'):
            return None
        if any(not isinstance(v.type, RegMemTensorType) for v in op.operands):
            return None
        if any(cast(RegMemTensorType, v.type).dtype != F32Type() for v in op.operands[1:]):
            return None
        dst_type = cast(RegMemTensorType, op.dst.type)
        compute_type = cast(RegMemTensorType, dst_type.with_shape(dst_type.static_shape, F32Type()))
        if all(value.type == compute_type for value in op.operands[1:]):
            return None
        ops, b = self.builder(op)
        compute = op.dst if dst_type.dtype == F32Type() else b.allocate(compute_type)
        destination = register_layout(compute)
        sources = [register_layout(v) for v in op.operands[1:]]
        for element in destination.elements:
            values = []
            for value, source in zip(op.operands[1:], sources, strict=True):
                index, shuffle = broadcast_register(source, destination, element)
                scalar = b.view(value, index)
                if shuffle:
                    scalar = b.value(ShuffleOp.build(scalar, 0))
                values.append(scalar)
            b.arithmetic(b.view(compute, element.register), values, op.fn)
        if compute is not op.dst:
            b.arithmetic(op.dst, [compute], 'convert')
        return ops, []
