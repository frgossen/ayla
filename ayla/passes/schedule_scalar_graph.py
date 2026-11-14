"""Scalar reference scheduling and pointwise/reduction graph phases.

The reference uses one CTA and scalar GEMMs. The persistent graph scheduler
also uses these helpers for fused pointwise and row-reduction regions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

from ..base.ir import Op, Type, Value, clone_type, find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    BarrierOp,
    BlockIdOp,
    CollectiveAxis,
    CollectiveLevel,
    ConstantOp,
    ElementwiseOp,
    F32Type,
    BF16Type,
    ForOp,
    GatherOp,
    GmemTensorType,
    IndexAddOp,
    IndexDivOp,
    IndexEqOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LoadIndexOp,
    MatmulOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    PTXTemplateOp,
    PredicateType,
    ReduceOp,
    root_buffer,
    ReshapeOp,
    RmemTensorType,
    TileOp,
    WrmemTensorType,
    SmemTensorType,
)
from .lower_to_scalar import _ScalarBuilder
from ._utils import can_reorder


def _row_invariant_maps(
    effects: list[ElementwiseOp | MoveOp | ReduceOp | GatherOp], live_destinations: set[Value],
) -> list[ElementwiseOp | MoveOp]:
    """Find private rank-one maps that may move before the row loop.

    Keep snapshots and in-place updates ordered against every crossed effect.
    Values are recomputed on each kernel invocation, including graph replays.
    """
    live_roots = {root_buffer(value) for value in live_destinations}
    result: list[ElementwiseOp | MoveOp] = []
    for position, effect in enumerate(effects):
        if (
            isinstance(effect, (ElementwiseOp, MoveOp))
            and cast(MemTensorType, effect.dst.type).rank == 1
            and root_buffer(effect.dst) not in live_roots
            and all(cast(MemTensorType, value.type).rank <= 1
                    for value in effect.operands[1:])
            and all(can_reorder(earlier, effect)
                    for earlier in effects[:position] if earlier not in result)
        ):
            result.append(effect)
    return result


class ScheduleScalarGraph(Pass):
    """Assign a static graph to one block of threads, using scalar F32 math.

    This schedule prioritizes correctness and compiler coverage. It does not
    use tensor cores and is unsuitable for performance measurements.
    """

    def __init__(
        self,
        threads: int = 128,
        blocks: int = 1,
        sync: Value | None = None,
        vector_width: int = 1,
        row_prefetch_slots: int = 0,
        row_threads: int | None = None,
        row_vector_width: int = 1,
        tile_tasks: bool = False,
        warp_row_threads: int = 32,
    ):
        super().__init__()
        if type(threads) is not int or threads < 32 or threads > 1024 or threads % 32:
            raise ValueError(
                "scalar graph threads must be a multiple of 32 in [32, 1024]"
            )
        if type(blocks) is not int or blocks < 1:
            raise ValueError("scalar graph blocks must be positive")
        if type(vector_width) is not int or vector_width < 1:
            raise ValueError("scalar graph vector width must be positive")
        if type(row_prefetch_slots) is not int or row_prefetch_slots < 0:
            raise ValueError("row prefetch slots must be a nonnegative integer")
        if type(warp_row_threads) is not int or warp_row_threads not in (16, 32):
            raise ValueError("warp row threads must be 16 or 32")
        self.warp_row_threads = warp_row_threads
        if row_threads is not None and (
            type(row_threads) is not int or row_threads < 64
            or row_threads % 32 or threads % row_threads
        ):
            raise ValueError("row threads must be a multiple of 32 dividing the CTA, at least 64")
        if blocks > 1 and sync is None:
            raise ValueError(
                "a multi-CTA scalar phase requires grid synchronization storage"
            )
        self.threads, self.blocks, self.sync = threads, blocks, sync
        self.vector_width = vector_width
        self.row_prefetch_slots = row_prefetch_slots
        self.row_threads = row_threads
        if row_vector_width not in (1, 2, 4) or type(row_vector_width) is not int:
            raise ValueError("row vector width must be 1, 2 or 4")
        self.row_vector_width = row_vector_width
        self.tile_tasks = tile_tasks

    def run(self, module: ModuleOp) -> ModuleOp:
        if len(module.body.ops) != 1 or not isinstance(module.body.ops[0], KernelOp):
            raise ValueError("scalar graph scheduling requires one kernel")
        kernel = module.body.ops[0]
        if kernel.grid_dim[2] not in (None, 1):
            raise ValueError("scalar graph scheduling requires unbatched matmuls")
        for op in find_ops_of_type(module, Op):
            for value in (*op.operands, *op.results):
                if isinstance(value.type, MemTensorType) and value.type.is_dynamic:
                    raise ValueError(
                        "scalar graph scheduling requires static shapes and strides"
                    )
        kernel.grid_dim = (self.blocks, 1, 1)
        kernel.block_dim = (self.threads, 1, 1)
        allocated = {
            cast(int, op.attrs["id"]) for op in find_ops_of_type(module, AllocOp)
        }
        scheduled: list[Op] = []
        for op in list(kernel.body.ops):
            if isinstance(op, (AllocOp, ConstantOp, TileOp, ReshapeOp)):
                scheduled.append(op)
                continue
            if not isinstance(
                op, (ElementwiseOp, MoveOp, ReduceOp, MatmulOp, GatherOp)
            ):
                raise ValueError(f"unsupported scalar graph operation {op.op_id}")
            b = _ScalarBuilder(op, scheduled, allocated)
            destination = op.dst
            dst = cast(MemTensorType, destination.type)

            def coordinates(flat: Value, shape: tuple[int, ...]) -> tuple[Value, ...]:
                return tuple(
                    b.binary(
                        IndexModOp,
                        b.binary(IndexDivOp, flat, math.prod(shape[d + 1 :])),
                        s,
                    )
                    for d, s in enumerate(shape)
                )

            def read(
                value: Value, index: tuple[Value, ...], dtype: Type = F32Type()
            ) -> Value:
                typ = cast(MemTensorType, value.type)
                if isinstance(typ, RmemTensorType):
                    if typ.rank != 0:
                        raise ValueError("scalar graph constants must have rank zero")
                    loaded = value
                else:
                    if not isinstance(typ, GmemTensorType) or typ.rank > len(index):
                        raise ValueError(
                            "scalar graph reads require broadcastable global tensors"
                        )
                    aligned = index[len(index) - typ.rank :] if typ.rank else ()
                    source_index = tuple(
                        0 if s == 1 else c
                        for c, s in zip(aligned, typ.static_shape, strict=True)
                    )
                    loaded = b.scalar(typ.dtype)
                    b.load(loaded, value, source_index)
                if typ.dtype != dtype:
                    widened = b.scalar(dtype)
                    b.arithmetic(widened, [loaded], "convert")
                    loaded = widened
                return loaded

            def write(index: tuple[Value, ...], value: Value) -> None:
                if cast(MemTensorType, value.type).dtype != dst.dtype:
                    rounded = b.scalar(dst.dtype)
                    b.arithmetic(rounded, [value], "convert")
                    value = rounded
                b.store(destination, index, value)

            def pointwise(index: tuple[Value, ...]) -> None:
                if isinstance(op, MoveOp):
                    source_type = cast(MemTensorType, op.src.type)
                    if source_type.sizes != dst.sizes:
                        raise ValueError("scalar graph copies must preserve shape")
                    write(index, read(op.src, index, dst.dtype))
                elif isinstance(op, ElementwiseOp):
                    for source in op.operands[1:]:
                        typ = cast(MemTensorType, source.type)
                        if typ.rank > dst.rank or any(
                            s not in (1, d)
                            for s, d in zip(
                                typ.sizes,
                                dst.sizes[dst.rank - typ.rank :],
                                strict=True,
                            )
                        ):
                            raise ValueError(
                                "scalar graph elementwise sources must broadcast to the destination"
                            )
                    if op.fn in ("convert", "fill"):
                        write(index, read(op.src(0), index, dst.dtype))
                    else:
                        result = b.scalar()
                        b.arithmetic(
                            result,
                            [read(op.src(i), index) for i in range(op.num_srcs)],
                            op.fn,
                        )
                        write(index, result)
                elif isinstance(op, GatherOp):
                    source = cast(GmemTensorType, op.src.type)
                    row_view = b.element(op.indices, (index[0],))
                    row = b.value(LoadIndexOp.build(row_view, source.static_shape[0]))
                    write(index, read(op.src, (row, index[1]), dst.dtype))
                elif isinstance(op, ReduceOp):
                    source = cast(GmemTensorType, op.src.type)
                    if (
                        op.fn != "sum"
                        or source.dtype != F32Type()
                        or dst.dtype != F32Type()
                        or dst.sizes[-1] != 1
                    ):
                        raise ValueError(
                            "scalar graph reductions require full last-axis F32 sums"
                        )
                    acc = b.scalar()
                    b.copy(
                        acc,
                        b.constant(op.init)
                        if op.init is not None
                        else read(op.dst, index),
                    )
                    loop = ForOp.build(0, source.static_shape[-1], 1)
                    b.emit(loop)
                    previous, b.ops = b.ops, loop.body.ops
                    term = read(op.src, (*index[:-1], loop.iv))
                    b.arithmetic(acc, [acc, term], "add")
                    b.ops = previous
                    write(index, acc)
                elif isinstance(op, MatmulOp):
                    if dst.rank != 2 or op.attrs.get("atomic_add"):
                        raise ValueError(
                            "scalar graph matmuls require rank-2 non-atomic outputs"
                        )
                    acc = b.allocate(RmemTensorType.build([1, 1], F32Type()))
                    init = read(op.init if op.init is not None else op.dst, index)
                    b.copy(b.view(acc, 0), init)
                    lhs = b.allocate(RmemTensorType.build([1, 1], F32Type()))
                    rhs = b.allocate(RmemTensorType.build([1, 1], F32Type()))
                    loop = ForOp.build(
                        0, cast(MemTensorType, op.lhs.type).static_shape[1], 1
                    )
                    b.emit(loop)
                    previous, b.ops = b.ops, loop.body.ops
                    b.copy(b.view(lhs, 0), read(op.lhs, (index[0], loop.iv)))
                    b.copy(b.view(rhs, 0), read(op.rhs, (index[1], loop.iv)))
                    b.emit(
                        MatmulOp.build(
                            lhs,
                            rhs,
                            acc,
                            collective={
                                axis: CollectiveLevel.THREAD for axis in CollectiveAxis
                            },
                        )
                    )
                    b.ops = previous
                    result = b.scalar()
                    b.copy(result, b.view(acc, 0))
                    write(index, result)

            if isinstance(op, ReduceOp) and self.blocks > 1:
                self._warp_reduce(b, op)
            else:
                self._parallel(b, dst.static_shape, coordinates, pointwise)
            scheduled.append(
                BarrierOp.build("grid", sync=self.sync)
                if self.blocks > 1
                else BarrierOp.build("block")
            )
        kernel.body.ops = scheduled
        return module

    def _parallel(
        self,
        b: _ScalarBuilder,
        shape: tuple[int, ...],
        coordinates: Callable,
        body: Callable,
        *,
        defer_stores: bool = False,
    ) -> None:
        total = math.prod(shape)
        tid, _ = b.block_thread()
        block = b.value(BlockIdOp.build(CollectiveAxis.X))
        start = b.binary(
            IndexAddOp,
            b.binary(IndexMulOp, block, self.threads * self.vector_width),
            tid,
        )
        stride = self.threads * self.blocks * self.vector_width
        full = total // stride * stride
        loop = ForOp.build(0, full, stride)
        if self.tile_tasks:
            # One guarded vector chunk per task, including the partial tail.
            # A stolen task has no fixed successor on the same CTA.
            loop.ub = total
            loop.attrs["tile_task_count"] = -(-total // (self.threads * self.vector_width))
            loop.attrs["tile_output_loop"] = True
        b.emit(loop)
        previous, b.ops = b.ops, loop.body.ops
        base = b.binary(IndexAddOp, loop.iv, start)
        # Complete chunks own disjoint outputs. Compute them before stores so
        # potentially aliasing global pointers do not serialize the next load.
        # Tail guards keep their loads and stores together inside each branch.
        pending = []
        for offset in range(self.vector_width):
            flat = b.binary(IndexAddOp, base, offset * self.threads)
            if self.tile_tasks:
                b.guard(b.binary(IndexLtOp, flat, total), lambda flat=flat: body(coordinates(flat, shape)))
            elif defer_stores:
                pending.append(body(coordinates(flat, shape), True))
            else:
                body(coordinates(flat, shape))
        for store in pending:
            store()
        b.ops = previous
        if full < total and not self.tile_tasks:
            base = b.binary(IndexAddOp, full, start)
            for offset in range(self.vector_width):
                flat = b.binary(IndexAddOp, base, offset * self.threads)
                b.guard(
                    b.binary(IndexLtOp, flat, total),
                    lambda flat=flat: body(coordinates(flat, shape)),
                )

    def _warp_reduce(self, b: _ScalarBuilder, op: ReduceOp) -> None:
        src, dst = cast(GmemTensorType, op.src.type), cast(GmemTensorType, op.dst.type)
        if src.rank != 2 or dst.static_shape[-1] != 1 or src.dtype != F32Type():
            raise ValueError(
                "persistent reductions require rank-2 full-row F32 reductions"
            )
        tid, _ = b.block_thread()
        block = b.value(BlockIdOp.build(CollectiveAxis.X))
        warps = self.threads // 32
        start = b.binary(
            IndexAddOp,
            b.binary(IndexMulOp, block, warps),
            b.binary(IndexDivOp, tid, 32),
        )
        loop = ForOp.build(0, src.static_shape[0], self.blocks * warps)
        b.emit(loop)
        previous, b.ops = b.ops, loop.body.ops
        row = b.binary(IndexAddOp, loop.iv, start)

        def reduce_row():
            source = b.value(
                TileOp.build(
                    op.src,
                    [row, 0],
                    clone_type(src, {"sizes": [1, src.static_shape[1]]}),
                )
            )
            destination = b.value(
                TileOp.build(op.dst, [row, 0], clone_type(dst, {"sizes": [1, 1]}))
            )
            accumulator = b.allocate(WrmemTensorType.build([1, 1], F32Type()))
            collective = {axis: CollectiveLevel.THREAD_32 for axis in CollectiveAxis}
            b.emit(ReduceOp.build(accumulator, source, op.fn, collective, init=op.init))
            b.emit(MoveOp.build(destination, accumulator, collective=collective))

        b.guard(b.binary(IndexLtOp, row, src.static_shape[0]), reduce_row)
        b.ops = previous

    def map_region(
        self,
        module: ModuleOp,
        effects: list[ElementwiseOp | MoveOp | GatherOp],
        live_destinations: set[Value],
    ) -> ModuleOp:
        """Inline private map producers through affine views and row gathers.

        Each live destination owns a grid-stride loop. Reads of intermediate
        maps become register expressions at the requested coordinates, retaining
        every original dtype conversion. Partial destination writes must have a
        phase boundary before reads of their containing buffer.
        """
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        structural = [op for op in kernel.body.ops if op not in effects]
        generated: list[Op] = []
        allocated = {
            cast(int, op.attrs["id"]) for op in find_ops_of_type(module, AllocOp)
        }
        b = _ScalarBuilder(effects[0], generated, allocated)
        producers = {op.dst: op for op in effects}

        def coordinates(flat: Value, shape: tuple[int, ...]) -> tuple[Value, ...]:
            return tuple(
                b.binary(
                    IndexModOp,
                    b.binary(IndexDivOp, flat, math.prod(shape[d + 1 :])),
                    size,
                )
                for d, size in enumerate(shape)
            )

        def convert(value: Value, dtype: Type) -> Value:
            if cast(MemTensorType, value.type).dtype == dtype:
                return value
            key = (value, dtype.name)
            if key not in conversions:
                result = b.scalar(dtype)
                b.arithmetic(result, [value], "convert")
                conversions[key] = result
            return conversions[key]

        # Only expand a view to reach an in-region producer. Ordinary input
        # views already describe the affine address; decomposing and rebuilding
        # their parent coordinates adds avoidable integer arithmetic.
        def has_producer(value: Value) -> bool:
            while value not in producers and isinstance(
                value.definition, (TileOp, ReshapeOp)
            ):
                value = value.definition.arg
            return value in producers

        reads: dict[tuple[Value, tuple[int | Value, ...]], Value] = {}
        conversions: dict[tuple[Value, str], Value] = {}

        def read(value: Value, index: tuple[int | Value, ...]) -> Value:
            key = (value, index)
            if key not in reads:
                reads[key] = read_uncached(value, index)
            return reads[key]

        def read_uncached(value: Value, index: tuple[int | Value, ...]) -> Value:
            typ = cast(MemTensorType, value.type)
            op = producers.get(value)
            if op is not None:

                def source(src: Value) -> Value:
                    st = cast(MemTensorType, src.type)
                    aligned = index[len(index) - st.rank :] if st.rank else ()
                    return read(
                        src,
                        tuple(
                            0 if size == 1 else i
                            for size, i in zip(st.static_shape, aligned, strict=True)
                        ),
                    )

                if isinstance(op, GatherOp):
                    row_view = b.element(op.indices, (index[0],))
                    row = b.value(
                        LoadIndexOp.build(
                            row_view, cast(MemTensorType, op.src.type).static_shape[0]
                        )
                    )
                    return read(op.src, (row, index[1]))
                if isinstance(op, MoveOp) or op.fn in ("convert", "fill"):
                    return convert(
                        source(op.src if isinstance(op, MoveOp) else op.src(0)),
                        typ.dtype,
                    )
                fused = isinstance(typ.dtype, BF16Type) and op.fn in ("silu", "sigmoid")
                result = b.scalar(typ.dtype if fused else F32Type())
                b.arithmetic(
                    result,
                    [convert(source(src), F32Type()) for src in op.operands[1:]],
                    op.fn,
                )
                return convert(result, typ.dtype)
            definition = value.definition
            if isinstance(definition, TileOp) and has_producer(value):
                parent = tuple(
                    b.binary(IndexAddOp, i, b.binary(IndexMulOp, tile, size))
                    for i, tile, size in zip(
                        index, definition.indices, typ.static_shape, strict=True
                    )
                )
                return read(definition.arg, parent)
            if isinstance(definition, ReshapeOp) and has_producer(value):
                flat: int | Value = 0
                for d, i in enumerate(index):
                    flat = b.binary(
                        IndexAddOp,
                        flat,
                        b.binary(IndexMulOp, i, math.prod(typ.static_shape[d + 1 :])),
                    )
                return read(
                    definition.arg,
                    coordinates(
                        cast(Value, flat),
                        cast(MemTensorType, definition.arg.type).static_shape,
                    ),
                )
            if isinstance(typ, RmemTensorType):
                return value
            result = b.scalar(typ.dtype)
            b.load(result, value, index)
            return result

        # Use source order for reproducible IR; sets of live values are unordered.
        for op in effects:
            if op.dst not in live_destinations:
                continue
            dst = op.dst
            shape = cast(MemTensorType, dst.type).static_shape

            def body(index, deferred=False, dst=dst):
                # Each emitted SIMD element owns its temporaries. Clear between
                # unrolled elements and guarded tails so cached values dominate
                # every use and retain the original rounding at each producer.
                reads.clear()
                conversions.clear()
                result = read(dst, index)
                if deferred:
                    return lambda: b.store(dst, index, result)
                b.store(dst, index, result)

            self._parallel(b, shape, coordinates, body, defer_stores=True)
        kernel.body.ops = [*structural, *generated]
        kernel.grid_dim = (self.blocks, 1, 1)
        kernel.block_dim = (self.threads, 1, 1)
        return module

    def pointwise_region(
        self,
        module: ModuleOp,
        effects: list[ElementwiseOp | MoveOp],
        live_destinations: set[Value],
        shape: tuple[int, ...],
    ) -> ModuleOp:
        """Fuse compatible pointwise effects into one grid-stride register loop."""
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        kernel.grid_dim = (self.blocks, 1, 1)
        kernel.block_dim = (self.threads, 1, 1)
        structural = [op for op in kernel.body.ops if op not in effects]
        generated: list[Op] = []
        allocated = {
            cast(int, op.attrs["id"]) for op in find_ops_of_type(module, AllocOp)
        }
        b = _ScalarBuilder(effects[0], generated, allocated)

        def coordinates(flat: Value, dimensions: tuple[int, ...]) -> tuple[Value, ...]:
            return tuple(
                b.binary(
                    IndexModOp,
                    b.binary(IndexDivOp, flat, math.prod(dimensions[d + 1 :])),
                    extent,
                )
                for d, extent in enumerate(dimensions)
            )

        def body(index: tuple[Value, ...]):
            registers: dict[Value, Value] = {}

            def at(value: Value) -> tuple[int | Value, ...]:
                typ = cast(MemTensorType, value.type)
                aligned = index[len(index) - typ.rank :] if typ.rank else ()
                return tuple(
                    0 if s == 1 else c
                    for c, s in zip(aligned, typ.static_shape, strict=True)
                )

            def read(value: Value, dtype: Type) -> Value:
                typ = cast(MemTensorType, value.type)
                result = registers.get(value)
                if result is None:
                    if isinstance(typ, RmemTensorType):
                        result = value
                    else:
                        result = b.scalar(typ.dtype)
                        b.load(result, value, at(value))
                if typ.dtype != dtype:
                    converted = b.scalar(dtype)
                    b.arithmetic(converted, [result], "convert")
                    result = converted
                return result

            for op in effects:
                typ = cast(MemTensorType, op.dst.type)
                if isinstance(op, MoveOp) or op.fn in ("convert", "fill"):
                    result = read(
                        op.src if isinstance(op, MoveOp) else op.src(0), typ.dtype
                    )
                else:
                    fused = isinstance(typ.dtype, BF16Type) and op.fn in (
                        "sigmoid",
                        "silu",
                    )
                    result = b.scalar(typ.dtype if fused else F32Type())
                    b.arithmetic(
                        result,
                        [read(source, F32Type()) for source in op.operands[1:]],
                        op.fn,
                    )
                    if typ.dtype != F32Type() and not fused:
                        rounded = b.scalar(typ.dtype)
                        b.arithmetic(rounded, [result], "convert")
                        result = rounded
                registers[op.dst] = result
            for destination in live_destinations:
                typ = cast(MemTensorType, destination.type)
                aligned = (1,) * (len(shape) - typ.rank) + typ.static_shape
                duplicated = [
                    c
                    for c, s, d in zip(index, aligned, shape, strict=True)
                    if s == 1 and d != 1
                ]

                def store(destination=destination):
                    b.store(destination, at(destination), registers[destination])

                def owned(axis: int):
                    if axis == len(duplicated):
                        store()
                    else:
                        b.guard(
                            b.binary(IndexEqOp, duplicated[axis], 0),
                            lambda: owned(axis + 1),
                        )

                owned(0)

        self._parallel(b, shape, coordinates, body)
        kernel.body.ops = [*structural, *generated]
        return module

    def row_region(
        self,
        module: ModuleOp,
        effects: list[ElementwiseOp | MoveOp | ReduceOp | GatherOp],
        live_destinations: set[Value],
        shape: tuple[int, ...],
    ) -> ModuleOp:
        """Keep compatible map/reduce row regions in CTA registers.

        Private copies explicitly tagged by the graph scheduler are read through
        their view chains without materializing an intermediate tensor.
        Optional row lookahead carries complete columns of the first read-only
        matrix, loading the next owned row after the first reduction. Partial
        columns keep scalar bounds checks and rounding remains unchanged.
        Narrow rows may use 16 threads with two virtual accumulators retaining
        the 32-thread reduction order; this requires complete CTA row groups.
        """
        if len(shape) != 2:
            raise ValueError("fused row reductions require rank-2 regions")
        rows, width = shape
        epilogue = any(op.attrs.get("row_map_epilogue") is True for op in effects)
        head_shapes = [
            typ.static_shape for op in effects
            if op.attrs.get("row_map_epilogue") is True
            and (typ := cast(MemTensorType, op.dst.type)).rank == 3
            and typ.static_shape[0] * typ.static_shape[1] == rows
        ]
        heads = head_shapes[0][1] if head_shapes else 1
        gathers = {
            effect.dst: effect for effect in effects
            if isinstance(effect, GatherOp) and effect.attrs.get("inline_row_gather") is True
        }
        if any(isinstance(effect, GatherOp) and effect.dst not in gathers for effect in effects):
            raise ValueError("row gathers require graph-scheduler legality checks")
        if set(gathers) & {root_buffer(value) for value in live_destinations}:
            raise ValueError("live row gathers must retain their materialization")
        copies = {
            effect.dst: effect.src for effect in effects
            if isinstance(effect, MoveOp) and effect.attrs.get("inline_row_copy") is True
        }
        if set(copies) & live_destinations:
            raise ValueError("live row copies must retain their materialization")
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        kernel.grid_dim = (self.blocks, 1, 1)
        kernel.block_dim = (self.threads, 1, 1)
        structural = [op for op in kernel.body.ops if op not in effects]
        generated: list[Op] = []
        allocated = {
            cast(int, op.attrs["id"]) for op in find_ops_of_type(module, AllocOp)
        }

        class RowBuilder(_ScalarBuilder):
            def allocate(self, typ):
                previous = self.ops
                self.ops = []
                try:
                    result = super().allocate(typ)
                    generated[0:0] = self.ops
                    return result
                finally:
                    self.ops = previous

        b = RowBuilder(effects[0], generated, allocated)
        tid, _ = b.block_thread()
        lane = b.binary(IndexModOp, tid, 32)
        warp = b.binary(IndexDivOp, tid, 32)
        partials = b.value(
            AllocOp.build_with_fresh_id(
                SmemTensorType.build([self.threads // 32], F32Type())
            )
        )
        block = b.value(BlockIdOp.build(CollectiveAxis.X))
        warp_rows = width <= 1024
        row_threads = self.warp_row_threads if warp_rows else (self.row_threads or self.threads)
        rows_per_cta = self.threads // row_threads
        # A partial subwarp row group would diverge around a full-member-mask
        # shuffle. Keep that case on the existing whole-warp schedule.
        if warp_rows and row_threads < 32 and rows % rows_per_cta:
            raise ValueError("subwarp rows require complete CTA row groups")
        if not warp_rows and rows % rows_per_cta:
            raise ValueError("grouped CTA rows require a complete final row group")
        row_lane = b.binary(IndexModOp, tid, row_threads) if rows_per_cta > 1 else tid
        row_group = b.binary(IndexDivOp, tid, row_threads) if not warp_rows and rows_per_cta > 1 else 0
        start = b.binary(IndexMulOp, block, rows_per_cta)
        if warp_rows:
            start = b.binary(IndexAddOp, start, b.binary(IndexDivOp, tid, row_threads))
        elif rows_per_cta > 1:
            start = b.binary(IndexAddOp, start, row_group)
        vector_width = self.row_vector_width if not warp_rows and not epilogue and width % self.row_vector_width == 0 else 1
        columns = [
            b.binary(IndexAddOp, b.binary(IndexMulOp, row_lane, vector_width), offset + adjacent)
            for offset in range(0, width, row_threads * vector_width)
            for adjacent in range(vector_width)
        ] if vector_width > 1 else [
            b.binary(IndexAddOp, row_lane, offset)
            for offset in range(0, width, row_threads)
        ]
        invariant_ops = (
            _row_invariant_maps(effects, live_destinations)
            if width > 1024 and rows >= self.blocks and rows_per_cta == 1 else []
        )
        written_roots = {root_buffer(op.dst) for op in effects}

        def vectorable(value: Value) -> bool:
            typ = value.type
            return (isinstance(typ, GmemTensorType) and typ.rank in (1, 2)
                    and typ.static_shape[-1] == width and typ.strides[-1] == 1
                    and all(stride % vector_width == 0 for stride in cast(tuple[int, ...], typ.strides)[:-1])
                    and not isinstance(value.definition, (TileOp, ReshapeOp)))

        vector_inputs = list(dict.fromkeys(value for effect in effects for value in effect.operands[1:]
                                         if vectorable(value) and root_buffer(value) not in written_roots))
        # Store in graph order: set iteration can change register allocation
        # throughout the composed kernel between otherwise identical builds.
        ordered_destinations = list(dict.fromkeys(
            effect.dst for effect in effects if effect.dst in live_destinations
        ))
        ordered_destinations.extend(value for value in live_destinations if value not in ordered_destinations)
        vector_outputs = [value for value in ordered_destinations if vectorable(value)]
        aligned = None
        if vector_width > 1:
            pointers = vector_inputs + vector_outputs
            names = [f"ptr{i}" for i in range(len(pointers))]
            template = "mov.b64 ${b64:alignment_bits}, 0;\n"
            for value, name in zip(pointers, names, strict=True):
                alignment = vector_width * cast(BF16Type | F32Type, cast(MemTensorType, value.type).dtype).byte_size
                template += f"and.b64 ${{b64:alignment_part}}, ${{b64:operand.{name}.ptr}}, {alignment - 1};\n"
                template += "or.b64 ${b64:alignment_bits}, ${b64:alignment_bits}, ${b64:alignment_part};\n"
            template += "setp.eq.u64 ${pred:result.aligned.val}, ${b64:alignment_bits}, 0;\n"
            aligned = b.value(PTXTemplateOp.build(pointers, [PredicateType()], template, names, ["aligned"],
                                                description="row pointer alignment", attrs={"side_effect_free": True}))
        # Carry complete columns of the first read-only matrix between rows.
        # Partial columns and warp-owned rows retain their normal load path.
        written = {root_buffer(effect.dst) for effect in effects}
        first_reduction = next((effect for effect in effects if isinstance(effect, ReduceOp)), None)
        sources = [
            value for effect in effects for value in effect.operands[1:]
            if isinstance(value.type, GmemTensorType)
            and value.type.static_shape == shape
            and root_buffer(value) not in written
        ]
        slots = (min(self.row_prefetch_slots, (width // (row_threads * vector_width)) * vector_width)
                 if not warp_rows and rows_per_cta == 1 and first_reduction is not None else 0)
        source = sources[0] if slots and sources else None
        carried = {(source, slot): b.scalar(cast(MemTensorType, source.type).dtype)
                   for slot in range(slots)} if source is not None else {}
        upcoming = {key: b.scalar(cast(MemTensorType, key[0].type).dtype) for key in carried}

        def prime():
            for (value, slot), register in carried.items():
                b.load(register, value, (start, columns[slot]))

        if carried:
            b.guard(b.binary(IndexLtOp, start, rows), prime)
        loop = ForOp.build(0, rows, self.blocks * rows_per_cta)
        # Metadata also identifies baseline row tasks for like-for-like tracing.
        loop.attrs["tile_task_count"] = -(-rows // rows_per_cta)
        loop.attrs["tile_rows_per_task"] = rows_per_cta
        loop.attrs["tile_output_loop"] = True
        b.emit(loop)
        previous, b.ops = b.ops, loop.body.ops
        row = b.binary(IndexAddOp, loop.iv, start)

        def process_row():
            registers: dict[tuple[Value, int | None], Value] = {
                key: value for key, value in carried.items()
            }
            next_row = b.binary(IndexAddOp, row, self.blocks) if carried else None
            next_valid = b.binary(IndexLtOp, next_row, rows) if next_row is not None else None

            def prefetch():
                for (value, slot), register in upcoming.items():
                    b.load(register, value, (cast(Value, next_row), columns[slot]))

            def columnwise(value: Value) -> bool:
                typ = cast(MemTensorType, value.type)
                return bool(typ.rank and typ.static_shape[-1] != 1)

            def coordinates(value: Value, column: Value | None):
                typ = cast(MemTensorType, value.type)
                if typ.rank == 0:
                    return ()
                col: int | Value = 0
                if columnwise(value):
                    assert column is not None
                    col = column
                if epilogue and typ.rank == 3:
                    return (b.binary(IndexDivOp, row, heads), 0 if typ.static_shape[1] == 1 else b.binary(IndexModOp, row, heads), col)
                if epilogue and typ.rank == 2 and typ.static_shape[0] != rows and typ.static_shape[0] != 1:
                    if typ.static_shape[1] == width * heads:
                        return (b.binary(IndexDivOp, row, heads), b.binary(IndexAddOp, b.binary(IndexMulOp, b.binary(IndexModOp, row, heads), width), col))
                    return (b.binary(IndexDivOp, row, heads), col)
                return (
                    (col,)
                    if typ.rank == 1
                    else (0 if typ.static_shape[0] == 1 else row, col)
                )

            def copy_source(value: Value, index: tuple[int | Value, ...]):
                # Expand views only to reach a copy removed by the owning graph
                # scheduler. Affine input views keep their direct address.
                while root_buffer(value) in copies or root_buffer(value) in gathers:
                    definition = value.definition
                    typ = cast(MemTensorType, value.type)
                    if value in gathers:
                        gather = gathers[value]
                        value = gather.src
                        index = (gather_rows[gather], index[1])
                    elif value in copies:
                        value = copies[value]
                    elif isinstance(definition, TileOp):
                        index = tuple(b.binary(IndexAddOp, at, b.binary(IndexMulOp, tile, size))
                                      for at, tile, size in zip(index, definition.indices, typ.static_shape, strict=True))
                        value = definition.arg
                    elif isinstance(definition, ReshapeOp):
                        flat: int | Value = 0
                        for dim, at in enumerate(index):
                            flat = b.binary(IndexAddOp, flat, b.binary(IndexMulOp, at, math.prod(typ.static_shape[dim + 1:])))
                        value = definition.arg
                        parent = cast(MemTensorType, value.type).static_shape
                        index = tuple(b.binary(IndexModOp, b.binary(IndexDivOp, flat, math.prod(parent[dim + 1:])), size)
                                      for dim, size in enumerate(parent))
                    else:
                        raise ValueError("row copy producer requires affine views or reshape")
                return value, index

            def read(value: Value, slot: int | None, dtype: Type) -> Value:
                typ = cast(MemTensorType, value.type)
                key = (value, slot if columnwise(value) else None)
                result = registers.get(key)
                if epilogue and result is None and slot is not None:
                    parent = value
                    shift = 0
                    while isinstance(parent.definition, (TileOp, ReshapeOp)):
                        definition = parent.definition
                        if isinstance(definition, TileOp):
                            assert all(isinstance(index, int) for index in definition.indices)
                            assert not any(definition.indices[:-1])
                            shift += cast(int, definition.indices[-1]) * cast(MemTensorType, parent.type).static_shape[-1]
                        parent = definition.arg
                        if (parent, slot + shift // row_threads) in registers:
                            assert shift % row_threads == 0
                            return read(parent, slot + shift // row_threads, dtype)
                if result is None:
                    if isinstance(typ, RmemTensorType):
                        result = value
                    else:
                        result = b.scalar(typ.dtype)
                        source, source_index = copy_source(
                            value, coordinates(value, columns[slot] if slot is not None else None)
                        )
                        b.load(result, source, source_index)
                    registers[key] = result
                if typ.dtype != dtype:
                    converted = b.scalar(dtype)
                    b.arithmetic(converted, [result], "convert")
                    result = converted
                return result

            def slots_for(value: Value) -> list[int | None]:
                typ = cast(MemTensorType, value.type)
                if not columnwise(value):
                    return [None]
                count = len(columns)
                if epilogue and typ.rank == 3:
                    count = min(count, (typ.static_shape[-1] + row_threads - 1) // row_threads)
                return list(range(count))

            def preload(effect: ElementwiseOp | MoveOp | ReduceOp | GatherOp):
                if vector_width == 1:
                    return
                for value in dict.fromkeys(effect.operands[1:]):
                    if value not in vector_inputs or (value, 0) in registers:
                        continue
                    typ = cast(GmemTensorType, value.type)
                    sizes = [1] * (typ.rank - 1) + [vector_width]
                    pairs = []
                    for first in range(0, len(columns), vector_width):
                        vector = b.allocate(RmemTensorType.build(sizes, typ.dtype))
                        index: list[int | Value] = list(coordinates(value, columns[first]))
                        index[-1] = b.binary(IndexDivOp, index[-1], vector_width)
                        tile = b.value(TileOp.build(value, index, clone_type(typ, {"sizes": sizes})))
                        pairs.append((first, vector, tile))
                        for adjacent in range(vector_width):
                            registers[(value, first + adjacent)] = b.view(vector, adjacent)
                    def load_all(load_width: int):
                        for first, vector, tile in pairs:
                            b.guard(b.binary(IndexLtOp, columns[first], width),
                                    lambda vector=vector, tile=tile: b.copy(vector, tile, {
                                        "vector_width": load_width,
                                        "alignment": vector_width * cast(BF16Type | F32Type, typ.dtype).byte_size,
                                        "src_tiling_scheme": b.scheme,
                                    }))
                    b.branch(cast(Value, aligned), lambda: load_all(vector_width), lambda: load_all(1))

            def compute_map(effect: ElementwiseOp | MoveOp, slot: int | None):
                typ = cast(MemTensorType, effect.dst.type)
                if isinstance(effect, MoveOp) or effect.fn in (
                    "convert",
                    "fill",
                ):
                    result = read(
                        effect.src
                        if isinstance(effect, MoveOp)
                        else effect.src(0),
                        slot,
                        typ.dtype,
                    )
                else:
                    # Preserve BF16 residual rounding with direct arithmetic.
                    # Narrow epilogue add/sub avoid widening and rounding
                    # intermediates. Multiplication retains F32 there: native
                    # BF16 mul caused spills in the composed forward kernel.
                    native_bf16 = (
                        (not epilogue or (warp_rows and effect.fn in ("add", "sub")))
                        and isinstance(typ.dtype, BF16Type)
                        and effect.fn in ("add", "sub", "mul")
                        and all(cast(MemTensorType, value.type).dtype == typ.dtype
                                for value in effect.operands[1:])
                    )
                    fused = native_bf16 or (
                        isinstance(typ.dtype, BF16Type)
                        and effect.fn in ("sigmoid", "silu")
                    )
                    result = b.scalar(typ.dtype if fused else F32Type())
                    b.arithmetic(
                        result,
                        [
                            read(source, slot, typ.dtype if native_bf16 else F32Type())
                            for source in effect.operands[1:]
                        ],
                        effect.fn,
                    )
                    if typ.dtype != F32Type() and not fused:
                        rounded = b.scalar(typ.dtype)
                        b.arithmetic(rounded, [result], "convert")
                        result = rounded
                registers[(effect.dst, slot)] = result

            # Build the invariant expressions with the same rounding/read rules,
            # then place their code before the loop. Allocations remain at scope.
            gather_rows = {}
            for gather in gathers.values():
                logical_row = cast(tuple[int | Value, ...], coordinates(gather.dst, columns[0]))[0]
                view = b.element(gather.indices, (logical_row,))
                gather_rows[gather] = b.value(LoadIndexOp.build(
                    view, cast(MemTensorType, gather.src.type).static_shape[0]
                ))
            body_ops = b.ops
            invariant_code: list[Op] = []
            b.ops = invariant_code
            for effect in invariant_ops:
                preload(effect)
                for slot in slots_for(effect.dst):
                    if slot is None:
                        compute_map(effect, slot)
                    else:
                        b.guard(b.binary(IndexLtOp, columns[slot], width),
                                lambda slot=slot, effect=effect: compute_map(effect, slot))
            at = generated.index(loop)
            generated[at:at] = invariant_code
            b.ops = body_ops

            for effect in effects:
                if effect in invariant_ops or isinstance(effect, GatherOp):
                    continue
                if isinstance(effect, MoveOp) and effect.attrs.get("inline_row_copy") is True:
                    continue
                preload(effect)
                if isinstance(effect, ReduceOp):
                    if (
                        cast(MemTensorType, effect.src.type).static_shape != shape
                        or effect.fn != "sum"
                        or effect.init != 0
                    ):
                        raise ValueError(
                            "fused row reduction requires an overwriting full-row sum"
                        )
                    accumulator = b.scalar()
                    b.copy(accumulator, b.constant(0.0))
                    # Preserve the original 32-lane accumulation and butterfly
                    # order with two virtual lanes per physical subwarp lane.
                    # Merge the virtual xor-16 pair before the xor-8..1 shuffles.
                    virtual_accumulators = [accumulator]
                    if warp_rows and row_threads < 32:
                        extra = b.scalar()
                        b.copy(extra, b.constant(0.0))
                        virtual_accumulators.append(extra)
                    for slot, column in enumerate(columns):
                        def accumulate(slot=slot, effect=effect):
                            value = read(effect.src, slot, F32Type())
                            local_accumulator = virtual_accumulators[slot % len(virtual_accumulators)]
                            b.arithmetic(local_accumulator, [local_accumulator, value], "add")
                        b.guard(b.binary(IndexLtOp, column, width), accumulate)
                    if len(virtual_accumulators) == 2:
                        b.arithmetic(accumulator, virtual_accumulators, "add")
                    b.shuffle_fold(accumulator, "add", row_threads if warp_rows else 32)
                    if warp_rows:
                        total = accumulator
                    else:
                        b.guard(
                            b.binary(IndexEqOp, lane, 0),
                            lambda: b.store(partials, (warp,), accumulator),
                        )
                        b.emit(BarrierOp.build("block"))
                        total = b.scalar()
                        b.copy(total, b.constant(0.0))
                        partial_start = b.binary(IndexMulOp, row_group, row_threads // 32) if rows_per_cta > 1 else 0
                        for index in range(row_threads // 32):
                            value = b.scalar()
                            at = b.binary(IndexAddOp, partial_start, index) if rows_per_cta > 1 else index
                            b.load(value, partials, (at,))
                            b.arithmetic(total, [total, value], "add")
                        b.emit(BarrierOp.build("block"))
                    registers[(effect.dst, None)] = total
                    if carried and effect is first_reduction:
                        b.guard(cast(Value, next_valid), prefetch)
                    continue
                assert isinstance(effect, (ElementwiseOp, MoveOp))
                slots = slots_for(effect.dst)
                for slot in slots:

                    if slot is None:
                        compute_map(effect, slot)
                    else:
                        b.guard(b.binary(IndexLtOp, columns[slot], width),
                                lambda slot=slot, effect=effect: compute_map(effect, slot))
            for destination in ordered_destinations:
                typ = cast(MemTensorType, destination.type)
                if (vector_width > 1 and vectorable(destination)
                        and typ.rank == 2 and typ.static_shape == shape):
                    sizes = [1, vector_width]
                    def store_all(store_width: int):
                        for first in range(0, len(columns), vector_width):
                            def vector_store(first=first):
                                vector = b.allocate(RmemTensorType.build(sizes, typ.dtype))
                                for adjacent in range(vector_width):
                                    value = read(destination, first + adjacent, typ.dtype)
                                    b.copy(b.view(vector, adjacent), value)
                                index: list[int | Value] = [row, b.binary(IndexDivOp, columns[first], vector_width)]
                                tile = b.value(TileOp.build(destination, index, clone_type(typ, {"sizes": sizes})))
                                b.copy(tile, vector, {"vector_width": store_width,
                                                    "alignment": vector_width * cast(BF16Type | F32Type, typ.dtype).byte_size,
                                                    "dst_tiling_scheme": b.scheme})
                            b.guard(b.binary(IndexLtOp, columns[first], width), vector_store)
                    b.branch(cast(Value, aligned), lambda: store_all(vector_width), lambda: store_all(1))
                    continue
                for slot in slots_for(destination):

                    def store(slot=slot):
                        b.store(
                            destination,
                            coordinates(
                                destination, columns[slot] if slot is not None else None
                            ),
                            registers[(destination, slot)],
                        )

                    if slot is None:
                        b.guard(
                            b.binary(IndexEqOp, row_lane, 0), store
                        )
                    else:
                        b.guard(b.binary(IndexLtOp, columns[slot], width), store)

            def advance():
                for key, current in carried.items():
                    b.copy(current, upcoming[key])

            if carried:
                b.guard(cast(Value, next_valid), advance)

        b.guard(b.binary(IndexLtOp, row, rows), process_row)
        b.ops = previous
        if self.tile_tasks:
            for op in generated[:generated.index(loop)]:
                op.attrs["tile_row_preamble"] = True
        kernel.body.ops = [*structural, *generated]
        return module
