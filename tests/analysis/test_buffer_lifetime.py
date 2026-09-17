"""Tests for buffer lifetime analysis."""

import pytest

from ayla.base.ir import Block, Op, Value
from ayla.gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F32Type,
    ForOp,
    GmemTensorType,
    IfOp,
    IndexConstantOp,
    KernelOp,
    LockType,
    ModuleOp,
    MoveOp,
    PTXTemplateOp,
    ReinterpretOp,
    SelectOp,
    SmemTensorType,
    gpu_dialect,
)
from ayla.analysis.buffer_lifetime import (
    BufferLifetimeAnalysis,
    _BufferAccessAnalysis,
)


def smem_type() -> SmemTensorType:
    return SmemTensorType.build([16], F32Type())


THREAD_COLLECTIVE = {
    axis: CollectiveLevel.THREAD for axis in CollectiveAxis
}


def test_buffer_is_live_from_first_to_last_access():
    alloc = AllocOp.build(smem_type(), id=0)
    before = BarrierOp.build()
    first_access = MoveOp.build(
        alloc.result(), alloc.result(), collective=THREAD_COLLECTIVE,
    )
    between = BarrierOp.build()
    last_access = MoveOp.build(
        alloc.result(), alloc.result(), collective=THREAD_COLLECTIVE,
    )
    after = BarrierOp.build()
    module = ModuleOp.build(Block(ops=[
        alloc, before, first_access, between, last_access, after,
    ]))

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(module)

    assert analysis.live_in(alloc).buffers == frozenset()
    assert analysis.live_in(before).buffers == frozenset()
    assert analysis.live_in(first_access).buffers == frozenset()
    assert analysis.live_out(first_access).buffers == frozenset({alloc.result()})
    assert analysis.live_in(between).buffers == frozenset({alloc.result()})
    assert analysis.live_in(last_access).buffers == frozenset({alloc.result()})
    assert analysis.live_out(last_access).buffers == frozenset()
    assert analysis.live_in(after).buffers == frozenset()


def test_initialized_allocation_is_the_first_access():
    alloc = AllocOp.build(smem_type(), id=0, init=0)
    between = BarrierOp.build()
    access = MoveOp.build(
        alloc.result(), alloc.result(), collective=THREAD_COLLECTIVE,
    )
    module = ModuleOp.build(Block(ops=[alloc, between, access]))

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(module)

    assert analysis.live_in(alloc).buffers == frozenset()
    assert analysis.live_out(alloc).buffers == frozenset({alloc.result()})
    assert analysis.live_in(between).buffers == frozenset({alloc.result()})


def test_kernel_arguments_extend_to_their_abi_boundaries():
    input_buffer = Value(name="input", type=GmemTensorType.build([16], F32Type()))
    output_buffer = Value(
        name="output", type=GmemTensorType.build([16], F32Type()),
    )
    unused_buffer = Value(
        name="unused", type=GmemTensorType.build([16], F32Type()),
    )
    temporary = AllocOp.build(smem_type(), id=0)
    before_read = IndexConstantOp.build(0)
    read = MoveOp.build(
        temporary.result(), input_buffer, collective=THREAD_COLLECTIVE,
    )
    between = IndexConstantOp.build(0)
    write = MoveOp.build(
        output_buffer, temporary.result(), collective=THREAD_COLLECTIVE,
    )
    after_write = IndexConstantOp.build(0)
    kernel = KernelOp.build(
        "test",
        Block(
            arguments=[input_buffer, output_buffer, unused_buffer],
            ops=[temporary, before_read, read, between, write, after_write],
        ),
    )
    module = ModuleOp.build(Block(ops=[kernel]))

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(module)

    assert analysis.live_in(before_read).buffers == frozenset({input_buffer})
    assert analysis.live_in(between).buffers == frozenset({temporary.result()})
    assert analysis.live_in(write).buffers == frozenset({temporary.result()})
    assert analysis.live_out(write).buffers == frozenset({output_buffer})
    assert analysis.live_in(after_write).buffers == frozenset({output_buffer})
    assert all(
        unused_buffer not in analysis.live_in(op).buffers
        for op in kernel.body.ops
    )


def test_in_place_handoff_lifetimes_do_not_intersect():
    input_buffer = Value(name="input", type=GmemTensorType.build([16], F32Type()))
    output_buffer = Value(
        name="output", type=GmemTensorType.build([16], F32Type()),
    )
    temporary = AllocOp.build(output_buffer.type, id=0)
    produce = MoveOp.build(
        temporary.result(), input_buffer, collective=THREAD_COLLECTIVE,
    )
    handoff = ElementwiseOp.build(
        output_buffer, [temporary.result()], fn="sigmoid",
    )
    kernel = KernelOp.build(
        "test",
        Block(
            arguments=[input_buffer, output_buffer],
            ops=[temporary, produce, handoff],
        ),
    )

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(kernel)

    assert not (
        analysis.lifetime(temporary.result())
        & analysis.lifetime(output_buffer)
    )


def test_views_resolve_to_their_physical_buffer():
    alloc = AllocOp.build(smem_type(), id=0)
    view = ReinterpretOp.build(alloc.result(), smem_type())
    first_access = MoveOp.build(
        view.result(), view.result(), collective=THREAD_COLLECTIVE,
    )
    between = BarrierOp.build()
    last_access = MoveOp.build(
        view.result(), view.result(), collective=THREAD_COLLECTIVE,
    )
    module = ModuleOp.build(Block(ops=[
        alloc, view, first_access, between, last_access,
    ]))

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(module)

    assert analysis.live_in(view).buffers == frozenset()
    assert analysis.live_out(first_access).buffers == frozenset({alloc.result()})
    assert analysis.live_in(between).buffers == frozenset({alloc.result()})
    assert analysis.live_in(last_access).buffers == frozenset({alloc.result()})


def test_select_keeps_every_possible_buffer_live():
    lhs = AllocOp.build(smem_type(), id=0)
    rhs = AllocOp.build(smem_type(), id=1)
    index = Value(name="index", type=F32Type())
    select = SelectOp.build(index, [lhs.result(), rhs.result()])
    first_access = MoveOp.build(
        select.result(), select.result(), collective=THREAD_COLLECTIVE,
    )
    between = BarrierOp.build()
    last_access = MoveOp.build(
        select.result(), select.result(), collective=THREAD_COLLECTIVE,
    )
    module = ModuleOp.build(Block(ops=[
        lhs, rhs, select, first_access, between, last_access,
    ]))

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(module)

    assert analysis.live_in(select).buffers == frozenset()
    assert analysis.live_in(between).buffers == frozenset({
        lhs.result(), rhs.result(),
    })
    assert analysis.live_in(rhs).buffers == frozenset()


def test_buffer_stays_live_until_its_async_await():
    lock = AllocOp.build(LockType.build(kind="mbar"), id=0)
    alloc = AllocOp.build(smem_type(), id=1)
    access = MoveOp.build(
        alloc.result(), alloc.result(), collective=THREAD_COLLECTIVE,
    )
    async_op = AsyncOp.build(
        lock.result(), collective=THREAD_COLLECTIVE, body=Block(ops=[access]),
    )
    between = BarrierOp.build()
    await_op = AwaitOp.build(lock.result())
    module = ModuleOp.build(Block(ops=[lock, alloc, async_op, between, await_op]))

    analysis = BufferLifetimeAnalysis()
    analysis.analyze(module)

    assert analysis.live_in(between).buffers == frozenset({alloc.result()})
    assert analysis.live_in(await_op).buffers == frozenset({alloc.result()})


def test_unmodelled_memory_effect_fails_loudly():
    alloc = AllocOp.build(smem_type(), id=0)
    ptx = PTXTemplateOp.build(
        operands=[alloc.result()], result_types=[], template="",
    )
    module = ModuleOp.build(Block(ops=[alloc, ptx]))

    analysis = BufferLifetimeAnalysis()
    with pytest.raises(
        NotImplementedError,
        match="unmodelled memory effects for PTXTemplateOp",
    ):
        analysis.analyze(module)


def test_every_gpu_leaf_op_has_explicit_transfer_semantics():
    structured_ops = {ModuleOp, KernelOp, ForOp, IfOp}
    dialect_ops = set(gpu_dialect._ops.values()) - {Op}

    assert set(_BufferAccessAnalysis._transfers) | structured_ops == dialect_ops
