"""Safety boundaries for the matrix/refill reorder and physical scheduling."""

import pytest

from ayla import gpu_dialect as g
from ayla.base.ir import Block, Value
from ayla.passes.pipeline_matrix_refills import PipelineMatrixRefills
from tests.passes.test_lower_to_sass import (
    COLLECTIVE,
)


def refill_example(lowered=True):
    shared = [
        g.AllocOp.build_with_fresh_id(g.SmemTensorType.build([64, 64], g.F16Type()))
        for _ in range(4)
    ]
    acc = g.AllocOp.build_with_fresh_id(
        g.WgrmemTensorType.build(
            [64, 64], g.F32Type(), layout="wgmma_layout_0", warps=4
        )
    )
    mma_lock = Value("matrix", g.LockType.build("wgmma"))
    a_lock, b_lock = [Value(name, g.LockType.build("mbar")) for name in ("a", "b")]
    gmem_type = g.GmemTensorType.build([64, 64], g.F16Type())
    desc = Value(
        "desc", g.TmaDescType.build(0, gmem_type, [64, 64]) if lowered else gmem_type
    )

    def tma(target, lock):
        return g.AsyncOp.build(
            lock,
            COLLECTIVE,
            Block(
                ops=[
                    g.TmaMoveOp.build(
                        target.result(), desc, indices=[], collective=COLLECTIVE
                    )
                    if lowered
                    else g.MoveOp.build(
                        target.result(),
                        desc,
                        collective=COLLECTIVE,
                        attrs={"tma": True},
                    )
                ]
            ),
        )

    def mma(left, right):
        return g.AsyncOp.build(
            mma_lock,
            COLLECTIVE,
            Block(
                ops=[
                    g.MatmulOp.build(
                        left.result(),
                        right.result(),
                        acc.result(),
                        COLLECTIVE,
                        attrs={"accumulate": True},
                    )
                ]
            ),
        )

    prefetched = tma(shared[2], b_lock)
    first = mma(*shared[:2])
    wait = g.AwaitOp.build(mma_lock)
    refill = tma(shared[0], a_lock)
    input_wait = g.AwaitOp.build(b_lock)
    next_matrix = mma(*shared[2:])
    final_wait = g.AwaitOp.build(mma_lock)
    body = Block(
        arguments=[desc, mma_lock, a_lock, b_lock],
        ops=[
            *shared,
            acc,
            prefetched,
            first,
            wait,
            refill,
            input_wait,
            next_matrix,
            final_wait,
        ],
    )
    module = g.ModuleOp.build(
        Block(
            ops=[
                g.KernelOp.build(
                    "proof",
                    body,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                    num_warp_groups=1,
                )
            ]
        )
    )
    return module, body, wait, refill, input_wait, next_matrix, final_wait


@pytest.mark.parametrize("lowered", [False, True])
def test_refill_reorder_keeps_old_input_completion_and_mma_order(lowered):
    module, body, wait, refill, input_wait, matrix, final = refill_example(lowered)
    reorder = PipelineMatrixRefills()
    reorder.run(module)
    assert reorder.moved == 1
    assert (
        body.ops.index(input_wait)
        < body.ops.index(matrix)
        < body.ops.index(wait)
        < body.ops.index(refill)
    )
    assert wait.pending_groups == 1 and final.is_full_wait
    assert body.ops[-1] is final
    second = PipelineMatrixRefills()
    second.run(module)
    assert second.moved == 0


@pytest.mark.parametrize(
    "unsafe",
    [
        "input_alias",
        "unissued_input",
        "overwrite",
        "shape",
        "named_alias",
        "observer",
        "publication",
        "ssa",
        "nested_ssa",
        "control",
        "queue8",
        "empty_completion",
        "multiple_warp_groups",
    ],
)
@pytest.mark.parametrize("lowered", [False, True])
def test_refill_reorder_rejects_unsafe_crossings(unsafe, lowered):
    module, body, wait, refill, input_wait, matrix, final = refill_example(lowered)
    mm = matrix.body.ops[0]
    if unsafe == "input_alias":
        refill.body.ops[0].operands[0] = mm.lhs
    elif unsafe == "unissued_input":
        body.ops = [
            op
            for op in body.ops
            if not (isinstance(op, g.AsyncOp) and op.lock is input_wait.lock)
        ]
    elif unsafe == "overwrite":
        mm.attrs["accumulate"] = False
    elif unsafe == "shape":
        mm.operands[2] = Value("different_accumulator", mm.dst.type)
    elif unsafe == "named_alias":
        body.ops[0].attrs["reuse_group"] = "shared"
    elif unsafe == "observer":
        body.ops.insert(
            body.ops.index(refill), g.PTXTemplateOp.build([], [], "bar.sync 0;")
        )
    elif unsafe == "publication":
        body.ops.insert(body.ops.index(refill), g.BarrierOp.build("block"))
    elif unsafe == "ssa":
        input_wait.lock = Value("not_dominating", input_wait.lock.type)
    elif unsafe == "nested_ssa":
        # A new matrix view created below the insertion point cannot move.
        view = g.ReinterpretOp.build(mm.lhs, mm.lhs.type)
        body.ops.insert(body.ops.index(refill), view)
        mm.operands[0] = view.result()
    elif unsafe == "control":
        # An unknown region means the outstanding matrix queue is unknown.
        body.ops.insert(
            body.ops.index(wait),
            g.IfOp(operands=[], results=[], attrs={}, blocks=[Block()]),
        )
    elif unsafe == "queue8":
        first = body.ops[body.ops.index(wait) - 1]
        body.ops[body.ops.index(wait) : body.ops.index(wait)] = [
            g.AsyncOp.build(first.lock, COLLECTIVE, Block(ops=list(first.body.ops)))
            for _ in range(7)
        ]
    elif unsafe == "empty_completion":
        wait.attrs["pending"] = 1
    else:
        module.body.ops[0].attrs["num_warp_groups"] = 2
    before = list(body.ops)
    reorder = PipelineMatrixRefills()
    reorder.run(module)
    assert reorder.moved == 0 and body.ops == before
    assert wait.pending_groups == (1 if unsafe == "empty_completion" else 0)


@pytest.mark.parametrize(
    "unsafe", ["synchronous_move", "shared_source", "global_destination"]
)
def test_refill_reorder_rejects_other_library_moves(unsafe):
    module, body, wait, refill, input_wait, matrix, final = refill_example(False)
    move = refill.body.ops[0]
    if unsafe == "synchronous_move":
        move.attrs.pop("tma")
    elif unsafe == "shared_source":
        move.operands[1] = matrix.body.ops[0].lhs
    else:
        move.operands[0] = move.src
    before = list(body.ops)
    reorder = PipelineMatrixRefills()
    reorder.run(module)
    assert reorder.moved == 0 and body.ops == before
