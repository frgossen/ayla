"""Physical bank rotation, event epochs, compact code and real reuse."""

import math
import pytest
from ayla.base.ir import clone_op, find_ancestor, find_ops_of_type, iter_ops
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    AsyncOp,
    AwaitOp,
    ForOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    LockViewOp,
    OperandPipelineOp,
    TileOp,
    physical_layout,
)
from ayla.passes.pipeline_rolled_k import LowerRolledK
from ayla.ptx_codegen import make_ptx_kernel
from tests.testing_utils import assert_dominance, check_gemm


def build(**kwargs):
    cfg = dict(
        m=256,
        n=128,
        k=1040,
        dtype="f16",
        output_m_tile=64,
        output_n_tile=64,
        operand_k_tile=64,
        operand_stages=4,
        cta_group=2,
        output_store="stmatrix",
        output_layout="tcgen05_16x256b",
        rolled_operand_pipeline=True,
        allow_partial_tiles=True,
    )
    cfg.update(kwargs)
    return blackwell_tcgen05_gemm_module(**cfg)


@pytest.mark.parametrize(
    "depth,k", [(2, 192), (3, 1040), (4, 1040), (5, 784), (8, 8192), (16, 8192)]
)
@pytest.mark.parametrize("kt", [64, 128, 256])
def test_physical_banks_and_event_epochs(depth, k, kt):
    k = max(kt * 3, math.ceil(k / kt) * kt)
    if depth > math.ceil(k / kt) or (64 + 32) * kt * 2 * depth > 227328:
        pytest.skip("bank capacity or reduction too small for this depth")
    mod = build(operand_stages=depth, k=k, operand_k_tile=kt)
    assert_dominance(mod.body)
    region = find_ops_of_type(mod, OperandPipelineOp)[0]
    lowered = LowerRolledK().run(clone_op(mod, {}))
    assert_dominance(lowered.body)
    # Interpret the expanded region only. Completion is immediate, while
    # epochs and physical contents are checked independently of issue labels.
    ops, _ = LowerRolledK().lower_pipeline(region)
    values = {}
    data = {}
    loads = []
    mmas = []
    issued = {}
    observed = {}

    def coordinates(value):
        result = [0, 0]
        while isinstance(value.definition, TileOp):
            tile = value.definition
            for i, index in enumerate(tile.indices):
                result[i] += (
                    index
                    if isinstance(index, int)
                    else values.get(index, 0)
                    if i == 0
                    else values[index]
                ) * value.type.sizes[i]
            value = tile.arg
        return value, result

    def walk(ops):
        for op in ops:
            if isinstance(op, IndexConstantOp):
                values[op.result()] = op.value
            elif isinstance(
                op, (IndexAddOp, IndexDivOp, IndexLtOp, IndexModOp, IndexMulOp)
            ):
                a, b = values[op.lhs], values[op.rhs]
                values[op.result()] = {
                    IndexAddOp: lambda: a + b,
                    IndexDivOp: lambda: a // b,
                    IndexLtOp: lambda: a < b,
                    IndexModOp: lambda: a % b,
                    IndexMulOp: lambda: a * b,
                }[type(op)]()
            elif isinstance(op, LockViewOp):
                values[op.result()] = (
                    op.operands[0],
                    values[op.operands[1]],
                    values[op.operands[2]],
                )
            elif isinstance(op, ForOp):
                for i in range(op.lb, op.ub, op.step):
                    values[op.iv] = i
                    walk(op.body.ops)
            elif isinstance(op, IfOp):
                if values[op.condition]:
                    walk(op.body.ops)
            elif isinstance(op, AsyncOp):
                arena, bank, phase = values[op.lock]
                if op.lock.type.kind == "mbar":
                    logical = []
                    for move in op.body.ops:
                        root, dst = coordinates(move.dst)
                        _, src = coordinates(move.src)
                        blocked = physical_layout(root.type) is not None
                        axis = 1 if blocked else 0
                        assert dst[axis] // move.dst.type.sizes[axis] == bank
                        data[(root, bank)] = src[1] // kt
                        logical.append(src[1] // kt)
                    assert len(set(logical)) == 1
                    logical = logical[0]
                    assert bank == logical % depth
                    assert phase == 0  # issue does not consume expected phase
                    loads.append(logical)
                else:
                    logical = len(mmas) + 1
                    assert bank == logical % depth
                    assert phase == 0  # issue does not consume expected phase
                    for mma in op.body.ops:
                        for operand in (mma.lhs, mma.rhs):
                            root, coord = coordinates(operand)
                            axis = 1 if physical_layout(root.type) is not None else 0
                            extent = root.type.sizes[axis] // depth
                            actual = coord[axis] // extent
                            assert actual == bank, (
                                "MMA retained a hoisted bank-zero view"
                            )
                            assert data[(root, actual)] == logical
                    mmas.append(logical)
                key = (arena, bank)
                actual_phase = ((logical - 1) // depth) % 2
                if key in issued:
                    assert observed.get(key) == issued[key], (
                        "event rearmed before old observation"
                    )
                    assert actual_phase == 1 - issued[key]
                issued[key] = actual_phase
            elif isinstance(op, AwaitOp):
                arena, bank, phase = values[op.lock]
                assert issued[(arena, bank)] == phase
                observed[(arena, bank)] = phase

    walk(ops)
    assert loads == mmas == list(range(1, math.ceil(k / kt)))


@pytest.mark.parametrize("kt,depths", [(64, (4, 8, 16)), (128, (2, 4, 8))])
def test_loop_body_does_not_scale_with_depth(kt, depths):
    sizes = []
    for depth in depths:
        mod = LowerRolledK().run(build(k=8192, operand_stages=depth, operand_k_tile=kt))
        loops = {op.attrs.get("label"): op for op in find_ops_of_type(mod, ForOp)}
        main = loops["rolled_k_main"]
        assert main.step == 1 and main.trip_count == 8192 // kt - 1
        assert len(find_ops_of_type(main, AsyncOp)) == 2
        assert len(find_ops_of_type(loops["rolled_k_fill"], AsyncOp)) == 1
        sizes.append(len(list(iter_ops(main))))
    assert len(set(sizes)) == 1


@pytest.mark.parametrize("extra", [dict(cta_group=1), {},
                                  dict(cluster_n_ctas=2, cluster_dim=(4, 1, 1))])
def test_expanded_operand_loops_share_issuer_guard(extra):
    from ayla.ptx_codegen import PtxLoweringPipeline

    lowered = PtxLoweringPipeline().run(build(k=4096, operand_stages=8, **extra))
    loops = {op.attrs.get("label"): op for op in find_ops_of_type(lowered, ForOp)}
    guards = []
    for label in ("rolled_k_fill", "rolled_k_main"):
        loop = loops[label]
        assert loop.attrs["unroll"] is False
        guard = find_ancestor(loop, lambda op: isinstance(op, IfOp)
                              and op.attrs.get("participant_warps") == [0])
        assert guard is not None, "nonissuers must skip the bank loop's arithmetic and guards"
        guards.append(guard)
    assert guards[0] is guards[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(m=129, n=72, k=1040, operand_stages=3),
        dict(
            m=129, n=72, k=784, operand_stages=5, dtype="bf16", tmem_writeback_warps=8
        ),
        dict(
            m=256,
            n=128,
            k=784,
            operand_stages=6,
            output_m_tile=128,
            output_n_tile=128,
            output_slices=4,
        ),
        dict(m=129, n=72, k=1040, operand_stages=3, cta_group=1),
        dict(m=129, n=72, k=8192, operand_stages=16),
        dict(m=256, n=128, k=1152, operand_k_tile=128, operand_stages=3),
        dict(m=129, n=72, k=2304, operand_k_tile=256, operand_stages=3,
             cta_group=1, dtype="bf16"),
        dict(m=129, n=72, k=8192, operand_k_tile=128, operand_stages=6,
             cta_group=1, operand_move_attrs={"lhs":{"issuer_warp":1},"rhs":{"issuer_warp":1}}),
        dict(m=129, n=72, k=2304, operand_k_tile=256, operand_stages=3,
             cta_group=1, dtype="bf16", operand_move_attrs={"lhs":{"issuer_warp":2},"rhs":{"issuer_warp":2}}),
    ],
)
def test_gpu_reuse_and_partial_tiles(kwargs):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    launch = make_ptx_launcher(make_ptx_kernel(build(**kwargs)))
    m, n, k = (kwargs[x] for x in ("m", "n", "k"))
    check_gemm(launch, m, n, k, kwargs.get("dtype", "f16"), graph=True)


def test_cleanup_preserves_distinct_event_observations():
    from ayla.base.ir import Block
    from ayla.passes.cse import _cse_block
    from ayla.passes.dce import DCE

    region = find_ops_of_type(build(k=8192, operand_stages=8), OperandPipelineOp)[0]
    ops, _ = LowerRolledK().lower_pipeline(region)
    body = Block(ops=ops)
    before = len(
        [op for root in body.ops for op in iter_ops(root) if isinstance(op, AwaitOp)]
    )
    _cse_block(body, {})
    DCE().run_block(body)
    after = len(
        [op for root in body.ops for op in iter_ops(root) if isinstance(op, AwaitOp)]
    )
    assert before == after == 3


def test_rejects_incomplete_iteration():
    from ayla.passes.pipeline_rolled_k import _parts

    region = find_ops_of_type(build(), OperandPipelineOp)[0]
    region.body.ops = [op for op in region.body.ops if not isinstance(op, AwaitOp)]
    with pytest.raises(ValueError, match="completed serial iteration"):
        _parts(region)


def test_rejects_nonserial_issue_order():
    from ayla.passes.pipeline_rolled_k import _parts

    region = find_ops_of_type(build(), OperandPipelineOp)[0]
    waits = find_ops_of_type(region, AwaitOp)
    region.body.ops.remove(waits[0])
    region.body.ops.append(waits[0])
    with pytest.raises(ValueError, match="ordering"):
        _parts(region)


def test_blocked_physical_banks_extend_k():
    mod = build(k=1152, operand_k_tile=128, operand_stages=3)
    region = find_ops_of_type(mod, OperandPipelineOp)[0]
    for bank in region.operands:
        assert bank.type.sizes[1] == 128
        assert tuple(bank.definition.arg.type.sizes) == (bank.type.sizes[0], 384)
        assert bank.definition.arg.type.strides == (384, 1)


@pytest.mark.parametrize("change", ["rows", "layout", "swizzle", "dtype"])
def test_blocked_bank_stride_exception_keeps_physical_layout(change):
    from ayla.base.ir import clone_type
    from ayla.gpu_dialect import F32Type
    from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
    mod = build(k=1152, operand_k_tile=128, operand_stages=3)
    bank = find_ops_of_type(mod, OperandPipelineOp)[0].operands[0]
    bank.type = clone_type(bank.type)
    if change == "rows":
        bank.type.sizes = [bank.type.sizes[0]//2,128]
    elif change == "layout":
        bank.type.attrs["physical_layout"] = {"block_width":32,"order":"block_major"}
    elif change == "swizzle":
        bank.type.attrs["swizzle"] = "64B"
    else:
        bank.type.attrs["dtype"] = F32Type()
    with pytest.raises(ValueError, match="strides"):
        VerifyGpuDialect()._verify_tile(bank.definition)
