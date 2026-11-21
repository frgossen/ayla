"""Deeper rolled MMA queues retain bank epochs and whole-job ownership."""

import math

import pytest

from ayla.base.ir import clone_op, find_ops_of_type
from ayla.gpu_dialect import PersistentPipelineOp
from ayla.passes.pipeline_rolled_shared_ring import LowerRolledSharedRing
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from tests.test_rolled_shared_ring import build, check_protocol
from tests.testing_utils import assert_dominance


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize(
    "steps,depth,queue",
    [
        (1, 3, 1),
        (1, 4, 2),
        (2, 5, 3),
        (3, 4, 2),
        (6, 5, 3),
        (12, 6, 4),
        (3, 4, 3),
        (12, 6, 5),
    ],
)
@pytest.mark.parametrize("mode", ["serial", "overlap", "whole_job_pingpong"])
def test_queued_bank_generations_and_single_stage_completion(
    group, steps, depth, queue, mode, monkeypatch
):
    module = build(
        k=steps * 64,
        operand_stages=depth,
        cyclic_operand_pipeline=queue,
        cta_group=group,
        persistent_output_overlap=mode != "serial",
        persistent_output_job_regions=mode == "whole_job_pingpong",
    )
    VerifyGpuDialect().run(module)
    if mode == "whole_job_pingpong":
        # Simulate global engine/event semantics before whole-job partition.
        # Actual common issuer roles are checked separately below, and GPU
        # graph/sanitizer checks exercise the partitioned execution.
        from ayla.passes.pipeline_loop import PipelineLoop

        monkeypatch.setattr(
            PipelineLoop,
            "assign_participants",
            lambda result, op: result,
        )
    expanded = LowerRolledSharedRing().run(clone_op(module, {}))
    assert_dominance(expanded.body)
    tiles = math.ceil(2049 / (128 * group)) * group * math.ceil(520 / 128)
    for block in range(4):
        jobs = len(range(block, tiles, 4))
        assert check_protocol(expanded, block, steps) == dict(
            load=jobs * steps, mma=jobs * steps, read=jobs * 2, store=jobs * 2
        )


def test_queued_pingpong_keeps_load_and_mma_issuers_common():
    from ayla.gpu_dialect import (
        MatmulOp,
        TmaMoveOp,
        SmemTensorType,
        execution_issuer_warp,
    )
    from ayla.passes.lower_to_tma_move import LowerToTmaMove

    module = build(
        k=192,
        operand_stages=5,
        cyclic_operand_pipeline=3,
        persistent_output_overlap=True,
        persistent_output_job_regions=True,
    )
    expanded = LowerRolledSharedRing().run(LowerToTmaMove().run(module))
    loads = {
        execution_issuer_warp(op)
        for op in find_ops_of_type(expanded, TmaMoveOp)
        if isinstance(op.dst.type, SmemTensorType) and op.attrs.get("tma")
    }
    mmas = {execution_issuer_warp(op) for op in find_ops_of_type(expanded, MatmulOp)}
    assert loads == mmas == {0, 4}


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("k", [64, 128])
def test_single_k128_batch_reaches_complete_ptx_lowering(group, k):
    from ayla.ptx_codegen import make_ptx_kernel

    # Exercise the late loop-peeling pass and K128 blocked operand layout;
    # the protocol interpreter alone stops before that lowering stage.
    module = build(
        k=k,
        operand_k_tile=128,
        output_n_tile=64,
        operand_stages=3,
        cyclic_operand_pipeline=2,
        cta_group=group,
    )
    kernel = make_ptx_kernel(module)
    assert kernel.ptx


@pytest.mark.parametrize("queue", [0, 4, 5, -1])
def test_queue_cannot_consume_all_operand_banks(queue):
    # False/zero remains the ordinary unqueued public schedule; explicit
    # malformed IR queue depths must still be rejected by its verifier.
    module = build(operand_stages=4)
    [region] = find_ops_of_type(module, PersistentPipelineOp)
    region.attrs["mma_queue_depth"] = queue
    with pytest.raises(ValueError, match="invalid compact persistent"):
        VerifyGpuDialect().run(module)
