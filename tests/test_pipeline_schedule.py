"""Normalized schedule contracts preserve geometry and ownership semantics."""

from copy import deepcopy

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_router import normalize_blackwell_configuration
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AllocOp, OperandPipelineOp, SmemTensorType
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.passes.pipeline_rolled_shared_ring import PrepareRolledSharedRing
from ayla.pipelines.ring_schedule import LoopSchedule, PipelineSchedule
from ayla.ptx_codegen import make_ptx_kernel


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("family", ["ordinary", "static", "rolled", "rotating", "serial"])
def test_normalized_schedule_preserves_generated_ptx(group, family):
    config = dict(m=2048, n=128, k=256, dtype="f16", output_n_tile=128,
                  operand_stages=3, cta_group=group, output_store="stmatrix",
                  allow_partial_tiles=True)
    if family != "ordinary":
        config.update(output_grid_size=4, persistent_output_pipeline=True,
                      output_slices=2, output_store_buffers=2,
                      rolled_operand_pipeline=family != "static",
                      persistent_output_overlap=family != "serial",
                      persistent_output_job_regions=family == "rotating")
        if family == "serial":
            config["persistent_output_frames"] = 1
        if family == "rolled":
            config.update(rolled_k_unroll=2, rolled_job_unroll=2,
                          peel_first_output_pair=True)
    normalized = normalize_blackwell_configuration(config)
    old = make_ptx_kernel(blackwell_tcgen05_gemm_module(**config))
    new = make_ptx_kernel(blackwell_tcgen05_gemm_module(**normalized))
    assert old == new


@pytest.mark.parametrize("group,n", [(2, 128), (4, 64), (8, 64)])
def test_cta_mma_geometry_batches_one_shared_rhs(group, n):
    module = blackwell_tcgen05_gemm_module(
        m=2048, n=n, k=256, dtype="f16", output_m_tile=128 * group,
        mma_m_tile=128, output_n_tile=n, operand_stages=1,
        output_store="stmatrix")
    storage = [op.result().type.sizes for op in find_ops_of_type(module, AllocOp)
               if isinstance(op.result().type, SmemTensorType)]
    assert [128 * group, 64] in storage
    assert storage.count([n, 64]) == 1


@pytest.mark.parametrize("options", [
    {"operand_scope": "output_jobs", "accumulator_slots": 1},
    {"participants": "round_robin"},
    {"output_jobs": {"peel_front": 1}},
    {"reduction": {"unroll": 2}},
    {"mma_pending": True},
    {"output_overlap": 1},
])
def test_incompatible_resource_plans_are_rejected(options):
    with pytest.raises(ValueError):
        PipelineSchedule.parse(options)


def test_serial_cross_job_ring_survives_configuration_roundtrip():
    plan = PipelineSchedule(operand_scope="output_jobs", accumulator_slots=1,
                            output_overlap=False, reduction=LoopSchedule("rolled"))
    assert PipelineSchedule.parse(plan.configuration()) == plan
    assert plan.cross_jobs and not plan.output_overlap


def test_zero_pending_survives_a_backend_with_default_one():
    plan = PipelineSchedule(mma_pending=0)
    config = plan.configuration(default_pending=1)
    assert config == {"mma_pending": 0}
    assert PipelineSchedule.parse(config, default_pending=1) == plan


def test_configuration_normalization_does_not_mutate_caller():
    original = dict(native_schedule=dict(rolled_operand_pipeline=True,
                                        cyclic_operand_pipeline=2), split_k=1)
    saved = deepcopy(original)
    normalized = normalize_blackwell_configuration(original)
    assert original == saved
    assert normalized["native_schedule"]["pipeline"]["mma_pending"] == 2


def test_removed_per_job_event_protocol_is_rejected():
    module = blackwell_tcgen05_gemm_module(
        m=2048, n=128, k=512, dtype="f16", output_n_tile=128,
        pipeline={"reduction": {"form": "rolled"}})
    region = find_ops_of_type(module, OperandPipelineOp)[0]
    region.attrs["carried_events"] = True
    with pytest.raises(ValueError, match="cross-job resource pipeline"):
        VerifyGpuDialect().run(module)


@pytest.mark.parametrize("group", [1, 2])
def test_generic_boundary_shaping_preserves_all_jobs_and_bank_generations(group):
    from tests.test_rolled_shared_ring import check_protocol
    from tests.testing_utils import assert_dominance
    import math

    module = blackwell_tcgen05_gemm_module(
        m=2049, n=520, k=784, dtype="f16", cta_group=group,
        output_n_tile=128, operand_stages=6, output_grid_size=4,
        output_store="stmatrix", output_slices=2, allow_partial_tiles=True,
        tmem_writeback_register_budget=64,
        pipeline=PipelineSchedule(
            operand_scope="output_jobs",
            reduction=LoopSchedule("rolled", unroll=2, peel_front=1, peel_back=1),
            output_jobs=LoopSchedule("rolled", unroll=2, peel_front=2, peel_back=2)))
    expanded = PrepareRolledSharedRing().run(module)
    assert_dominance(expanded.body)
    tiles = math.ceil(2049 / (128 * group)) * group * math.ceil(520 / 128)
    steps = math.ceil(784 / 64)
    for block in range(4):
        jobs = len(range(block, tiles, 4))
        assert check_protocol(expanded, block, steps) == dict(
            load=jobs * steps, mma=jobs * steps, read=jobs * 2, store=jobs * 2)
