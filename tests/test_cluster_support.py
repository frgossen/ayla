import math

import pytest
from ayla.cluster_layout import ClusterLayout, arrival_counts, validate_masks
from ayla.base.ir import find_ops_of_type, clone_op
from ayla.gemm_collection.blackwell_tcgen05 import (
    blackwell_tcgen05_gemm_module as build,
)
from tests.testing_utils import check_gemm
from ayla.gpu_dialect import (
    AsyncOp,
    BarrierOp,
    ForOp,
    LockType,
)
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.passes.insert_syncs import InsertSyncs
from ayla.ptx_codegen import PtxLoweringPipeline, make_ptx_kernel


def module(**extra):
    cfg = dict(
        m=512,
        n=512,
        k=1040,
        dtype="f16",
        output_n_tile=64,
        operand_stages=4,
        cyclic_operand_pipeline=True,
        output_store="stmatrix",
        allow_partial_tiles=True,
    )
    cfg.update(extra)
    return build(**cfg)


def before_sync(m):
    for stage in PtxLoweringPipeline().passes:
        if isinstance(stage, InsertSyncs):
            return m
        m = stage.run(m)


@pytest.mark.parametrize("physical", [(2, 2, 1), (4, 1, 1), (1, 4, 1), (2, 1, 2)])
def test_logical_pair_masks_are_independent_of_physical_shape(physical):
    layout = ClusterLayout(2, 2, physical, 2)
    assert layout.operand_masks("lhs") == (5, 10, 5, 10)
    assert layout.operand_masks("rhs") == (1, 2, 4, 8)
    masks = layout.bank_release_masks(
        layout.operand_masks("lhs"), layout.operand_masks("rhs")
    )
    assert masks == (15, 15, 15, 15) and arrival_counts(masks, 2) == (2, 2, 2, 2)


def test_ordinary_row_column_groups_and_consumers():
    layout = ClusterLayout(2, 2, (2, 2, 1))
    assert layout.operand_masks("lhs") == (5, 10, 5, 10)
    assert layout.operand_masks("rhs") == (3, 3, 12, 12)
    masks = layout.bank_release_masks(
        layout.operand_masks("lhs"), layout.operand_masks("rhs")
    )
    assert masks == (7, 11, 13, 14) and arrival_counts(masks) == (3, 3, 3, 3)


def test_rhs_shards_match_adjacent_hardware_pairs():
    assert ClusterLayout(4, 1, (4, 1, 1), 2).operand_masks("rhs") == (5, 10, 5, 10)


@pytest.mark.parametrize("masks", [[1], [3, 2], [5, 2], [0, 2], [1, 1]])
def test_invalid_recipient_groups(masks):
    with pytest.raises(ValueError):
        validate_masks(masks, 2, groups=True)


@pytest.mark.parametrize(
    "args", [(1, 2, (1, 2, 1), 2), (2, 2, (2, 1, 1), 2), (2, 1, (0, 2, 1), 2)]
)
def test_incomplete_or_mismatched_mapping_rejected(args):
    with pytest.raises(ValueError):
        ClusterLayout(*args)


@pytest.mark.parametrize(
    "extra",
    [
        dict(cta_group=2, cluster_dim=(1, 2, 1)),
        dict(cta_group=2, cluster_n_ctas=2),
        dict(cta_group=2, cluster_n_ctas=2, cluster_dim=(4, 1, 1)),
        dict(cta_group=2, cluster_m_ctas=4),
        dict(cluster_m_ctas=2),
        dict(cluster_n_ctas=2),
        dict(cluster_m_ctas=2, cluster_n_ctas=2),
    ],
)
def test_native_cluster_kernel_has_no_steady_software_gathers(extra):
    m = module(**extra)
    VerifyGpuDialect().run(m)
    sync = InsertSyncs().run(before_sync(clone_op(m, {})))
    for loop in find_ops_of_type(sync, ForOp):
        assert not [
            b
            for b in find_ops_of_type(loop, BarrierOp)
            if b.scope in ("cluster", "cluster_to_leader")
        ]
    ptx = make_ptx_kernel(m).ptx
    assert ".multicast::cluster.b64" in ptx
    assert "mbarrier.try_wait.parity.acquire.cluster" not in ptx


def test_tmem_readiness_cannot_release_other_pairs_operand_reads():
    m = module(cta_group=2, cluster_n_ctas=2)
    issue = [op for op in find_ops_of_type(m, AsyncOp) if len(op.completion_locks) > 1][
        -1
    ]
    # Pair (0, 1) writes the shared A tile for both pairs: it hears both
    # pairs' releases, while pair (2, 3) hears only its own.
    assert issue.lock.type.attrs["arrival_counts"] == [2, 2, 1, 1]
    assert issue.completion_locks[-1].type.attrs == {
        "kind": "tcgen05",
        "completion_role": "result",
        "cta_group": 2,
    }
    assert "arrival_counts" not in issue.completion_locks[-1].type.attrs


def test_incorrect_consumer_mask_rejected_even_if_count_matches():
    m = module(cta_group=2, cluster_n_ctas=2)
    issue = next(
        op for op in find_ops_of_type(m, AsyncOp) if op.lock.type.kind == "tcgen05"
    )
    issue.lock.type.attrs.update(
        completion_cta_masks=[3, 3, 12, 12], arrival_counts=[1] * 4
    )
    with pytest.raises(ValueError, match="operand consumers"):
        VerifyGpuDialect().run(m)


def test_collective_polling_cannot_be_silently_added_to_bank_event():
    m = module(cluster_m_ctas=2)
    issue = next(
        op for op in find_ops_of_type(m, AsyncOp) if op.lock.type.kind == "tcgen05"
    )
    from ayla.gpu_dialect import AwaitOp

    next(w for w in issue.lock.users() if isinstance(w, AwaitOp)).attrs.pop(
        "waiter_warps"
    )
    with pytest.raises(ValueError, match="issuer observations"):
        VerifyGpuDialect().run(m)


def test_generic_gather_only_notifies_row_or_column_consumers(monkeypatch):
    from ayla.passes.pipeline_cluster_banks import PipelineClusterBanks

    monkeypatch.setattr(PipelineClusterBanks, "run", lambda self, m: m)
    sync = InsertSyncs().run(before_sync(module(cluster_m_ctas=2, cluster_n_ctas=2)))
    masks = [
        b.sync.type.attrs["release_cta_masks"]
        for b in find_ops_of_type(sync, BarrierOp)
        if b.scope == "cluster_to_leader"
        and b.sync is not None
        and "release_cta_masks" in b.sync.type.attrs
    ]
    assert masks and all(x == [7, 11, 13, 14] for x in masks)


@pytest.mark.parametrize("rolled", [False, True])
@pytest.mark.parametrize("physical", [(2, 2, 1), (4, 1, 1)])
def test_persistent_cluster_grid_preserves_complete_work_groups(rolled, physical):
    m = module(
        m=768,
        n=264,
        cta_group=2,
        cluster_n_ctas=2,
        cluster_dim=physical,
        output_grid_size=12,
        reuse_tmem_allocation=True,
        reuse_completion_locks=True,
        rolled_operand_pipeline=rolled,
    )
    kernel = make_ptx_kernel(m)
    assert kernel.grid_dim == (12 // 4 * physical[0], physical[1], 1)
    assert kernel.cluster_dim == physical


def test_incomplete_persistent_cluster_grid_rejected():
    with pytest.raises(ValueError, match="complete CTA rank groups"):
        module(cta_group=2, cluster_n_ctas=2, output_grid_size=6)


def test_multicast_groups_support_distinct_operation_issuers():
    m = module(
        cluster_m_ctas=2,
        cluster_n_ctas=2,
        operand_move_attrs={"lhs": {"issuer_warp": 1}, "rhs": {"issuer_warp": 1}},
    )
    assert not m.body.ops[0].attrs.get("cluster_bank_pipeline")
    sync = InsertSyncs().run(before_sync(m))
    waits = [
        b
        for b in find_ops_of_type(sync, BarrierOp)
        if b.scope == "cluster_to_leader" and b.attrs.get("phase") == "wait"
    ]
    assert waits and all(b.attrs.get("issuer_warp") == 1 for b in waits)
    make_ptx_kernel(
        module(
            cluster_m_ctas=2,
            cluster_n_ctas=2,
            operand_move_attrs={"lhs": {"issuer_warp": 1}, "rhs": {"issuer_warp": 1}},
        )
    )


def test_user_can_disable_one_operands_multicast():
    m = module(
        cluster_m_ctas=2,
        cluster_n_ctas=2,
        operand_move_attrs={"lhs": {"multicast": False}},
    )
    moves = [
        move
        for op in find_ops_of_type(m, AsyncOp)
        if op.lock.type.kind == "mbar"
        for move in op.body.ops
    ]
    assert any(not move.multicast for move in moves) and any(
        move.multicast for move in moves
    )


def test_result_await_models_only_tmem_writes():
    from ayla.gpu_dialect import AwaitOp, TmemTensorType, memory_effects

    m = module(cta_group=2, cluster_n_ctas=2)
    wait = next(
        w
        for w in find_ops_of_type(m, AwaitOp)
        if w.lock.type.attrs.get("completion_role") == "result"
    )
    effects = memory_effects(wait)
    assert effects and all(
        e.kind == "W" and isinstance(e.value.type, TmemTensorType) for e in effects
    )


def test_rolled_event_retirement_observes_each_banks_last_generation():
    from ayla.gpu_dialect import AllocOp, AwaitOp
    from ayla.passes.pipeline_rolled_k import LowerRolledK

    m = InsertSyncs().run(
        before_sync(
            module(
                cta_group=2,
                cluster_n_ctas=2,
                rolled_operand_pipeline=True,
            )
        )
    )
    # The completed region already publishes its result: no additional
    # whole-cluster rendezvous is needed immediately outside that region.
    cluster_barriers = [
        b for b in find_ops_of_type(m, BarrierOp) if b.scope == "cluster"
    ]
    assert all(b.attrs.get("initialization_publication") for b in cluster_barriers)
    lowered = LowerRolledK().run(m)
    free = next(
        a.result()
        for a in find_ops_of_type(lowered, AllocOp)
        if isinstance(a.result().type, LockType)
        and a.result().type.attrs.get("cluster_bank_release")
        and a.result().type.attrs.get("count") == 4
    )
    # Releases reach each read bank's next writer: pair (2, 3) notifies the
    # A-tile writer pair (0, 1) as well as itself.
    assert free.type.attrs["completion_cta_masks"] == [3, 3, 15, 15]
    assert free.type.attrs["arrival_counts"] == [2, 2, 1, 1]
    drain = next(
        loop
        for loop in find_ops_of_type(lowered, ForOp)
        if loop.attrs.get("label") == "rolled_k_drain"
    )
    assert set(i % 4 for i in range(drain.lb, drain.ub)) == set(range(4))
    assert all(w.waiter_warps == (0,) for w in find_ops_of_type(drain, AwaitOp))
    parent = drain.parent_block
    assert isinstance(parent.ops[parent.ops.index(drain) + 1], BarrierOp)


@pytest.mark.parametrize("rolled", [False, True])
@pytest.mark.parametrize("logical,physical", [((2, 2), (4, 1, 1)), ((4, 2), (4, 2, 1))])
def test_gpu_cluster_persistent_partial_final_wave(rolled, logical, physical):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher
    from ayla.ptx_codegen import make_ptx_kernel

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires Blackwell")
    cm, cn = logical
    cfg = dict(
        m=768,
        n=264,
        k=1040,
        cta_group=2,
        cluster_m_ctas=cm,
        cluster_n_ctas=cn,
        cluster_dim=physical,
        output_grid_size=3 * cm * cn,
        reuse_tmem_allocation=True,
        reuse_completion_locks=True,
        rolled_operand_pipeline=rolled,
    )
    launch = make_ptx_launcher(make_ptx_kernel(module(**cfg)))
    check_gemm(launch, 768, 264, 1040, "f16", mutate=True, runs=6, atol=math.inf, rms=0.002)


@pytest.mark.parametrize(
    "extra, masks, counts",
    [
        # Whole-tile A multicast from rank 0: rank 1 releases to the A writer
        # and itself, rank 0 only to itself.
        (dict(cluster_n_ctas=2), [1, 3], [2, 1]),
        # Two slices: each member writes one, so both hear both releases.
        (dict(cluster_n_ctas=2, operand_move_attrs={"lhs": {"multicast_slices": 2}}), [3, 3], [2, 2]),
        # A and B shared along different axes: their writers differ.
        (dict(cluster_m_ctas=2, cluster_n_ctas=2), [1, 3, 5, 14], [3, 2, 2, 1]),
    ],
)
def test_bank_release_notifies_the_next_writers(extra, masks, counts):
    m = module(**extra)
    lock = next(
        op.lock for op in find_ops_of_type(m, AsyncOp) if op.lock.type.kind == "tcgen05"
    )
    assert lock.type.attrs["completion_cta_masks"] == masks
    assert lock.type.attrs["arrival_counts"] == counts
