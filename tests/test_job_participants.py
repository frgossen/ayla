import pytest
from ayla.base.ir import Block, find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    LockType,
    LockViewOp,
    IndexConstantOp,
    IndexLtOp,
    WarpIdsOp,
    IfOp,
    KernelOp,
    ModuleOp,
)
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect


def region_test(inner_warps, outer=True):
    w = WarpIdsOp.build()
    lo = IndexConstantOp.build(3)
    hi = IndexConstantOp.build(5)
    suffix = IndexLtOp.build(lo.result(), w.x)
    prefix = IndexLtOp.build(w.x, hi.result())
    inner = IfOp.build(prefix.result(), Block(ops=[]))
    inner.attrs["participant_warps"] = inner_warps
    parent = IfOp.build(suffix.result(), Block(ops=[inner]))
    parent.attrs["participant_warps"] = [4, 5, 6, 7]
    k = KernelOp.build(
        "bounds",
        Block(ops=[w, lo, hi, suffix, prefix, parent if outer else inner]),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )
    return ModuleOp.build(Block(ops=[k]), attrs={"target": "sm_100a"})


def test_nested_bound_verifies_exact_middle_warp():
    VerifyGpuDialect().run(region_test([4]))


def test_standalone_bound_cannot_claim_middle_warp():
    with pytest.raises(ValueError, match="declared warps"):
        VerifyGpuDialect().run(region_test([4], False))


def test_incorrect_intersection_is_rejected():
    with pytest.raises(ValueError, match="declared warps"):
        VerifyGpuDialect().run(region_test([4, 5]))


def test_scalar_event_explicit_generation():
    event = AllocOp.build_with_fresh_id(LockType.build("tcgen05"))
    zero = IndexConstantOp.build(0)
    phase = IndexConstantOp.build(1)
    view = LockViewOp.build(event.result(), zero.result(), phase.result())
    assert view.operands == [event.result(), zero.result(), phase.result()]
    with pytest.raises(ValueError):
        LockViewOp.build(event.result(), phase.result(), phase.result())


@pytest.mark.parametrize(
    "m,n,k,stages,nt,buffers,column",
    [
        (2049, 520, 384, 4, 128, 1, False),
        (3073, 384, 448, 5, 192, 1, False),
        (2049, 520, 784, 6, 192, 2, False),
        (1280, 320, 512, 4, 128, 1, True),
        # Rings deeper than one job: a peer's loads run jobs ahead.
        (2049, 520, 128, 5, 128, 1, False),
        (2049, 520, 128, 7, 128, 1, False),
    ],
)
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gpu_job_participants_changed_input_replays(
    m, n, k, stages, nt, buffers, column, dtype
):
    import torch
    from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher
    from tests.testing_utils import check_gemm

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    cfg = dict(
        cta_group=2,
        output_m_tile=128,
        output_n_tile=nt,
        operand_k_tile=64,
        operand_stages=stages,
        cyclic_operand_pipeline=False,
        rolled_operand_pipeline=True,
        output_grid_fast_axis="m",
        output_grid_size=4,
        persistent_output_pipeline=True,
        persistent_output_schedule="shared_ring",
        persistent_output_overlap=True,
        output_store="stmatrix",
        output_slices=2,
        output_store_buffers=buffers,
        tmem_writeback_register_budget=32,
        allow_partial_tiles=True,
        hoist_mma_descriptors=True,
        output_column_major=column,
        persistent_output_job_regions=True,
    )
    kernel = make_ptx_kernel(
        blackwell_tcgen05_gemm_module(m=m, n=n, k=k, dtype=dtype, **cfg)
    )
    check_gemm(make_ptx_launcher(kernel), m, n, k, dtype, graph=True, runs=3,
               transpose_out=column)


def test_issue_handoff_is_cta_local_and_released_by_the_actor():
    """Rotating issuers hand off through each CTA's own event.

    Ordering on the previous job's MMA completion instead let a paired peer,
    whose loads ran more than one job ahead, fall two event phases behind and
    wait forever on an aliased parity.
    """
    from ayla.base.ir import clone_op, find_ancestor_of_type
    from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
    from ayla.gpu_dialect import AwaitOp, BarrierOp, ForOp
    from ayla.passes.pipeline_rolled_shared_ring import LowerRolledSharedRing

    mod = blackwell_tcgen05_gemm_module(
        m=2049, n=520, k=128, dtype="f16", cta_group=2, output_m_tile=128,
        output_n_tile=128, operand_k_tile=64, operand_stages=7,
        cyclic_operand_pipeline=False, rolled_operand_pipeline=True,
        output_grid_fast_axis="m", output_grid_size=4,
        persistent_output_pipeline=True, persistent_output_schedule="shared_ring",
        persistent_output_overlap=True, persistent_output_job_regions=True,
        output_store="stmatrix", output_slices=2,
        tmem_writeback_register_budget=64, allow_partial_tiles=True,
    )
    expanded = LowerRolledSharedRing().run(clone_op(mod, {}))
    waits = [op for op in find_ops_of_type(expanded, AwaitOp)
             if op.attrs.get("label") == "previous_job_issued"]
    assert len(waits) == 2
    [handoff] = {op.lock.definition.operands[0] for op in waits}
    assert handoff.type.kind == "mbar" and "cta_group" not in handoff.type.attrs
    assert handoff.type.attrs["arrivals"] == 1
    releases = [op for op in find_ops_of_type(expanded, BarrierOp) if handoff in op.operands]
    assert sorted(op.attrs["participant_warps"] for op in releases) == [[0], [4]]
    for release in releases:
        # The actor releases only after issuing its whole job.
        block = release.parent_block.ops
        loop = block[block.index(release) - 1]
        assert isinstance(loop, ForOp) and loop.attrs["label"] == "shared_ring_rolled_k"
    assert not any(op.attrs.get("label") == "previous_job_mma_finished"
                   for op in find_ops_of_type(expanded, AwaitOp))
