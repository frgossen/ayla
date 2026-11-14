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
