"""Structural and H100 correctness tests for the Muse MLP GEMM chain."""

import pytest
import torch

from ayla.base.ir import find_ops
from ayla.gemm_collection.muse_mlp_gemm_chain import (
    muse_mlp_gemm_chain_module,
)
from ayla.gpu_dialect import (
    AllocOp,
    BarrierOp,
    ElementwiseOp,
    KernelOp,
    MatmulOp,
    SmemTensorType,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


@pytest.mark.parametrize("m", [32, 64])
def test_muse_mlp_chain_uses_small_m_schedule_and_composable_epilogue(m):
    module = muse_mlp_gemm_chain_module(m=m)
    [kernel_op] = module.body.ops
    assert isinstance(kernel_op, KernelOp)
    assert kernel_op.grid_dim == (312, 1, 1)
    assert [value.type.sizes for value in kernel_op.body.arguments[:5]] == [
        [m, 6656],
        [19968, 6656],
        [6656, 19968],
        [m, 19968],
        [m, 6656],
    ]

    matmuls = find_ops(module, lambda op: isinstance(op, MatmulOp))
    assert {op.attrs["label"].split("_")[0] for op in matmuls} == {
        "up", "down",
    }

    elementwise = find_ops(module, lambda op: isinstance(op, ElementwiseOp))
    assert [op.fn for op in elementwise] == ["silu", "convert"]

    barriers = find_ops(module, lambda op: isinstance(op, BarrierOp))
    grid_barriers = [op for op in barriers if op.scope == "grid"]
    assert len(grid_barriers) == 1
    assert grid_barriers[0].attrs == {"scope": "grid"}

    allocations = find_ops(
        module,
        lambda op: isinstance(op, AllocOp)
        and isinstance(op.result().type, SmemTensorType),
    )
    assert {op.attrs.get("reuse_group") for op in allocations} == {
        "up_compute", "up_writeback", "down_compute", "down_writeback",
    }

    kernel = make_ptx_kernel(module)
    assert kernel.block_dim == (128, 1, 1)
    assert kernel.requires_resident_grid
    assert kernel.ptx.count(".shared .align 128 .b8") == 1
    assert "ex2.approx.f32" in kernel.ptx
    after_store_read = kernel.ptx.split(
        "cp.async.bulk.wait_group.read 0;", 1,
    )[1]
    before_grid_arrival = after_store_read.split(
        "atom.acq_rel.gpu.global.inc.u32", 1,
    )[0]
    assert "cp.async.bulk.wait_group 0;" in before_grid_arrival
    assert "bar.sync 0;" not in before_grid_arrival


def test_muse_mlp_chain_rejects_unrouted_m():
    with pytest.raises(ValueError, match="unsupported M=128"):
        muse_mlp_gemm_chain_module(m=128)


@requires_cuda
@pytest.mark.parametrize("m", [32, 64])
def test_muse_mlp_chain_matches_torch(m):
    torch.manual_seed(20260902 + m)
    a = torch.randn((m, 6656), device="cuda", dtype=torch.bfloat16)
    w_up = (
        torch.randn((19968, 6656), device="cuda", dtype=torch.bfloat16) * .01
    )
    w_down = (
        torch.randn((6656, 19968), device="cuda", dtype=torch.bfloat16) * .01
    )
    intermediate = torch.empty(
        (m, 19968), device="cuda", dtype=torch.bfloat16,
    )
    actual = torch.empty((m, 6656), device="cuda", dtype=torch.bfloat16)
    sync = torch.zeros(2, device="cuda", dtype=torch.float32)
    launcher = make_ptx_launcher(make_ptx_kernel(
        muse_mlp_gemm_chain_module(m=m),
    ))

    launcher(a, w_up, w_down, intermediate, actual, sync)
    expected = torch.nn.functional.silu(a @ w_up.t()) @ w_down.t()
    torch.cuda.synchronize()

    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    assert max_abs <= .125
    assert relative_rms <= .006
