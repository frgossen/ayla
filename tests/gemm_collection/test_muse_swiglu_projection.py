"""Tests for the standalone Muse Glimmer SwiGLU projection."""
import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.muse_swiglu_projection import (
    make_dynamic_m_muse_swiglu_projection,
    muse_swiglu_projection_module,
    muse_swiglu_projection_template,
)
from ayla.gpu_dialect import (
    AllocOp,
    BarrierOp,
    ElementwiseOp,
    KernelOp,
    MatmulOp,
    SmemTensorType,
    TileOp,
)
from ayla.ptx_codegen import make_ptx_kernel
from tests.testing_utils import requires_cuda


def test_projection_template_keeps_m_dynamic():
    module = muse_swiglu_projection_template()
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    assert [value.type.sizes for value in kernel.body.arguments] == [
        [None, 6656], [39936, 6656], [None, 19968],
    ]
    assert kernel.grid_dim == (104, None, 1)


@pytest.mark.parametrize("m", [8, 128, 512])
def test_projection_uses_explicit_paired_register_schedule(m: int):
    module = muse_swiglu_projection_module(m=m)
    [kernel_op] = module.body.ops
    assert isinstance(kernel_op, KernelOp)
    assert kernel_op.grid_dim == (104, (m + 63) // 64, 1)
    assert kernel_op.block_dim == (128, 1, 1)
    assert [value.type.sizes for value in kernel_op.body.arguments] == [
        [m, 6656], [39936, 6656], [m, 19968],
    ]

    matmuls = find_ops_of_type(module, MatmulOp)
    assert {op.attrs["label"].split("_")[0] for op in matmuls} == {
        "gate", "up",
    }
    assert [op.fn for op in find_ops_of_type(module, ElementwiseOp)] == [
        "silu", "mul", "convert",
    ]
    assert all(
        op.scope == "thread" for op in find_ops_of_type(module, BarrierOp)
    )

    packed_weight = kernel_op.body.arguments[1]
    packed_views = [
        op for op in find_ops_of_type(module, TileOp)
        if op.arg is packed_weight and op.result().type.sizes == [19968, 6656]
    ]
    assert sorted(op.static_indices for op in packed_views) == [[0, 0], [1, 0]]

    smem_allocs = [
        op for op in find_ops_of_type(module, AllocOp)
        if isinstance(op.result().type, SmemTensorType)
    ]
    assert {op.attrs.get("reuse_group") for op in smem_allocs} == {
        "gate_compute", "up_compute", "up_writeback",
    }

    kernel = make_ptx_kernel(module)
    assert kernel.ptx.count(".shared .align 128 .b8") == 1
    assert "[98304]" in kernel.ptx
    assert "ex2.approx.f32" in kernel.ptx
    assert "mul.f32" in kernel.ptx


def test_projection_rejects_nonpositive_m():
    with pytest.raises(ValueError, match="M must be positive"):
        muse_swiglu_projection_module(m=0)


@requires_cuda
@pytest.mark.parametrize("m", [8, 512])
def test_dynamic_projection_matches_torch(m: int):
    torch.manual_seed(20260918 + m)
    lhs = torch.randn((m, 6656), device="cuda", dtype=torch.bfloat16)
    packed_weight = (
        torch.randn(
            (39936, 6656), device="cuda", dtype=torch.bfloat16,
        ) * .01
    )
    op = make_dynamic_m_muse_swiglu_projection(
        namespace=f"test_muse_swiglu_projection_m{m}",
    )
    actual = op(lhs, packed_weight)
    gate_weight, up_weight = packed_weight.split(19968)
    expected = (
        torch.nn.functional.silu(lhs @ gate_weight.T)
        * (lhs @ up_weight.T)
    )
    torch.cuda.synchronize()

    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    assert max_abs <= .125
    assert relative_rms <= .006
