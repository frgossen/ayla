"""Tests for the deferred grid-synchronized Muse MLP experiment."""
import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.muse_mlp_gemm_chain import (
    muse_glimmer_swiglu_module,
)
from ayla.gpu_dialect import BarrierOp, ElementwiseOp, KernelOp, MatmulOp
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


@pytest.mark.parametrize("m", [32, 64])
def test_chain_reuses_the_standalone_projection(m: int):
    module = muse_glimmer_swiglu_module(m=m)
    [kernel_op] = module.body.ops
    assert isinstance(kernel_op, KernelOp)
    assert kernel_op.grid_dim == (104, 1, 1)
    assert [value.type.sizes for value in kernel_op.body.arguments] == [
        [m, 6656], [39936, 6656], [6656, 19968],
        [m, 19968], [m, 6656], [2],
    ]

    matmuls = find_ops_of_type(module, MatmulOp)
    assert {op.attrs["label"].split("_")[0] for op in matmuls} == {
        "gate", "up", "down",
    }
    assert [op.fn for op in find_ops_of_type(module, ElementwiseOp)] == [
        "silu", "mul", "convert", "convert",
    ]
    assert [
        op.scope for op in find_ops_of_type(module, BarrierOp)
        if op.scope == "grid"
    ] == ["grid"]

    kernel = make_ptx_kernel(module)
    assert kernel.block_dim == (128, 1, 1)
    assert kernel.ptx.count(".shared .align 128 .b8") == 1
    assert "ex2.approx.f32" in kernel.ptx
    assert "mul.f32" in kernel.ptx


def test_chain_rejects_unrouted_m():
    with pytest.raises(ValueError, match="unsupported M=128"):
        muse_glimmer_swiglu_module(m=128)


@requires_cuda
@pytest.mark.parametrize("m", [32, 64])
def test_chain_matches_torch(m: int):
    torch.manual_seed(20260902 + m)
    lhs = torch.randn((m, 6656), device="cuda", dtype=torch.bfloat16)
    packed_weight = (
        torch.randn((39936, 6656), device="cuda", dtype=torch.bfloat16) * .01
    )
    down_weight = (
        torch.randn((6656, 19968), device="cuda", dtype=torch.bfloat16) * .01
    )
    hidden = torch.empty(
        (m, 19968), device="cuda", dtype=torch.bfloat16,
    )
    actual = torch.empty((m, 6656), device="cuda", dtype=torch.bfloat16)
    sync = torch.zeros(2, device="cuda", dtype=torch.float32)
    launcher = make_ptx_launcher(make_ptx_kernel(
        muse_glimmer_swiglu_module(m=m),
    ))

    launcher(lhs, packed_weight, down_weight, hidden, actual, sync)
    gate_weight, up_weight = packed_weight.split(19968)
    expected = (
        torch.nn.functional.silu(lhs @ gate_weight.T)
        * (lhs @ up_weight.T)
    ) @ down_weight.T
    torch.cuda.synchronize()

    difference = actual.float() - expected.float()
    max_abs = float(difference.abs().max())
    relative_rms = float(difference.square().mean().sqrt()) / float(
        expected.float().square().mean().sqrt()
    )
    assert max_abs <= .125
    assert relative_rms <= .006
