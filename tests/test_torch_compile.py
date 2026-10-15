"""Routing a matmul and its epilogue through ayla under torch.compile."""
import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code

from tests.testing_utils import requires_cuda


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize("fn,op", [
    pytest.param(
        lambda x, w: torch.sigmoid(x @ w.t()),
        "gemm_sigmoid_4096",
        id="sigmoid_epilogue",
    ),
    pytest.param(
        lambda x, w: x @ w.t(),
        "gemm_4096",
        id="no_epilogue",
    ),
])
def test_a_matmul_region_compiles_to_one_ayla_kernel(fn, op):
    torch.manual_seed(0)
    x, w = (
        (torch.randn(4096, 4096, device="cuda") * 0.02).to(torch.bfloat16)
        for _ in range(2)
    )
    expected = fn(x, w)

    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    calls = [
        line for line in code.splitlines()
        if f"torch.ops.ayla.{op}.default(" in line
    ]
    assert len(calls) == 1
    assert op in calls[0]
    assert code.count("async_compile.triton") == 0


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_a_shared_matmul_falls_back_to_the_plain_gemm_rule():
    torch.manual_seed(0)
    x, w = (
        (torch.randn(4096, 4096, device="cuda") * 0.02).to(torch.bfloat16)
        for _ in range(2)
    )

    def fn(x, w):
        y = x @ w.t()
        return torch.sigmoid(y), y

    expected = fn(x, w)
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
    assert "torch.ops.ayla.gemm_4096.default(" in code
    assert "torch.ops.ayla.gemm_sigmoid_4096.default(" not in code
    assert code.count("async_compile.triton") == 1


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_a_matmul_off_the_gpu_is_not_rewritten():
    torch.manual_seed(0)
    x, w = (
        (torch.randn(4096, 4096) * 0.02).to(torch.bfloat16) for _ in range(2)
    )

    def fn(x, w):
        return torch.sigmoid(x @ w.t())

    expected = fn(x, w)
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert "torch.ops.ayla" not in code
