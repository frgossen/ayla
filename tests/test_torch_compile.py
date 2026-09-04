"""Routing a matmul and its epilogue through ayla under torch.compile."""
import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code

from ayla import torch_compile
from tests.testing_utils import requires_cuda


def _op_name(overload) -> str:
    """The bare op name, as it appears in generated code."""
    return overload.name().split("::")[-1]


# Which kernel serves which case is the lowering's to say, so read it off the
# lowering rather than pinning a name a swap would then have to chase.
_MM = _op_name(torch_compile._MM_OUT)
_MM_SIGMOID = _op_name(torch_compile._MM_SIGMOID_OUT)


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize("fn,op", [
    pytest.param(
        lambda x, w: torch.sigmoid(x @ w.t()),
        _MM_SIGMOID,
        id="sigmoid_epilogue",
    ),
    pytest.param(
        lambda x, w: x @ w.t(),
        _MM,
        id="no_epilogue",
    ),
])
def test_a_matmul_compiles_to_one_ayla_kernel(fn, op):
    """The epilogue, where there is one, is folded in rather than left over."""
    torch.manual_seed(0)
    # Scaled so the sigmoid lands in its transition region rather than
    # saturating, where any wrong function would still look about right.
    x, w = (
        (torch.randn(4096, 4096, device="cuda") * 0.02).to(torch.bfloat16)
        for _ in range(2)
    )
    expected = fn(x, w)

    # The backend is not part of the Inductor cache key, so a warm cache would
    # serve a stock-compiled graph and the lowering would never run.
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    calls = [line for line in code.splitlines() if "torch.ops.ayla" in line]
    assert len(calls) == 1
    assert op in calls[0]
    # Nothing is left over to read the output back and apply the epilogue.
    assert code.count("async_compile.triton") == 0


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_a_second_reader_of_the_matmul_keeps_both_kernels():
    """The fused kernel stores the sigmoid and never the matmul behind it.

    So it can only stand in where nothing else reads that. Fusing anyway would
    leave the graph returning a buffer the kernel never wrote.
    """
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

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
    assert _MM in code, "the matmul is still ayla's"
    assert _MM_SIGMOID not in code, "but the epilogue cannot fold into it"
    assert code.count("async_compile.triton") == 1


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_a_matmul_off_the_gpu_is_declined():
    """Shape and dtype match, device does not -- the kernels are cuda only."""
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
