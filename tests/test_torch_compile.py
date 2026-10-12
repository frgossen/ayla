"""Routing a matmul and its epilogue through ayla under torch.compile."""
import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code

from tests.testing_utils import requires_cuda


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_glimmer_attention_subgraph_lowers_executable_matmuls():
    def glimmer_attention_subgraph(
        attention_output,
        gate_input,
        gate_weight,
        output_projection_weight,
    ):
        attention_2d = torch.ops.aten.view.default(
            attention_output, [-1, 4096],
        )
        gate_weight_transpose = torch.ops.aten.permute.default(
            gate_weight, [1, 0],
        )
        gate_logits = torch.ops.aten.mm.default(
            gate_input, gate_weight_transpose,
        )
        gate_activation = torch.ops.aten.sigmoid.default(gate_logits)
        gated_attention = torch.ops.aten.mul.Tensor(
            gate_activation, attention_2d,
        )
        output_weight_transpose = torch.ops.aten.permute.default(
            output_projection_weight, [1, 0],
        )
        projected_attention = torch.ops.aten.mm.default(
            gated_attention, output_weight_transpose,
        )
        return projected_attention

    num_tokens = 8
    inputs = tuple(
        (torch.randn(shape, device="cuda") * 0.02).to(torch.bfloat16)
        for shape in (
            (num_tokens, 32, 128),
            (num_tokens, 6656),
            (4096, 6656),
            (6656, 4096),
        )
    )
    expected = glimmer_attention_subgraph(*inputs)
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(
            torch.compile(glimmer_attention_subgraph, dynamic=False), *inputs,
        )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    calls = [line for line in code.splitlines() if "torch.ops.ayla" in line]
    assert len(calls) == 2
    assert "dynamic_m_matmul_n4096_k6656_out" in calls[0]
    assert "dynamic_m_matmul_n6656_k4096_out" in calls[1]
    assert "# ayla nodes: mm" in code
    assert "# ayla nodes: mm_1" in code


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize("fn,op", [
    pytest.param(
        lambda x, w: torch.sigmoid(x @ w.t()),
        "auto_matmul_4096_wgmma256_wg_sigmoid_out",
        id="sigmoid_epilogue",
    ),
    pytest.param(
        lambda x, w: x @ w.t(),
        "auto_matmul_4096_wgmma256_wg_out",
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
    assert "auto_matmul_4096_wgmma256_wg_out" in code, "the matmul is still ayla's"
    assert "_sigmoid_out" not in code, "but the epilogue cannot fold into it"
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
