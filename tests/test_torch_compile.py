"""Routing a matmul and its epilogue through ayla under torch.compile."""
import pytest
import torch
import torch._inductor.config as inductor_config
from torch.fx import Graph, GraphModule
from torch._inductor.utils import run_and_get_code

from tests.testing_utils import requires_cuda


def _gemm_graph(m: int, n: int, k: int):
    graph = Graph()
    lhs = graph.placeholder("lhs")
    lhs.meta["val"] = torch.empty(
        (m, k), dtype=torch.bfloat16, device="meta",
    )
    weight = graph.placeholder("weight")
    weight.meta["val"] = torch.empty(
        (n, k), dtype=torch.bfloat16, device="meta",
    )
    weight_t = graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [1, 0]),
    )
    weight_t.meta["val"] = weight.meta["val"].t()
    output = graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, weight_t),
    )
    output.meta["val"] = torch.empty(
        (m, n), dtype=torch.bfloat16, device="meta",
    )
    graph.output(output)
    return GraphModule({}, graph), lhs, weight


def test_a_non_4096_gemm_is_routed_and_compiled():
    module, lhs, weight = _gemm_graph(128, 64, 64)

    from ayla import torch_compile

    assert torch_compile._POST_GRAD_PATTERNS.apply(module) == 1
    calls = [
        node for node in module.graph.nodes if node.op == "call_function"
    ]
    assert len(calls) == 1
    assert calls[0].target is torch.ops.ayla.gemm.default
    result = calls[0].target(lhs.meta["val"], weight.meta["val"])
    assert result.shape == (128, 64)


def test_a_gemm_with_an_unsupported_k_is_not_routed():
    module, _, _ = _gemm_graph(128, 64, 32)

    from ayla import torch_compile

    assert torch_compile._POST_GRAD_PATTERNS.apply(module) == 0


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize("fn,op,shape", [
    pytest.param(
        lambda x, w: torch.sigmoid(x @ w.t()),
        "gemm_sigmoid_4096",
        (4096, 4096, 4096),
        id="sigmoid_epilogue",
    ),
    pytest.param(
        lambda x, w: x @ w.t(),
        "gemm",
        (4096, 4096, 4096),
        id="plain_4096",
    ),
    pytest.param(
        lambda x, w: x @ w.t(),
        "gemm",
        (128, 64, 64),
        id="plain_non_4096",
    ),
])
def test_a_matmul_region_compiles_to_one_ayla_kernel(fn, op, shape):
    m, n, k = shape
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * 0.02).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
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
@pytest.mark.parametrize("shape", [(4096, 4096, 4096), (100, 192, 128)])
def test_a_biased_linear_compiles_to_one_ayla_addmm(shape):
    m, n, k = shape
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * 0.02).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    b = torch.randn(n, device="cuda").to(torch.bfloat16)
    fn = torch.nn.functional.linear
    expected = fn(x, w, b)

    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(
            torch.compile(fn, dynamic=False), x, w, b,
        )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert code.count("torch.ops.ayla.addmm.default(") == 1
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
    assert "torch.ops.ayla.gemm.default(" in code
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
