"""Routing a matmul and its epilogue through ayla under torch.compile."""

import pytest
import torch
import torch._inductor.config as inductor_config
from torch.fx import Graph, GraphModule
from torch._inductor.utils import run_and_get_code

from tests.testing_utils import requires_cuda


def _gemm_graph(m: int, n: int, k: int, *, batch: int | None = None):
    graph = Graph()
    batch_dims = () if batch is None else (batch,)
    lhs = graph.placeholder("lhs")
    lhs.meta["val"] = torch.empty(
        (*batch_dims, m, k),
        dtype=torch.bfloat16,
        device="meta",
    )
    weight = graph.placeholder("weight")
    weight.meta["val"] = torch.empty(
        (*batch_dims, n, k),
        dtype=torch.bfloat16,
        device="meta",
    )
    permutation = [1, 0] if batch is None else [0, 2, 1]
    weight_t = graph.call_function(
        torch.ops.aten.permute.default,
        args=(weight, permutation),
    )
    weight_t.meta["val"] = weight.meta["val"].permute(permutation)
    target = torch.ops.aten.mm if batch is None else torch.ops.aten.bmm
    output = graph.call_function(target.default, args=(lhs, weight_t))
    output.meta["val"] = torch.empty(
        (*batch_dims, m, n),
        dtype=torch.bfloat16,
        device="meta",
    )
    graph.output(output)
    return GraphModule({}, graph), lhs, weight


@pytest.mark.parametrize(
    "op,batch,shape",
    [
        ("gemm", None, (128, 64)),
        ("bmm", 3, (3, 128, 64)),
    ],
)
def test_a_non_4096_gemm_is_routed_and_compiled(op, batch, shape):
    module, lhs, weight = _gemm_graph(128, 64, 64, batch=batch)

    from ayla import torch_compile

    assert torch_compile._POST_GRAD_PATTERNS.apply(module) == 1
    calls = [node for node in module.graph.nodes if node.op == "call_function"]
    assert len(calls) == 1
    assert calls[0].target is getattr(torch.ops.ayla, op).default
    result = calls[0].target(lhs.meta["val"], weight.meta["val"])
    assert result.shape == shape


def test_a_gemm_with_an_unsupported_k_is_not_routed():
    module, _, _ = _gemm_graph(128, 64, 32)

    from ayla import torch_compile

    assert torch_compile._POST_GRAD_PATTERNS.apply(module) == 0


@requires_cuda
@pytest.mark.parametrize("m", [1, 4, 32, 64, 512, 2048])
def test_muse_swiglu_is_captured_and_run(m: int):
    from benchmarks.bench_torch_compile.corpus.muse_swiglu import build_config

    config = build_config(m)
    inputs = config.make_inputs()
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        expected = torch.compile(config.compute, dynamic=False)(*inputs)
        from ayla import torch_compile

        torch_compile.install()
        try:
            actual, codes = run_and_get_code(
                torch.compile(config.compute, dynamic=False),
                *inputs,
            )
        finally:
            torch_compile.uninstall()

    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.006)
    calls = [
        line
        for code in codes
        for line in code.splitlines()
        if "torch.ops.ayla.muse_swiglu.default(" in line
    ]
    assert len(calls) == 1
    outer_code = next(code for code in codes if calls[0] in code)
    assert outer_code.count("async_compile.triton") == 0


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize(
    "fn,op,shape",
    [
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
    ],
)
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
        line for line in code.splitlines() if f"torch.ops.ayla.{op}.default(" in line
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
    assert "torch.ops.ayla.gemm.default(" in code
    assert "torch.ops.ayla.gemm_sigmoid_4096.default(" not in code
    assert code.count("async_compile.triton") == 1


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_a_matmul_off_the_gpu_is_not_rewritten():
    torch.manual_seed(0)
    x, w = ((torch.randn(4096, 4096) * 0.02).to(torch.bfloat16) for _ in range(2))

    def fn(x, w):
        return torch.sigmoid(x @ w.t())

    expected = fn(x, w)
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, (code,) = run_and_get_code(torch.compile(fn, dynamic=False), x, w)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert "torch.ops.ayla" not in code


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("operation", ["gemm", "addmm", "bmm"])
def test_measured_gemm_routes_preserve_results_and_graph_replay(dtype, operation):
    torch.manual_seed(21)
    lead = (3,) if operation == "bmm" else ()
    lhs = torch.randn((*lead, 128, 256), device="cuda", dtype=dtype) * 0.02
    weight = torch.randn((*lead, 128, 256), device="cuda", dtype=dtype) * 0.02
    bias = torch.randn((128,), device="cuda", dtype=dtype) * 0.02

    if operation == "addmm":

        def compute(lhs, weight, bias):
            return torch.addmm(bias, lhs, weight.t())
    elif operation == "bmm":

        def compute(lhs, weight, bias):
            return torch.bmm(lhs, weight.transpose(-1, -2))
    else:

        def compute(lhs, weight, bias):
            return lhs @ weight.t()

    expected = compute(lhs, weight, bias)
    compiled = torch.compile(compute, dynamic=False)
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        actual, codes = run_and_get_code(compiled, lhs, weight, bias)
        route = operation if dtype == torch.bfloat16 else f"{operation}_fp16"
        assert (
            sum(code.count(f"torch.ops.ayla.{route}.default(") for code in codes) == 1
        )
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
        # Warming above must populate both the specialization and device caches.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_result = compiled(lhs, weight, bias)
        graph.replay()
        torch.testing.assert_close(replay_result, expected, atol=2e-3, rtol=2e-2)
