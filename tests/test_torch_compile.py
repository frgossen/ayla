"""Measured single-GEMM routing and CUDA graph replay."""

import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code

from tests.testing_utils import requires_cuda


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
        route = f"jit_{'mm' if operation == 'gemm' else operation}_{'bf16' if dtype == torch.bfloat16 else 'f16'}"
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
