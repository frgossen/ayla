"""Measured M192 routes retain exact shape scope and launch through the builder."""

import pytest
import torch

from ayla.gemm_collection.blackwell_tcgen05 import (
    blackwell_tcgen05_gemm_module,
    select_blackwell_gemm_schedule,
)

ROUTES = ((1000, 1280, 32), (1000, 1280, 96), (1000, 1536, 64))


@pytest.mark.parametrize("shape", ROUTES)
def test_m192_route_selection_is_exact_and_returns_an_independent_config(shape):
    config = select_blackwell_gemm_schedule(*shape)
    assert config is not None
    assert config["output_m_tile"] == 192
    assert config["output_n_tile"] == 64
    assert config["cta_group"] == 2
    config["output_m_tile"] = -1
    assert select_blackwell_gemm_schedule(*shape)["output_m_tile"] == 192
    assert select_blackwell_gemm_schedule(*shape, dtype="bf16") is None
    assert select_blackwell_gemm_schedule(shape[0], shape[1], shape[2] + 1) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)
@pytest.mark.parametrize("shape", ROUTES)
def test_selected_m192_routes_replay_changed_inputs_with_output_guards(shape):
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher

    m, n, k = shape
    launch = make_ptx_launcher(
        make_ptx_kernel(
            blackwell_tcgen05_gemm_module(
                m=m, n=n, k=k, dtype="f16", **select_blackwell_gemm_schedule(*shape)
            )
        )
    )
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    storage = torch.full((m * n + 256,), 71.0, device="cuda", dtype=torch.float16)
    c = storage[128:-128].view(m, n)
    launch(a, b, c)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(a, b, c)
    for seed in (23, 41, 67):
        torch.manual_seed(seed)
        a.normal_()
        b.normal_()
        expected = a.float() @ b.float().T
        c.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        rms = ((c.float() - expected).square().mean() / expected.square().mean()).sqrt()
        assert float(rms) < 0.002
        assert bool((storage[:128] == 71.0).all() & (storage[-128:] == 71.0).all())
