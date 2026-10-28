"""Main-backed JIT selection, execution, graph capture, and Inductor routing."""
import pytest
import torch
from torch._inductor.utils import run_and_get_code

from ayla.gemm_jit import (
    _DEFAULTS,
    _fallback_schedule,
    _seed,
    gemm_jit_compiler,
    measured_gemm_configs,
    select_gemm_schedule,
)
from ayla.gemm_schedule_table import CONFIGS, SCHEDULES
from tests.testing_utils import requires_cuda


def test_measured_configs_preserve_full_feature_winners_and_are_copies():
    (batch, m, n, k), indices = next(iter(SCHEDULES['mm', 'bfloat16'].items()))
    configs = measured_gemm_configs('mm', 'bfloat16', m, n, k, batch)
    assert configs[0] == {**_DEFAULTS, **CONFIGS[indices[0]]}
    configs[0]['operand_stages'] = -1
    assert measured_gemm_configs('mm', 'bfloat16', m, n, k, batch)[0]['operand_stages'] != -1


def test_main_declines_feature_only_candidates_and_selects_supported_ones():
    fallback = _fallback_schedule('mm', 'bfloat16', 256, 256, 128)
    assert fallback is not None
    for feature in ('cyclic_operand_pipeline', 'pipeline_output_waves', 'swap_output_grid_axes', 'transpose_output'):
        config = {**fallback.captured_config, feature: True}
        assert _seed(config, 'mm', 256, 256, 128, (1, 256, 256, 128), True) is None
    config = {**fallback.captured_config, 'cluster_n_ctas': 2, 'multicast': True}
    assert _seed(config, 'mm', 256, 256, 128, (1, 256, 256, 128), True) is None
    selected = select_gemm_schedule('mm', 'bfloat16', 8192, 768, 768)
    assert selected is not None and selected.builder_config is not None
    assert 'output_grid_size' not in selected.builder_config
    assert 'cyclic_operand_pipeline' not in selected.builder_config


@pytest.mark.parametrize('dimensions', [(0, 64, 64), (64, 65, 64), (64, 64, 32)])
def test_invalid_or_unsupported_shapes_are_declined(dimensions):
    assert select_gemm_schedule('mm', 'bfloat16', *dimensions) is None


def test_unknown_dtype_is_declined():
    assert select_gemm_schedule('mm', 'float32', 256, 256, 128) is None


@pytest.mark.parametrize('operation', ['mm', 'addmm', 'bmm'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('m,n,k', [(96, 128, 128), (256, 256, 256)])
@requires_cuda
def test_jit_matches_torch_and_replays_cuda_graph(operation, dtype, m, n, k):
    compiler = gemm_jit_compiler(operation, str(dtype).removeprefix('torch.'))
    lead = (2,) if operation == 'bmm' else ()
    a = torch.randn((*lead, m, k), device='cuda', dtype=dtype)
    bt = torch.randn((*lead, n, k), device='cuda', dtype=dtype)
    out = torch.empty((*lead, m, n), device='cuda', dtype=dtype)
    args = [a, bt, out]
    expected = a @ bt.transpose(-1, -2)
    if operation == 'addmm':
        bias = torch.randn(n, device='cuda', dtype=dtype)
        args.insert(0, bias)
        expected = torch.addmm(bias, a, bt.T)
    compiler(*args)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.02)
    assert compiler.cache_size == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiler(*args)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.02)
    assert compiler.cache_size == 1


@pytest.mark.parametrize('operation', ['mm', 'addmm', 'bmm'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@requires_cuda
@pytest.mark.usefixtures('ayla_installed')
def test_torch_compile_routes_supported_gemms_to_jit(operation, dtype):
    lead = (2,) if operation == 'bmm' else ()
    a = torch.randn((*lead, 96, 128), device='cuda', dtype=dtype)
    bt = torch.randn((*lead, 128, 128), device='cuda', dtype=dtype)
    bias = torch.randn(128, device='cuda', dtype=dtype)

    def compute(a, bt, bias):
        if operation == 'addmm':
            return torch.addmm(bias, a, bt.T)
        if operation == 'bmm':
            return torch.bmm(a, bt.transpose(-1, -2))
        return torch.mm(a, bt.T)

    with torch.no_grad():
        actual, code = run_and_get_code(torch.compile(compute, fullgraph=True, dynamic=False), a, bt, bias)
        expected = compute(a, bt, bias)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
    assert any(f'jit_{operation}_' in source for source in code)
