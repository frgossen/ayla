"""Warpgroup pairing through the general scheduler and measured GEMM JIT."""


import pytest
import torch

from ayla.gemm_jit import (
    _DEFAULTS,
    _seed,
    build_gemm_module,
    gemm_op,
    select_gemm_schedule,
)
from ayla.gpu_dialect import gpu_dialect
from ayla.pipelines.tile_matmul_pipeline import TileMatmulPipeline
from ayla.ptx_codegen import make_ptx_kernel


def configuration(stages=5):
    return {
        **_DEFAULTS,
        **dict(
            output_cta_count=4,
            unroll_output_tiles=False,
            output_m_tile=128,
            output_n_tile=168,
            warpgroup_m_tile=128,
            warpgroup_n_tile=168,
            warpgroup_ctas=2,
            operand_k_tile=64,
            operand_stages=stages,
            cyclic_operand_pipeline=True,
            output_tiles_per_cta=1,
            output_slices=1,
            output_store="stmatrix",
            output_store_buffers=1,
            output_fragment_cols=64,
            output_fragment_swizzle="128B",
            output_tma_inner_size=None,
            output_tile_fast_axis="m",
            lhs_l2_cache_hint="evict_first",
            rhs_l2_cache_hint="evict_first",
            dst_l2_cache_hint="evict_last",
        ),
    }


@pytest.mark.parametrize("dtype", ["bf16", "f16", "e4m3"])
def test_general_pipeline_configures_pairing_without_collection_builder(dtype):
    mod = gpu_dialect.parse_module(f"""module {{
      kernel @generic(%a: gmem<768x1024x{dtype}>, %b: gmem<168x1024x{dtype}>, %c: gmem<768x168xbf16>) {{
        %zero: rmem<f32> = constant 0.0
        matmul(%a: gmem<768x1024x{dtype}>, %b: gmem<168x1024x{dtype}>, %c: gmem<768x168xbf16>, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
      }}
    }}""")
    mod = TileMatmulPipeline(
        m=768,
        n=168,
        k=1024,
        output_grid_size=2,
        output_m_tile=128,
        output_n_tile=168,
        warpgroup_m_tile=128,
        warpgroup_n_tile=168,
        operand_k_tile=64,
        operand_stages=5,
        cyclic_operand_pipeline=True,
        output_fragment_cols=64,
        output_fragment_swizzle="128B",
        output_tma_inner_size=None,
        warpgroup_ctas=2,
    ).run(mod)
    [kernel] = mod.body.ops
    assert kernel.attrs["warpgroup_operand_steps"] == 16
    ptx = make_ptx_kernel(mod)
    assert ptx.grid_dim == (1, 1, 1) and ptx.block_dim == (256, 1, 1)
    assert kernel.attrs["paired_warpgroup_ctas"]["rotating"]
    assert "setmaxnreg" not in ptx.ptx


@pytest.mark.parametrize(
    "damage",
    [
        dict(warpgroup_ctas=True),
        dict(warpgroup_ctas=3),
        dict(unroll_output_tiles=True),
        dict(output_cta_count=3),
        dict(output_cta_count=8),
        dict(warpgroup_m_tile=64),
        dict(split_k_parts=2),
        dict(pipeline_output_waves=True),
        dict(pipeline_output_writeback=True),
        dict(transpose_output=True),
    ],
)
def test_jit_declines_pairing_that_cannot_preserve_handoffs(damage):
    config = {**configuration(), **damage}
    assert (
        _seed(config, "mm", 1472, 168, 4096, (1, 1472, 168, 4096), exact=True) is None
    )


def test_jit_does_not_transfer_pairing_to_partial_waves_or_batches():
    config = configuration()
    assert (
        _seed(config, "mm", 1536, 168, 4096, (1, 1472, 168, 4096), exact=False)
        is not None
    )
    assert (
        _seed(config, "mm", 1280, 168, 4096, (1, 1472, 168, 4096), exact=False) is None
    )
    assert (
        _seed(config, "bmm", 1472, 168, 4096, (2, 1472, 168, 4096), exact=True) is None
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(2048, 11008, 4096), (4096, 4096, 4096)])
def test_measured_pairing_is_used_by_default_routes(dtype, shape):
    standalone = select_gemm_schedule("mm", dtype, *shape)
    assert standalone.adaptations == () and standalone.source_shape == shape
    assert standalone.builder_config["warpgroup_ctas"] == 2
    assert standalone.fixed_output_unroll_factor is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("operation", ["mm", "addmm"])
def test_jit_preserves_fragment_and_pairing_configuration(
    dtype, operation
):
    seed = _seed(configuration(), operation, 1472, 168, 4096, (1, 1472, 168, 4096), exact=True)
    assert seed.builder_config["warpgroup_ctas"] == 2
    assert seed.builder_config["output_fragment_cols"] == 64
    assert seed.builder_config["output_fragment_swizzle"] == "128B"
    assert seed.fixed_output_unroll_factor is None
    ptx = make_ptx_kernel(build_gemm_module(operation, dtype, 1, 1472, 168, 4096, seed))
    assert ptx.grid_dim == (2, 1, 1) and ptx.block_dim == (256, 1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("operation", ["mm", "addmm"])
def test_jit_pairing_compile_and_replay_changed_inputs(monkeypatch, dtype, operation):
    m, n, k = 1472, 168, 4096
    import ayla.gemm_jit as jit

    index = len(jit.CONFIGS)
    monkeypatch.setattr(jit, "CONFIGS", (*jit.CONFIGS, configuration()))
    monkeypatch.setitem(jit.SCHEDULES, (operation, dtype), {(1, m, n, k): (index,)})
    op = gemm_op(operation, dtype, namespace=f"test_paired_{operation}_{dtype}")
    typ = getattr(torch, dtype)
    a = torch.zeros(m, k, device="cuda", dtype=typ)
    b = torch.zeros(n, k, device="cuda", dtype=typ)
    bias = torch.zeros(n, device="cuda", dtype=typ)

    def compute(a, b, bias):
        result = torch.empty((m, n), device=a.device, dtype=a.dtype)
        if operation == "addmm":
            op(bias, a, b, result)
        else:
            op(a, b, result)
        return result

    compiled = torch.compile(compute, fullgraph=True)
    compiled(a, b, bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(a, b, bias)
    generator = torch.Generator(device="cuda").manual_seed(20261122)
    for _ in range(2):
        a.copy_(
            torch.randint(-4, 5, a.shape, device="cuda", generator=generator).float()
            / 4
        )
        b.copy_(
            torch.randint(-4, 5, b.shape, device="cuda", generator=generator).float()
            / 4
        )
        bias.copy_(
            torch.randint(-4, 5, bias.shape, device="cuda", generator=generator).float()
            / 4
        )
        result.fill_(float("nan"))
        graph.replay()
        expected = a.double() @ b.double().T
        if operation == "addmm":
            expected += bias.double()
        torch.testing.assert_close(result, expected.to(typ), rtol=0, atol=0)


def test_resident_geometry_does_not_inherit_standalone_pairing():
    from ayla.gemm_jit import select_persistent_gemm_schedule

    seed = select_persistent_gemm_schedule(2048, 11008, 4096, blocks=132)
    assert seed is not None
    assert seed.builder_config["output_grid_size"] == 132
    for key in ("warpgroup_ctas", "output_fragment_cols", "output_fragment_swizzle"):
        assert key not in seed.builder_config
