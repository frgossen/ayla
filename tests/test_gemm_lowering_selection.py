"""Measured lowering choices preserve the existing schedule selection."""

from dataclasses import replace

import pytest

from ayla.gemm_jit import build_gemm_module, gemm_jit_compiler, select_gemm_schedule
from ayla.ptx_codegen import make_ptx_kernel


def test_opt_in_preserves_schedule_and_transfers_with_catalog_neighbor():
    for m in (512, 520):
        base = select_gemm_schedule("mm", "float16", m, 1536, 1536)
        selected = select_gemm_schedule(
            "mm", "float16", m, 1536, 1536, measured_lowering=True
        )
        assert selected.matrix_refill_pipeline
        assert base == replace(selected, matrix_refill_pipeline=False)
        assert selected.source_shape == (512, 1536, 1536)


def test_no_implicit_shape_rule_or_dtype_transfer():
    # This BF16 query does not use the winning FP16 catalog source.
    selected = select_gemm_schedule(
        "mm", "bfloat16", 1576, 384, 1536, measured_lowering=True
    )
    assert not selected.matrix_refill_pipeline
    # An unrecorded K64/stages8 schedule remains unchanged.
    import ayla.gemm_jit as jit
    from ayla.gemm_schedule_table import SCHEDULES

    sources = {r[3] for r in jit._MATRIX_REFILL_SCHEDULES if r[1] == "float16"}
    for b, m, n, k in SCHEDULES["mm", "float16"]:
        seed = select_gemm_schedule("mm", "float16", m, n, k, b, measured_lowering=True)
        if (
            (b, m, n, k) not in sources
            and seed is not None
            and (
                seed.captured_config["operand_k_tile"] == 64
                and seed.captured_config["operand_stages"] >= 7
            )
        ):
            assert not seed.matrix_refill_pipeline
            break
    else:
        pytest.fail("expected an unrecorded qualifying schedule")


def test_backend_specific_records_and_stale_config_are_declined(monkeypatch):
    import ayla.gemm_jit as jit

    # This source has only a PTX winner; native correctness excluded it.
    assert select_gemm_schedule(
        "mm", "bfloat16", 512, 1024, 4096, measured_lowering=True
    ).matrix_refill_pipeline
    assert not select_gemm_schedule(
        "mm", "bfloat16", 512, 1024, 4096, backend="sass", measured_lowering=True
    ).matrix_refill_pipeline
    monkeypatch.setattr(
        jit,
        "_MATRIX_REFILL_SCHEDULES",
        {("mm", "float16", "ptx", (1, 512, 1536, 1536), "stale")},
    )
    assert not select_gemm_schedule(
        "mm", "float16", 512, 1536, 1536, measured_lowering=True
    ).matrix_refill_pipeline


def test_measured_ptx_emits_partial_wait_and_keeps_final_completion():
    from ayla.base.ir import find_ops_of_type
    from ayla.gpu_dialect import AwaitOp, TmaMoveOp

    seed = select_gemm_schedule(
        "mm", "float16", 512, 1536, 1536, measured_lowering=True
    )
    module = build_gemm_module("mm", "float16", 1, 512, 1536, 1536, seed)
    assert not find_ops_of_type(module, TmaMoveOp)
    assert any(
        a.attrs.get("matrix_refill_pipelined")
        for a in find_ops_of_type(module, AwaitOp)
    )
    assert "gemm_lowering_features" not in module.body.ops[0].attrs
    ptx = make_ptx_kernel(module).ptx
    assert "wgmma.wait_group.sync.aligned 1;" in ptx
    assert "wgmma.wait_group.sync.aligned 0;" in ptx


def test_library_scheduler_enables_overlap_before_descriptor_lowering():
    from ayla.base.ir import find_ops_of_type
    from ayla.gemm_collection.gemm import schedule_matmul
    from ayla.gemm_jit import _module
    from ayla.gpu_dialect import AwaitOp, TmaMoveOp
    from ayla.passes.pipeline_matrix_refills import pipeline_matrix_refills

    seed = select_gemm_schedule("mm", "float16", 512, 1536, 1536)
    config = dict(seed.builder_config)
    config.pop("ptx_optimization_level")
    module = schedule_matmul(
        _module("mm", "float16", 1, 512, 1536, 1536, False),
        **config,
        matrix_refill_pipeline=True,
    )
    assert not find_ops_of_type(module, TmaMoveOp)
    waits = find_ops_of_type(module, AwaitOp)
    assert any(
        a.attrs.get("matrix_refill_pipelined") and a.pending_groups == 1 for a in waits
    )
    assert waits[-1].is_full_wait
    before = str(module)
    pipeline_matrix_refills(module)
    assert str(module) == before


@pytest.mark.parametrize("backend", ("ptx", "sass"))
def test_backend_receives_already_scheduled_ir_without_refill_flag(
    backend, monkeypatch
):
    from ayla.gpu_dialect import AwaitOp
    from ayla.base.ir import find_ops_of_type
    from ayla.passes.pipeline_matrix_refills import PipelineMatrixRefills

    seed = select_gemm_schedule(
        "mm", "float16", 512, 1536, 1536, backend=backend, measured_lowering=True
    )
    module = build_gemm_module("mm", "float16", 1, 512, 1536, 1536, seed)
    assert any(
        a.attrs.get("matrix_refill_pipelined")
        for a in find_ops_of_type(module, AwaitOp)
    )

    def unexpected_backend_reorder(*args, **kwargs):
        pytest.fail("backend reran the library scheduling pass")

    monkeypatch.setattr(PipelineMatrixRefills, "run", unexpected_backend_reorder)
    if backend == "ptx":
        make_ptx_kernel(module)
    else:
        from tests.passes.test_lower_to_sass import requires_assembler

        if requires_assembler.args[0]:
            pytest.skip("requires native assembler")
        from ayla.passes.lower_to_sass import lower_to_sass

        lower_to_sass(module)


@pytest.mark.parametrize("backend", ("ptx", "sass"))
def test_jit_variant_has_distinct_name(backend):
    normal = gemm_jit_compiler("mm", "float16", backend=backend)
    selected = gemm_jit_compiler(
        "mm", "float16", backend=backend, measured_lowering=True
    )
    # Custom calls and specialization caches must distinguish the opt-in route.
    assert normal._kernel.name != selected._kernel.name


def test_graph_schedule_is_not_reinterpreted():
    base = select_gemm_schedule("mm", "float16", 512, 1536, 1536, graph=True)
    selected = select_gemm_schedule(
        "mm", "float16", 512, 1536, 1536, graph=True, measured_lowering=True
    )
    assert base == selected


@pytest.mark.parametrize("backend", ("ptx", "sass"))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_opt_in_jit_executes_and_captures(backend, dtype):
    import torch
    from tests.passes.test_lower_to_sass import requires_assembler, requires_hopper

    if requires_hopper.args[0] or requires_assembler.args[0]:
        pytest.skip("requires Hopper and the native assembler")
    m, n, k = (512, 1536, 1536) if dtype == "float16" else (768, 768, 2048)
    compiler = gemm_jit_compiler("mm", dtype, backend=backend, measured_lowering=True)
    torch.manual_seed(29)
    a = torch.randint(-8, 9, (m, k), device="cuda").to(getattr(torch, dtype)) * 0.125
    b = torch.randint(-8, 9, (n, k), device="cuda").to(a.dtype) * 0.125
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    expected = (a.cpu().float() @ b.cpu().float().T).to(device="cuda", dtype=a.dtype)
    compiler(a, b, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiler(a, b, out)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert compiler.cache_size == 1


@pytest.mark.parametrize("backend", ("ptx", "sass"))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_direct_library_scheduler_executes_without_catalog_or_backend_flag(
    backend, dtype
):
    import torch
    from tests.passes.test_lower_to_sass import requires_assembler, requires_hopper
    from ayla.gemm_collection.gemm import schedule_matmul
    from ayla.gemm_jit import _module
    from ayla.ptx_runtime import assemble_ptx, make_ptx_launcher
    from ayla.passes.lower_to_sass import lower_to_sass, make_sass_launcher

    if requires_hopper.args[0] or requires_assembler.args[0]:
        pytest.skip("requires Hopper and the native assembler")
    m, n, k = (512, 1536, 1536) if dtype == "float16" else (768, 768, 2048)
    seed = select_gemm_schedule("mm", dtype, m, n, k)
    assert not seed.matrix_refill_pipeline
    config = dict(seed.builder_config)
    config.pop("ptx_optimization_level")
    module = schedule_matmul(
        _module("mm", dtype, 1, m, n, k, False), **config, matrix_refill_pipeline=True
    )
    assert "gemm_lowering_features" not in module.body.ops[0].attrs
    torch.manual_seed(29)
    a = torch.randint(-8, 9, (m, k), device="cuda").to(getattr(torch, dtype)) * 0.125
    b = torch.randint(-8, 9, (n, k), device="cuda").to(a.dtype) * 0.125
    out = torch.empty((m, n), device="cuda", dtype=a.dtype)
    expected = (a.cpu().float() @ b.cpu().float().T).to(device="cuda", dtype=a.dtype)
    if backend == "ptx":
        kernel = make_ptx_kernel(module)
        launcher = make_ptx_launcher(
            kernel, assemble_ptx(kernel, "/usr/local/cuda-13/bin/ptxas")
        )
    else:
        launcher = make_sass_launcher(lower_to_sass(module))
    for _ in range(3):
        out.fill_(float("nan"))
        launcher(a, b, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
