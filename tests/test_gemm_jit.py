import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_jit import (
    _DEFAULTS,
    _module,
    _seed,
    gemm_jit_compiler,
    select_gemm_schedule,
)
from ayla.gemm_schedule_table import CONFIGS, SCHEDULES
from ayla.gpu_dialect import MatmulOp


def test_measured_shape_selects_its_winner():
    [((batch, m, n, k), seeds)] = list(SCHEDULES["mm", "bfloat16"].items())[:1]
    schedule = select_gemm_schedule("mm", "bfloat16", m, n, k, batch)
    assert schedule is not None
    assert schedule.adaptations == ()
    assert schedule.source_shape == (m, n, k)
    assert all(
        schedule.captured_config[key] == value
        for key, value in CONFIGS[seeds[0]].items()
    )


def test_unmeasured_shape_borrows_a_legal_nearby_schedule():
    assert (1, 4100, 4096, 4096) not in SCHEDULES["mm", "float16"]
    schedule = select_gemm_schedule("mm", "float16", 4100, 4096, 4096)
    assert schedule is not None
    assert schedule.adaptations[0].startswith("nearest measured shape")
    assert schedule.builder_config is not None
    assert 4096 % int(schedule.builder_config["operand_k_tile"]) == 0
    assert 4096 >= int(schedule.builder_config["output_n_tile"])


def test_nothing_legal_is_declined():
    # K = 32 divides neither operand K tile.
    assert select_gemm_schedule("mm", "float16", 4096, 4096, 32) is None
    assert select_gemm_schedule("mm", "float32", 4096, 4096, 4096) is None
    # N = 65 leaves the output rows unaligned for TMA.
    assert select_gemm_schedule("addmm", "bfloat16", 96, 65, 1024) is None


@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm"])
def test_specialization_builds_the_table_schedule(operation, monkeypatch):
    import torch

    [(batch, m, n, k)] = list(SCHEDULES[operation, "bfloat16"])[:1]
    lead = (batch,) if operation == "bmm" else ()
    shapes = [(*lead, m, k), (*lead, n, k), (*lead, m, n)]
    if operation == "addmm":
        shapes.insert(0, (n,))
    compiler = gemm_jit_compiler(operation, "bfloat16")
    monkeypatch.setattr(
        compiler,
        "_metadata",
        lambda _: (
            torch.device("cuda:0"),
            tuple(
                (
                    tuple(shape),
                    tuple(reversed(range(1, len(shape) + 1))),
                    torch.bfloat16,
                )
                for shape in shapes
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    launcher = compiler._launcher(())
    assert launcher.name == operation
    assert compiler.cache_size == 1


def test_addmm_transpose_uses_existing_column_major_view():
    module = _module("addmm", "bfloat16", 1, 256, 512, 128, True)
    [mm] = find_ops_of_type(module, MatmulOp)
    assert list(mm.lhs.type.sizes) == [512, 128]
    assert list(mm.dst.type.sizes) == [512, 256]
    assert list(mm.dst.type.strides) == [1, 512]
    assert list(mm.init.type.strides) == [1, 0]


def test_addmm_transpose_can_transfer_when_geometry_is_legal():
    incumbent = select_gemm_schedule("addmm", "bfloat16", 16384, 4096, 1024)
    cfg = {**_DEFAULTS, **incumbent.captured_config, "transpose_output": True}
    assert (
        _seed(cfg, "addmm", 16384, 4096, 1024, (1, 16384, 4096, 1024), exact=False)
        is not None
    )


@pytest.mark.parametrize('operation', ['mm', 'bmm', 'addmm'])
@pytest.mark.parametrize('m,n,k,parts', [(5, 5, 264, 3), (64, 128, 512, 4), (5, 5, 64, 3)])
def test_reduction_schedule_builds_generic_phases_for_tails_and_batches(operation, m, n, k, parts):
    import math
    from ayla.gemm_jit import GemmScheduleSeed, build_gemm_module
    from ayla.passes.tile_reduce import ReductionModules
    from ayla.gpu_dialect import KernelOp
    from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
    from ayla.ptx_codegen import make_ptx_kernel

    kwargs = dict(output_m_tile=64, output_n_tile=8 if n < 8 else 128,
                  operand_k_tile=64, operand_stages=2, output_store='scalar',
                  output_tma_inner_size=None, ptx_optimization_level=None)
    seed = GemmScheduleSeed({}, kwargs, (), None, None, reduction_parts=parts,
                            reduction_load_unroll=4)
    stages = build_gemm_module(operation, 'float16', 2 if operation == 'bmm' else 1,
                               m, n, k, seed)
    assert isinstance(stages, ReductionModules)
    assert math.prod(stages.workspace_shape) == parts * (2 if operation == 'bmm' else 1) * m * ((n + 7) // 8) * 8
    [final_kernel] = find_ops_of_type(stages.reduction, KernelOp)
    batch = 2 if operation == "bmm" else 1
    assert final_kernel.grid_dim[0] == math.ceil(batch * m * n / 128)
    assert final_kernel.grid_dim[2] in (None, 1)
    for phase in (stages.partials, stages.reduction):
        VerifyGpuDialect().run(phase)
        assert make_ptx_kernel(phase).ptx



def test_reduction_schedule_does_not_transfer_without_measurements():
    captured = {**_DEFAULTS, **CONFIGS[0], 'split_k_parts': 4}
    assert _seed(captured, 'mm', 4096, 4096, 4096, (1, 4096, 4096, 4096), exact=False) is None
