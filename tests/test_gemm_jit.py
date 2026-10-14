import pytest

from ayla.gemm_jit import gemm_jit_compiler, select_gemm_schedule
from ayla.gemm_schedule_table import CONFIGS, SCHEDULES


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
    monkeypatch.setattr(compiler, "_metadata", lambda _: (("cpu",), shapes))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    launcher = compiler._launcher(())
    assert launcher.name == operation
    assert compiler.cache_size == 1
