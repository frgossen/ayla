import warnings

import pytest

from ayla.gemm_schedule_metrics import (
    GemmScheduleMetrics,
    GemmScheduleRejected,
    gemm_schedule_metrics,
    validate_compiled_gemm,
)


@pytest.mark.parametrize("tiles,slots,rejected", [(97, 100, True), (98, 100, False),
                                                   (99, 100, False)])
def test_tail_cutoff(tiles, slots, rejected):
    metrics = GemmScheduleMetrics(tiles, slots, 100, True)
    if rejected:
        with pytest.raises(GemmScheduleRejected, match="below 98%"):
            metrics.validate()
    else:
        metrics.validate()


def test_realized_groups_and_batch_not_requested_group_proxy():
    config = dict(output_m_tile=64, output_n_tile=64, output_tiles_per_cta=6)
    metrics = gemm_schedule_metrics(config, 20 * 64, 64, batch=12)
    assert metrics.logical_tiles == metrics.output_slots == 240
    assert metrics.grid_ctas == 48
    metrics.validate()
    # Ten tiles grouped by four really have twelve slots.
    config["output_tiles_per_cta"] = 4
    with pytest.raises(GemmScheduleRejected):
        gemm_schedule_metrics(config, 10 * 64, 64).validate()


def test_fixed_grid_overrides_groups_and_transposes_output():
    config = dict(output_m_tile=64, output_n_tile=128, transpose_output=True,
                  output_tiles_per_cta=9, output_n_tiles_per_cta=9,
                  output_cta_count=4)
    metrics = gemm_schedule_metrics(config, 128, 256, batch=3)
    assert metrics.logical_tiles == metrics.output_slots == metrics.grid_ctas == 12


def test_edge_padding_is_separate_from_output_slots():
    metrics = gemm_schedule_metrics(dict(output_m_tile=64, output_n_tile=64), 65, 65)
    assert metrics.output_tail_efficiency == 1
    metrics.validate()


def test_native_builder_grid_arguments_are_checked():
    config = dict(output_m_tile=64, output_n_tile=64, output_grid_size=3,
                  output_m_tiles_per_cta=1)
    with pytest.raises(GemmScheduleRejected, match="below 98%"):
        gemm_schedule_metrics(config, 128, 128).validate()


def test_selector_skips_bad_exact_head_and_keeps_next_candidate(monkeypatch):
    from ayla import gemm_jit

    shape = (1, 128, 128, 256)
    legal = gemm_jit.select_gemm_schedule("mm", "float16", *shape[1:]).captured_config
    legal = dict(legal, output_cta_count=None, output_tiles_per_cta=1,
                 output_n_tiles_per_cta=1)
    bad = dict(legal, output_cta_count=1000)
    monkeypatch.setattr(gemm_jit, "CONFIGS", [bad, legal])
    monkeypatch.setattr(gemm_jit, "SCHEDULES", {("mm", "float16"): {shape: (0, 1)}})
    seed = gemm_jit.select_gemm_schedule("mm", "float16", *shape[1:])
    assert seed is not None and seed.captured_config["output_cta_count"] is None


def test_spills_rejected_even_with_low_register_count():
    with pytest.raises(GemmScheduleRejected, match="spill backing"):
        validate_compiled_gemm(dict(local_bytes_per_thread=16,
                                    registers_per_thread=32,
                                    resident_ctas_on_device=132, grid_ctas=132))


@pytest.mark.parametrize("grid,warns", [(69, True), (70, False), (100, False),
                                        (101, True), (140, False)])
def test_wave_warning_uses_compiled_capacity_and_wave_rounding(grid, warns):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_compiled_gemm(dict(local_bytes_per_thread=0,
                                    resident_ctas_on_device=100, grid_ctas=grid))
    assert bool(caught) == warns
    if caught:
        assert "below 70%" in str(caught[0].message)


def test_no_residency_is_rejected():
    with pytest.raises(GemmScheduleRejected, match="no resident CTA"):
        validate_compiled_gemm(dict(local_bytes_per_thread=0,
                                    resident_ctas_on_device=0, grid_ctas=132))


def test_loaded_kernel_reports_actual_spill_backing():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from ayla.gemm_jit import _DEFAULTS, _seed, build_gemm_module
    from ayla.gemm_schedule_table import CONFIGS
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher

    torch.empty(1, device="cuda")  # Establish the current primary CUDA context.
    shape = (1, 12608, 2304, 768)
    old = _seed(_DEFAULTS | CONFIGS[1987], "addmm", *shape[1:], shape, exact=True)
    assert old is not None
    launcher = make_ptx_launcher(make_ptx_kernel(
        build_gemm_module("addmm", "float16", *shape, old)))
    resources = launcher.resources()
    assert resources["local_bytes_per_thread"] > 0
    with pytest.raises(GemmScheduleRejected, match="spill backing"):
        validate_compiled_gemm(resources)
