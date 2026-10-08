from __future__ import annotations

from types import SimpleNamespace

from benchmarks import benchmark_oss_pytorch_gemm_shapes, profile_oss_pytorch_gemm
from benchmarks.oss_gemm_corpus import seed_configs


def test_profiler_reuses_benchmark_builder_without_dropping_fixed_grid(
    monkeypatch,
):
    dimensions = (2856, 3072, 768)
    schedule = seed_configs(profile_oss_pytorch_gemm._profile_shape(dimensions))[0]
    schedule = schedule.__class__(
        **{
            **schedule.__dict__,
            "output_cta_count": 120,
            "output_tile_fast_axis": "m",
        }
    )
    compute = object()
    observed = []

    def fake_make(shape, config):
        observed.append((shape, config))
        return SimpleNamespace(compute=compute, make_inputs=lambda: []), 0.0, 0.0

    monkeypatch.setattr(
        benchmark_oss_pytorch_gemm_shapes, "_make_ayla_config", fake_make,
    )

    observed_compute, inputs = (
        profile_oss_pytorch_gemm._make_profile_ayla_compute(
            dimensions, schedule, "mm",
        )
    )
    assert observed_compute is compute
    assert inputs == []
    [(shape, forwarded)] = observed
    assert (shape.m, shape.n, shape.k) == dimensions
    assert forwarded.output_cta_count == 120
    assert forwarded.output_tile_fast_axis == "m"


def test_profiler_builds_rank_three_bmm_shape(monkeypatch):
    dimensions = (64, 512, 512)
    shape = profile_oss_pytorch_gemm._profile_shape(
        dimensions, "bmm", batch=64,
    )
    schedule = seed_configs(shape)[0]
    observed = []

    def fake_make(profile_shape, config):
        observed.append((profile_shape, config))
        return SimpleNamespace(compute=object(), make_inputs=lambda: []), 0.0, 0.0

    monkeypatch.setattr(
        benchmark_oss_pytorch_gemm_shapes, "_make_ayla_config", fake_make,
    )

    profile_oss_pytorch_gemm._make_profile_ayla_compute(
        dimensions, schedule, "bmm", batch=64,
    )
    [(profile_shape, forwarded)] = observed
    assert profile_shape.batch == 64
    assert profile_shape.args[0].shape == (64, 64, 512)
    assert profile_shape.args[1].shape == (64, 512, 512)
    assert forwarded is schedule
