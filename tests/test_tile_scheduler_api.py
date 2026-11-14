"""Public workload API, isolated profiling, and replay with changed inputs."""

import json

import pytest
import torch

from ayla.tile_scheduler import compile_workload
from ayla.profiling.sm_overlap import plot_capture, profile_workload


def _chain(x, weight, down, unused, rows):
    projected = x @ weight.T
    fp32 = projected.float()
    normalized = (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
    return {"normalized": normalized, "result": normalized.view(rows, -1) @ down.T,
            "again": (normalized, None)}


def test_scheduler_validation_precedes_compilation():
    with pytest.raises(ValueError, match="scheduler"):
        compile_workload(lambda x: x, [], scheduler="dynamic")
    with pytest.raises(ValueError, match="nonempty"):
        compile_workload(lambda x: x, [])
    with pytest.raises(ValueError, match="CUDA"):
        compile_workload(lambda x: x, [torch.ones(2)])


@pytest.mark.parametrize("option", ["samples", "rounds", "replays", "capacity"])
def test_profile_rejects_invalid_counts_before_gpu_work(tmp_path, option):
    with pytest.raises(ValueError, match=option):
        profile_workload(lambda x: x, [], output=tmp_path, **{option: 0})
    assert not list(tmp_path.iterdir())


def test_replot_accepts_arbitrary_variant_names_without_cuda(tmp_path, monkeypatch):
    def no_gpu(*args, **kwargs):
        raise AssertionError("offline plotting must not query CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_name", no_gpu)
    data = dict(name="different workload", gpu="saved GPU", variants={
        "custom schedule": dict(report={"plain_ms": 1.1, "observer_overhead_pct": .1},
            profiler_kernels=[dict(name="kernel", start_ms=0., duration_ms=1.)],
            intervals=[dict(sm=7, cta=0, root=0, kind="other GEMM", start_ms=0., end_ms=.9)],
            lifetimes=[dict(sm=7, cta=0, start_ms=0., end_ms=1.)]),
        "custom reference": dict(report={"plain_ms": 1.2},
            profiler_kernels=[dict(name="gemm", start_ms=0., duration_ms=1.)],
            intervals=[], lifetimes=[]),
    })
    plot_capture(data, tmp_path)
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"sm_overlap.{suffix}").stat().st_size > 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("closed_over", [False, True])
def test_callable_outputs_and_changed_input_graph_replay(closed_over):
    torch.manual_seed(12)
    inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .05
              for shape in ((256, 256), (512, 256), (256, 512), (1,))]
    inputs.append(256)
    if closed_over:
        weight = inputs[1]

        def compute(x, down, unused, rows):
            return _chain(x, weight, down, unused, rows)
        inputs = [inputs[0], inputs[2], inputs[3], inputs[4]]
    else:
        compute = _chain
    resident = compile_workload(compute, inputs, scheduler=None)
    static = compile_workload(compute, inputs)
    for workload in (resident, static):
        expected = compute(*inputs)
        actual = workload(*inputs)
        torch.testing.assert_close(actual, expected, rtol=.015, atol=.015)
        assert actual["normalized"] is actual["again"][0]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workload(*inputs)
        for _ in range(3):
            inputs[0].normal_(std=.05)
            graph.replay()
            torch.testing.assert_close(actual, compute(*inputs), rtol=.015, atol=.015)
        with pytest.raises(ValueError, match="compiled"):
            workload(*([inputs[0][:128]] + inputs[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_profile_supplied_workload_and_reference(tmp_path):
    inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .05
              for shape in ((256, 256), (512, 256), (256, 512), (1,))]
    inputs.append(256)
    data = profile_workload(_chain, inputs, output=tmp_path,
                            name="small chain", samples=1, rounds=1, replays=2)
    assert set(data["variants"]) == {"resident", "static", "reference"}
    assert data["reference_sm_activity_available"] is False
    assert json.loads((tmp_path / "sm_overlap.json").read_text()) == data
    for name in ("resident", "static"):
        variant = data["variants"][name]
        assert variant["report"]["observer_outputs_bitwise_equal"]
        assert variant["intervals"] and variant["lifetimes"]
        assert variant["report"]["plain_ms"] > 0
    plot_capture(data, tmp_path)
