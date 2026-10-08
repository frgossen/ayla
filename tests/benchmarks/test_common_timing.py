import pytest
import torch

import benchmarks.common as benchmark_common
from benchmarks.common import (
    _consume_output_read_only,
    gemm_bench_config,
    out_parameter_op,
    run,
    torch_mm_nt,
)
from benchmarks.oss_gemm_corpus import require_same_timing, timing_protocol
from tests.testing_utils import requires_cuda


def test_rows_without_a_protocol_are_legacy() -> None:
    rows = [
        {"latency_ms": 1.0},
        {"latency_ms": None, "timing_protocol": "inference"},
    ]

    assert timing_protocol(rows, "old.json") == "legacy"


def test_mixed_protocols_are_rejected() -> None:
    old = [{"latency_ms": 1.0}]
    new = [{"latency_ms": 1.0, "timing_protocol": "inference"}]

    require_same_timing(("a", new), ("b", new), ("errors", [{}]))
    with pytest.raises(ValueError, match="mixes"):
        timing_protocol(old + new, "merged.json")
    with pytest.raises(ValueError, match="not comparable"):
        require_same_timing(("old", old), ("new", new))


def test_inference_protocol_versions_are_incompatible() -> None:
    v1 = [{"latency_ms": 1.0, "timing_protocol": "inference"}]
    v2 = [{"latency_ms": 1.0, "timing_protocol": "inference-v2"}]

    with pytest.raises(ValueError, match="not comparable"):
        require_same_timing(("v1", v1), ("v2", v2))


def test_inference_consumer_does_not_rewrite_output() -> None:
    output = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    before = output.clone()

    first = _consume_output_read_only(output)
    second = _consume_output_read_only(output)

    assert torch.equal(output, before)
    assert first.item() == second.item() == before.sum().item()


@requires_cuda
def test_inference_gives_every_layer_its_own_weights(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_common, "_MAX_ROUNDS", 3)
    monkeypatch.setattr(benchmark_common, "_WARMUP_ITERS", 1)
    weight_ptrs = set()

    def mm(a: torch.Tensor, bt: torch.Tensor, out: torch.Tensor) -> None:
        weight_ptrs.add(bt.data_ptr())
        torch.mm(a, bt.t(), out=out)

    configs = [
        gemm_bench_config(
            name=name, m=256, n=256, k=256, compute=compute,
            dtype=torch.float16,
        )
        for name, compute in (
            ("torch", torch_mm_nt), ("out", out_parameter_op(mm)),
        )
    ]
    # 128 KiB of weights per layer: the budget allows more than the cap.
    torch_result, out_result = run(configs, inference=True)

    layers = benchmark_common._INFERENCE_MAX_LAYERS
    assert torch_result.iters == out_result.iters == layers
    assert len(weight_ptrs) == layers
    assert torch_result.median_ms > 0 and out_result.median_ms > 0
