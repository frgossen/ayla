"""The Muse Glimmer decoder block, compiled through the ayla backend.

An end-to-end check that a real model graph survives the backend: what ayla
covers goes to ayla, what it does not is left to Inductor, and the block still
computes the same thing either way.

Today ayla covers none of the five shapes, so this asserts the fall-through
path. It becomes a positive assertion the moment a kernel is built for one of
them -- which is the point of having it now.
"""
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import glimmer_block  # noqa: E402


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

# Enough rows to be a real decode batch shape without paying for a big one;
# the ~1 GB of weights dominates either way.
TOKENS = 8

# The graph returns a view of an `aten.empty` it never writes: in vLLM the
# attention op fills that buffer, and this dump is the block *without*
# attention. Its contents are whatever the allocator handed back.
UNWRITTEN_OUTPUT = 3


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
def test_glimmer_block_compiles_through_the_backend():
    import torch._inductor.config as inductor_config
    from torch._inductor.utils import run_and_get_code

    inputs = glimmer_block.make_inputs(TOKENS)
    with torch.no_grad():
        expected = glimmer_block.mod(*inputs)

        # The backend is not in the Inductor cache key, so a warm cache would
        # serve a stock graph and the lowering would never run.
        with inductor_config.patch(force_disable_caches=True):
            compiled = torch.compile(glimmer_block.mod, dynamic=False)
            actual, code = run_and_get_code(compiled, *inputs)

    # None of the five shapes is in the collection, so every matmul is
    # declined and falls back rather than being mis-claimed.
    assert not any("ayla_" in graph for graph in code)

    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected)):
        if i == UNWRITTEN_OUTPUT:
            continue
        # Tolerance scaled to the output's own range: these span 8 to 450
        # across the six, and bf16 accumulated over K up to 19968 does not
        # reproduce eager bit for bit.
        scale = want.float().abs().max().clamp(min=1.0).item()
        torch.testing.assert_close(got, want, atol=2e-2 * scale, rtol=2e-2)


def test_block_weight_traffic_is_the_decode_floor():
    """The five weight matrices, which a decode step streams in full.

    Guards the shapes the backend would have to cover, and the traffic figure
    the benchmark reports against.
    """
    assert [name for name, _, _ in glimmer_block.GEMM_SHAPES] == [
        "gate", "o_proj", "gate_up", "down", "qkv",
    ]
    assert glimmer_block.WEIGHT_BYTES == 967_835_648

    # gate_up alone is over half the traffic, and half the block's runtime.
    gate_up = next(k * n * 2 for name, k, n in glimmer_block.GEMM_SHAPES
                   if name == "gate_up")
    assert gate_up / glimmer_block.WEIGHT_BYTES > 0.5
