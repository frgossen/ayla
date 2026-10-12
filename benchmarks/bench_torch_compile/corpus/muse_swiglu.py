"""The Muse Glimmer packed SwiGLU MLP captured from vLLM."""
from __future__ import annotations

import torch

from ayla.gemm_collection.muse_swiglu import (
    MUSE_SWIGLU_HIDDEN,
    MUSE_SWIGLU_INTERMEDIATE,
    MUSE_SWIGLU_M_VALUES,
    MUSE_SWIGLU_PACKED_INTERMEDIATE,
)
from benchmarks.common import BenchConfig


def muse_swiglu(
    lhs: torch.Tensor,
    packed_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Run Muse's packed SwiGLU projection and down projection in Torch."""
    # The captured vLLM post-grad FX graph represents its packed gate/up
    # projection as one matmul followed by two slices; preserve that structure
    # so this corpus exercises the graph pattern we intend to replace.
    projection = lhs @ packed_weight.T
    gate = projection[:, :MUSE_SWIGLU_INTERMEDIATE]
    up = projection[:, MUSE_SWIGLU_INTERMEDIATE:]
    return (torch.nn.functional.silu(gate) * up) @ down_weight.T


def build_config(m: int) -> BenchConfig:
    """Build one concrete Muse SwiGLU case."""
    if m not in MUSE_SWIGLU_M_VALUES:
        raise ValueError(
            f"unsupported M={m}; expected one of {MUSE_SWIGLU_M_VALUES}"
        )

    def make_inputs() -> list[torch.Tensor]:
        torch.manual_seed(20260921 + m)
        return [
            torch.randn(
                (m, MUSE_SWIGLU_HIDDEN),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.randn(
                (MUSE_SWIGLU_PACKED_INTERMEDIATE, MUSE_SWIGLU_HIDDEN),
                device="cuda",
                dtype=torch.bfloat16,
            ) * .01,
            torch.randn(
                (MUSE_SWIGLU_HIDDEN, MUSE_SWIGLU_INTERMEDIATE),
                device="cuda",
                dtype=torch.bfloat16,
            ) * .01,
        ]

    return BenchConfig(
        name=f"muse_swiglu M={m} bf16",
        make_inputs=make_inputs,
        compute=muse_swiglu,
    )


def build_configs() -> list[BenchConfig]:
    """What this module contributes to the torch.compile corpus."""
    return [build_config(m) for m in MUSE_SWIGLU_M_VALUES]
