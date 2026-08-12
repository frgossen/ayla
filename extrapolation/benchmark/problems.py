"""Canonical inputs, references, and correctness checks for all problems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Problem:
    name: str
    flops: int
    make_inputs: Callable[[str, int], dict]
    reference: Callable[[dict], object]
    check: Callable[[object, object], dict]


def _generator(device: str, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _bf16(shape, device, generator):
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)


def _float_check(actual, expected, *, atol: float, rtol: float) -> dict:
    if not isinstance(actual, torch.Tensor):
        raise TypeError(f"candidate returned {type(actual).__name__}, expected Tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"shape/dtype mismatch: got {actual.shape}/{actual.dtype}, "
            f"expected {expected.shape}/{expected.dtype}"
        )
    diff = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "exact_fraction": (actual == expected).float().mean().item(),
    }


def _gemm_inputs(device, seed):
    g = _generator(device, seed)
    # B is physically [N,K] row-major. This is the same storage as a
    # mathematical [K,N] column-major matrix, and every backend computes A@B.T.
    return {"a": _bf16((4096, 4096), device, g), "b": _bf16((4096, 4096), device, g)}


def _gemm_ref(x):
    return torch.mm(x["a"], x["b"].t())


def _dual_inputs(device, seed):
    g = _generator(device, seed)
    return {
        "a": _bf16((4096, 4096), device, g),
        "b1": _bf16((128, 4096), device, g),
        "b2": _bf16((4096, 128), device, g),
    }


def _dual_ref(x):
    intermediate = torch.mm(x["a"], x["b1"].t())
    return torch.mm(intermediate, x["b2"].t())


def _softplus_inputs(device, seed):
    values = _gemm_inputs(device, seed)
    values["tau"] = 1.0
    return values


def _softplus_ref(x):
    c = torch.mm(x["a"], x["b"].t()).float()
    return F.softplus(c, beta=1.0 / x["tau"], threshold=20.0).bfloat16()


def _quant_inputs(device, seed):
    return _gemm_inputs(device, seed)


def _quant_ref(x):
    c = torch.mm(x["a"], x["b"].t()).float()
    scales = c.abs().amax(dim=1, keepdim=True).clamp_min_(1e-7).div_(127.0)
    y = torch.round(c / scales).clamp_(-128, 127).to(torch.int8)
    return y, scales


def _quant_check(actual, expected):
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise TypeError("dynamic_quant candidate must return (int8_y, fp32_scales)")
    y, scales = actual
    ref_y, ref_scales = expected
    if y.shape != ref_y.shape or y.dtype != torch.int8:
        raise AssertionError(f"Y must be {ref_y.shape}/int8; got {y.shape}/{y.dtype}")
    if scales.shape != ref_scales.shape or scales.dtype != torch.float32:
        raise AssertionError(
            f"scales must be {ref_scales.shape}/float32; got {scales.shape}/{scales.dtype}"
        )
    scale_diff = (scales - ref_scales).abs()
    torch.testing.assert_close(scales, ref_scales, atol=0.02, rtol=0.02)
    int_diff = (y.to(torch.int16) - ref_y.to(torch.int16)).abs()
    off_by_more_than_one = (int_diff > 1).float().mean().item()
    if off_by_more_than_one > 0.01:
        raise AssertionError(
            f"{off_by_more_than_one:.3%} of quantized values differ by more than one"
        )
    return {
        "scale_max_abs": scale_diff.max().item(),
        "scale_mean_abs": scale_diff.mean().item(),
        "int_max_abs": int_diff.max().item(),
        "int_exact_fraction": (int_diff == 0).float().mean().item(),
        "int_off_by_more_than_one_fraction": off_by_more_than_one,
    }


PROBLEMS = {
    "bf16_gemm": Problem(
        "bf16_gemm", 2 * 4096**3, _gemm_inputs, _gemm_ref,
        lambda a, e: _float_check(a, e, atol=2.0, rtol=0.02),
    ),
    "dual_gemm": Problem(
        "dual_gemm", 2 * 4096 * 4096 * 128 + 2 * 4096 * 128 * 4096,
        _dual_inputs, _dual_ref,
        lambda a, e: _float_check(a, e, atol=16.0, rtol=0.04),
    ),
    "softplus": Problem(
        "softplus", 2 * 4096**3, _softplus_inputs, _softplus_ref,
        lambda a, e: _float_check(a, e, atol=2.0, rtol=0.02),
    ),
    "dynamic_quant": Problem(
        "dynamic_quant", 2 * 4096**3, _quant_inputs, _quant_ref, _quant_check,
    ),
}
