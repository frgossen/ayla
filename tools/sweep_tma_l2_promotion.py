#!/usr/bin/env python3
"""Sweep TMA tensor-map L2 promotion on the two-tile STSM GEMM."""
from __future__ import annotations

import statistics
import time

import torch

from ayla.base.ir import find_ops
from ayla.gpu_dialect import MoveOp
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


# Import the builder without triggering its module-level torch registration.
import ayla.ptx_runtime as _runtime
import ayla.torch_custom_call as _custom_call

_runtime.make_ptx_launcher = lambda kernel: None
_custom_call.make_torch_custom_call = lambda *args, **kwargs: None

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store_k3 import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_tma_store_k3_module,
)


PROMOTIONS = ("64B", "128B", "256B")
SCOPES = ("A", "B", "A+B", "output")


def _launcher(scope: str | None, promotion: str | None):
    module = auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
        output_tiles_per_cta=2,
    )
    for move in find_ops(module, lambda op: isinstance(op, MoveOp)):
        label = str(move.attrs.get("label", ""))
        selected = (
            scope == "A" and "_lhs_smem" in label
            or scope == "B" and "_rhs_smem" in label
            or scope == "A+B" and (
                "_lhs_smem" in label or "_rhs_smem" in label
            )
            or scope == "output" and "_dst_tma" in label
        )
        if selected:
            move.attrs["l2_promotion"] = promotion
    return make_ptx_launcher(make_ptx_kernel(module))


def _measure(launcher, a, b, c) -> tuple[float, float, float]:
    time.sleep(1.0)
    for _ in range(100):
        launcher(a, b, c)
    torch.cuda.synchronize()

    rounds, iterations = 10, 200
    pairs = [
        (torch.cuda.Event(enable_timing=True),
         torch.cuda.Event(enable_timing=True))
        for _ in range(rounds)
    ]
    for start, end in pairs:
        start.record()
        for _ in range(iterations):
            launcher(a, b, c)
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) / iterations for start, end in pairs]
    median_ms = statistics.median(samples)
    stdev_us = statistics.stdev(samples) * 1e3
    tflops = 2 * 4096**3 / (median_ms * 1e-3) / 1e12
    return median_ms, stdev_us, tflops


def main() -> None:
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    variants = [("default", _launcher(None, None))]
    variants += [
        (f"{scope}={promotion}", _launcher(scope, promotion))
        for scope in SCOPES
        for promotion in PROMOTIONS
    ]
    # Re-measure default at the end to expose run-order/thermal drift.
    variants.append(("default (repeat)", _launcher(None, None)))

    for name, launcher in variants:
        ms, stdev_us, tflops = _measure(launcher, a, b, c)
        print(
            f"{name:20s} {ms * 1e3:8.2f} us  "
            f"sd={stdev_us:5.2f} us  {tflops:8.2f} TFLOP/s"
        )


if __name__ == "__main__":
    main()
