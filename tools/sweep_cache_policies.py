#!/usr/bin/env python3
"""Sweep scalar global-store cache policies on the 4096 WGMMA kernel.

The fastest STSM kernel uses TMA for both global loads and stores.  PTX does
not define cache operators or eviction priorities for those tensor-copy
instructions, so the policy sweep uses its scalar-store sibling and reports
the TMA/STSM kernel as the current reference.
"""
from __future__ import annotations

import statistics
import time

import torch

from ayla.base.ir import find_ops
from ayla.gpu_dialect import GmemTensorType, MoveOp
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


# Import builders without paying for their module-level torch registration.
import ayla.ptx_runtime as _runtime
import ayla.torch_custom_call as _custom_call

_runtime.make_ptx_launcher = lambda kernel: None
_custom_call.make_torch_custom_call = lambda *args, **kwargs: None

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_module,
)
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store_k3 import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_tma_store_k3_module,
)


POLICIES = [
    ("default", {}),
    *((f"cache={v}", {"cache": v}) for v in ("wb", "cg", "cs", "wt")),
    *((f"evict={v}", {"evict": v}) for v in (
        "normal", "unchanged", "first", "last", "no_allocate",
    )),
]


def _scalar_store_launcher(attrs: dict[str, str]):
    module = auto_matmul_4096_wgmma256_wg_module()
    stores = find_ops(
        module,
        lambda op: isinstance(op, MoveOp)
        and isinstance(op.dst.type, GmemTensorType),
    )
    assert stores, "expected at least one scalar global store"
    for store in stores:
        store.attrs.update(attrs)
    return make_ptx_launcher(make_ptx_kernel(module))


def _reference_launcher():
    module = auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
        output_tiles_per_cta=2,
    )
    return make_ptx_launcher(make_ptx_kernel(module))


def _measure(launcher, a, b, c) -> tuple[float, float]:
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
    tflops = 2 * 4096**3 / (median_ms * 1e-3) / 1e12
    return median_ms, tflops


def main() -> None:
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    variants = [("TMA/STSM reference", _reference_launcher())]
    variants += [
        (f"scalar {name}", _scalar_store_launcher(attrs))
        for name, attrs in POLICIES
    ]
    for name, launcher in variants:
        ms, tflops = _measure(launcher, a, b, c)
        print(f"{name:31s} {ms * 1e3:8.2f} us  {tflops:8.2f} TFLOP/s")


if __name__ == "__main__":
    main()
