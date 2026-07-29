#!/usr/bin/env python3
"""Dump PTX for the acceptance-criterion kernels to files for before/after diffing.

Codegen is pure Python (no CUDA needed), so this runs on a GPU-less box.

Usage:
    python tools/dump_ptx.py baseline   # capture main state before edits
    python tools/dump_ptx.py current    # capture after edits
    python tools/dump_ptx.py            # defaults to "current"

Then compare:
    diff -u tools/ptx_baseline/auto_matmul_4096.ptx \
            tools/ptx_current/auto_matmul_4096.ptx
"""
from __future__ import annotations

import sys
from pathlib import Path

# Importing the kernel modules self-registers a torch custom call, which builds
# a CUDA C++ extension (needs ninja + a GPU). Codegen itself is pure Python, so
# stub out the runtime/registration hooks before importing the kernels.
import ayla.ptx_runtime as _rt
import ayla.torch_custom_call as _tcc

_rt.make_ptx_launcher = lambda kernel: None
_tcc.make_torch_custom_call = lambda *args, **kwargs: None

from ayla.gemm_collection.auto_matmul_4096 import auto_matmul_4096_module
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg import (
    auto_matmul_4096_wgmma256_wg_module,
)
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_storepipe import (
    auto_matmul_4096_wgmma256_wg_storepipe_module,
)
from ayla.ptx_codegen import make_ptx_kernel

KERNELS = {
    "auto_matmul_4096": auto_matmul_4096_module,
    "auto_matmul_4096_wgmma256_wg": auto_matmul_4096_wgmma256_wg_module,
    # Store-pipelining variant at a few reduction factors.
    "auto_matmul_4096_wgmma256_wg_storepipe_R1": (
        lambda: auto_matmul_4096_wgmma256_wg_storepipe_module(reduction_factor=1)
    ),
    "auto_matmul_4096_wgmma256_wg_storepipe_R4": (
        lambda: auto_matmul_4096_wgmma256_wg_storepipe_module(reduction_factor=4)
    ),
}


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    out_dir = Path(__file__).parent / f"ptx_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, build_module in KERNELS.items():
        ptx = make_ptx_kernel(build_module()).ptx
        path = out_dir / f"{name}.ptx"
        path.write_text(ptx)
        print(f"wrote {path} ({len(ptx)} chars)")


if __name__ == "__main__":
    main()
