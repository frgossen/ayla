"""Pytest setup shared by the test suite.

Selects the CUDA device with the most free memory at session start so that
``device="cuda"`` (which targets the current device) lands on an idle GPU
rather than always cuda:0 — which may be occupied by another user.

Free memory is queried via ``nvidia-smi`` rather than ``torch.cuda`` so we
never create a CUDA context on a full GPU (which would itself OOM).
"""
import subprocess

import torch


def _select_free_gpu() -> None:
    if not torch.cuda.is_available():
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return  # nvidia-smi unavailable — fall back to the default device.
    free = [int(x) for x in out.split()][: torch.cuda.device_count()]
    if not free:
        return
    best = max(range(len(free)), key=lambda i: free[i])
    torch.cuda.set_device(best)
    print(f"[conftest] using cuda:{best} ({free[best]} MiB free)")


_select_free_gpu()
