"""Profile a single GEMM kernel with NVIDIA Nsight Compute.

Usage:
  ncu --set roofline -o profile python benchmarks/profile_gemm.py
  ncu --set full -o profile python benchmarks/profile_gemm.py
Then open profile.ncu-rep in the Nsight Compute UI.
"""

import torch

torch.zeros(1, device="cuda")
import ayla.gemm_collection  # noqa: E402

M, N, K = 4096, 4096, 4096
dtype = torch.bfloat16
op = torch.ops.ayla.fast_mma_matmul_4096

a = torch.randn(M, K, dtype=dtype, device="cuda")
b = torch.randn(K, N, dtype=dtype, device="cuda")

# Warmup
for _ in range(5):
    op(a, b)
torch.cuda.synchronize()

op(a, b)
torch.cuda.synchronize()
