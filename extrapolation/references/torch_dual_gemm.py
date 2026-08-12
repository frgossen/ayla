"""Unfused two-cuBLAS-call correctness/performance baseline."""
import torch

NAME = "torch_two_cublas_gemms"
PROBLEM = "dual_gemm"


def prepare(inputs):
    a, b1, b2 = inputs["a"], inputs["b1"], inputs["b2"]
    intermediate = torch.empty((a.shape[0], b1.shape[0]), device=a.device, dtype=a.dtype)
    output = torch.empty((a.shape[0], b2.shape[0]), device=a.device, dtype=a.dtype)

    def run():
        torch.mm(a, b1.t(), out=intermediate)
        torch.mm(intermediate, b2.t(), out=output)
        return output

    return run
