"""cuBLAS BF16 4096 GEMM baseline through torch.mm."""
import torch

NAME = "torch_cublas_bf16"
PROBLEM = "bf16_gemm"
METADATA = {"backend": "PyTorch torch.mm / cuBLAS"}


def prepare(inputs):
    a, b = inputs["a"], inputs["b"]
    output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)

    def run():
        return torch.mm(a, b.t(), out=output)

    return run
