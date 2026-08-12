"""Unfused cuBLAS + PyTorch row-wise dynamic quantization baseline."""
import torch

NAME = "torch_cublas_then_dynamic_quant"
PROBLEM = "dynamic_quant"


def prepare(inputs):
    a, b = inputs["a"], inputs["b"]

    def run():
        c = torch.mm(a, b.t()).float()
        scales = c.abs().amax(dim=1, keepdim=True).clamp_min_(1e-7).div_(127.0)
        y = torch.round(c / scales).clamp_(-128, 127).to(torch.int8)
        return y, scales

    return run
