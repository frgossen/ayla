"""Unfused cuBLAS + stable PyTorch Softplus baseline."""
import torch
import torch.nn.functional as F

NAME = "torch_cublas_then_softplus"
PROBLEM = "softplus"


def prepare(inputs):
    a, b, tau = inputs["a"], inputs["b"], inputs["tau"]

    def run():
        c = torch.mm(a, b.t()).float()
        return F.softplus(c, beta=1.0 / tau, threshold=20.0).bfloat16()

    return run
