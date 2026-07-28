"""Correctness tests for all GEMM kernels in the collection."""

import ayla.gemm_collection  # noqa: F401  (registers torch.ayla.ayla.*)
import pytest
import torch
from torch.ops import ayla

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

F32 = torch.float32
BF16 = torch.bfloat16


@requires_cuda
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize(
    "op,dtype",
    [
        pytest.param(lambda a, b: torch.mm(a, b.t()), F32, id="torch.mm"),
        pytest.param(ayla.naive_matmul_4096_loop, F32, id="naive_matmul_4096_loop"),
        pytest.param(ayla.loop_4x4_matmul_4096, F32, id="loop_4x4_matmul_4096"),
        pytest.param(ayla.loop_8x8_matmul_4096, F32, id="loop_8x8_matmul_4096"),
        pytest.param(ayla.tma_matmul_4096, F32, id="tma_matmul_4096"),
        pytest.param(ayla.pranjalssh_4096, BF16, id="pranjalssh_4096"),
        pytest.param(ayla.pranjalssh_4096_pipelined, BF16, id="pranjalssh_4096_pipelined"),
        pytest.param(ayla.auto_matmul_4096, BF16, id="auto_matmul_4096"),
        pytest.param(ayla.auto_matmul_4096_wgmma256_wg, BF16, id="auto_matmul_4096_wgmma256_wg"),
        pytest.param(ayla.auto_matmul_4096_wgmma256_wg_storepipe, BF16, id="auto_matmul_4096_wgmma256_wg_storepipe"),
        pytest.param(ayla.fastcu_5_1, BF16, id="fastcu_5_1"),
    ],
)
def test_gemm_correctness(op, dtype, seed):
    torch.manual_seed(seed)
    a = torch.randn(4096, 4096, dtype=dtype, device="cuda")
    b = torch.randn(4096, 4096, dtype=dtype, device="cuda")

    c = op(a, b)

    assert c.dtype == dtype
    assert c.shape == (4096, 4096)

    ref = torch.mm(a.float(), b.float().t()).to(dtype)

    print(f"\n{op} c[0:64, 0:64]:")
    print(c[0:64, 0:64])
    print(f"\n{op} ref[0:64, 0:64]:")
    print(ref[0:64, 0:64])

    atol = 1e-3 if dtype == torch.float32 else 2.0
    assert torch.isfinite(c).all(), f"{op}: output contains NaN or Inf"
    assert torch.allclose(
        c, ref, atol=atol
    ), f"{op}: max error {(c - ref).abs().max().item()}"
    # Poison the output and inputs so the next test can't accidentally
    # reuse correct results from the CUDA allocator.
    c.fill_(float("nan"))
    a.fill_(float("nan"))
    b.fill_(float("nan"))
