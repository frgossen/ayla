"""Correctness tests for all GEMM kernels in the collection."""

import ayla.gemm_collection  # noqa: F401  (registers torch.ayla.ayla.*)
import pytest
import torch
from torch.ops import ayla

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store_k3 import (
    auto_matmul_4096_wgmma256_wg_tma_store_k3_module,
)
from ayla.ptx_codegen import make_ptx_kernel


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

F32 = torch.float32
BF16 = torch.bfloat16


def test_two_tile_tma_store_ptx_overlaps_next_tile_compute():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
        output_tiles_per_cta=2,
    ))
    lines = kernel.ptx.splitlines()
    stores = [i for i, line in enumerate(lines)
              if "cp.async.bulk.tensor.2d.global.shared" in line]
    waits = [i for i, line in enumerate(lines)
             if "cp.async.bulk.wait_group 0" in line]
    gathers = [i for i, line in enumerate(lines) if "st.shared.b32" in line]
    second_tile_mmas = [
        i for i, line in enumerate(lines)
        if "wgmma.mma_async" in line and stores[0] < i < waits[0]
    ]
    assert kernel.grid_dim == (16, 16, 1)
    assert len(stores) == len(waits) == 2
    assert second_tile_mmas
    assert any(
        "bar.sync 0" in line
        for line in lines[stores[0]:second_tile_mmas[0]]
    )
    assert waits[0] < gathers[64] < stores[1] < waits[1]


def test_k2_one_arena_static_smem_layout():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
        input_stages=2, output_tiles_per_cta=2,
    ))
    assert kernel.ptx.count("[65536];") == 1
    assert kernel.ptx.count("[32768];") == 2
    assert kernel.ptx.count("[16384];") == 2


def test_k2_two_arena_ptx_layout_and_overlap():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
        input_stages=2, output_tiles_per_cta=2, output_buffers=2,
    ))
    lines = kernel.ptx.splitlines()
    stores = [i for i, line in enumerate(lines)
              if "cp.async.bulk.tensor.2d.global.shared" in line]
    waits = [i for i, line in enumerate(lines)
             if "cp.async.bulk.wait_group 0" in line]
    assert kernel.ptx.count("[65536];") == 2
    assert kernel.ptx.count("[32768];") == 2
    assert kernel.ptx.count("[16384];") == 2
    assert len(stores) == len(waits) == 2
    assert stores[1] < waits[0]
    assert any(
        "wgmma.mma_async" in line
        for line in lines[stores[0]:stores[1]]
    )


def test_k2_two_arena_grid_8x8_ptx_layout():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
        input_stages=2, output_tiles_per_cta=4,
        output_tiles_n_per_cta=2, output_buffers=2,
    ))
    assert kernel.grid_dim == (8, 8, 1)
    assert kernel.ptx.count("cp.async.bulk.tensor.2d.global.shared") == 8
    assert kernel.ptx.count("cp.async.bulk.wait_group 0") == 8
    assert kernel.ptx.count("[65536];") == 2


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
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k2_2tiles,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k2_2tiles",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k2_2tiles_2arenas,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k2_2tiles_2arenas",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k2_4x2tiles_2arenas,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k2_4x2tiles_2arenas",
        ),
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
