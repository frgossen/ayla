"""Correctness tests for all GEMM kernels in the collection."""

import ayla.gemm_collection  # noqa: F401  (registers torch.ayla.ayla.*)
import pytest
import torch
from torch.ops import ayla

from ayla.gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
from ayla.gpu_dialect import TmaDescType
from ayla.ptx_codegen import make_ptx_kernel
from tests.testing_utils import requires_cuda


F32 = torch.float32
BF16 = torch.bfloat16


def test_two_tile_tma_store_ptx_overlaps_next_tile_compute():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2,
    ))
    lines = kernel.ptx.splitlines()
    stores = [i for i, line in enumerate(lines)
              if "cp.async.bulk.tensor.3d.global.shared" in line]
    waits = [i for i, line in enumerate(lines)
             if "cp.async.bulk.wait_group" in line]
    gathers = [i for i, line in enumerate(lines) if "stmatrix." in line]
    second_tile_mmas = [
        i for i, line in enumerate(lines)
        if "wgmma.mma_async" in line and stores[0] < i < waits[0]
    ]
    assert kernel.grid_dim == (16, 16, 1)
    descs = [typ for typ in kernel.signature if isinstance(typ, TmaDescType)]
    assert len(descs) == 3
    output_desc = next(typ for typ in descs if typ.arg_idx == 2)
    assert output_desc.swizzle == "128B"
    assert output_desc.attrs["physical_layout"] == {
        "block_width": 64,
        "order": "block_major",
    }
    first_fence = kernel.ptx.index("wgmma.fence.sync.aligned")
    first_mma = kernel.ptx.index("wgmma.mma_async", first_fence)
    descriptor_setup = kernel.ptx[first_fence:first_mma]
    assert descriptor_setup.count("shr.b32") == 2
    assert descriptor_setup.count("add.u64") == 6
    assert ".shared .align 1024 .b8" in kernel.ptx
    assert len(stores) == len(waits) == 2
    assert len(gathers) == 32
    assert second_tile_mmas
    assert all(
        any("bar.sync " in line and "bar.sync 0" not in line
            for line in lines[max(0, store - 20):store])
        for store in stores
    )
    assert not any(
        "bar.sync 0" in line
        for line in lines[stores[0]:second_tile_mmas[0]]
    )
    assert waits[0] < gathers[len(gathers) // 2] < stores[1] < waits[1]


def test_configurable_tma_gemm_supports_rectangular_shapes():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=512,
        n=19968,
        k=6656,
        output_tiles_per_cta=2,
        operand_stages=4,
        output_slices=2,
    ))
    assert kernel.name.startswith("auto_matmul_m512_n19968_k6656_")
    assert kernel.grid_dim == (2, 78, 1)
    assert kernel.block_dim == (256, 1, 1)


def test_four_tile_output_evict_first_reaches_tma_stores():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=2,
        output_store_buffers=2,
        operand_move_attrs={
            "lhs": {"l2_cache_hint": "evict_first"},
            "dst": {"l2_cache_hint": "evict_first"},
        },
    ))
    assert kernel.name.endswith("_4tiles_split2_dbuf2")
    assert kernel.ptx.count("createpolicy.fractional.L2::evict_first") == 1
    assert kernel.ptx.count("bytes.L2::cache_hint") == 28
    assert kernel.ptx.count("bulk_group.L2::cache_hint") == 8


def test_lhs_l2_promotion_reaches_tma_descriptor():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=2,
        output_store_buffers=2,
        operand_move_attrs={
            "lhs": {
                "l2_cache_hint": "evict_first",
                "l2_promotion": "256B",
            },
            "dst": {"l2_cache_hint": "evict_first"},
        },
    ))
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }
    assert input_descs[0].l2_promotion == "256B"
    assert input_descs[1].l2_promotion is None


@requires_cuda
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize(
    "op,dtype,epilogue",
    [
        pytest.param(lambda a, b: torch.mm(a, b.t()), F32, None, id="torch.mm"),
        pytest.param(ayla.naive_matmul_4096_loop, F32, None, id="naive_matmul_4096_loop"),
        pytest.param(ayla.loop_4x4_matmul_4096, F32, None, id="loop_4x4_matmul_4096"),
        pytest.param(ayla.loop_8x8_matmul_4096, F32, None, id="loop_8x8_matmul_4096"),
        pytest.param(ayla.tma_matmul_4096, F32, None, id="tma_matmul_4096"),
        pytest.param(ayla.pranjalssh_4096, BF16, None, id="pranjalssh_4096"),
        pytest.param(ayla.pranjalssh_4096_pipelined, BF16, None, id="pranjalssh_4096_pipelined"),
        pytest.param(ayla.auto_matmul_4096, BF16, None, id="auto_matmul_4096"),
        pytest.param(ayla.auto_matmul_4096_wgmma256_wg, BF16, None, id="auto_matmul_4096_wgmma256_wg"),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3,
            BF16,
            None,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles,
            BF16,
            None,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2,
            BF16,
            None,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2",
        ),
        pytest.param(ayla.fastcu_5_1, BF16, None, id="fastcu_5_1"),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_sigmoid,
            BF16,
            torch.sigmoid,
            id="auto_matmul_4096_wgmma256_wg_sigmoid",
        ),
    ],
)
def test_gemm_correctness(op, dtype, epilogue, seed):
    torch.manual_seed(seed)
    a = torch.randn(4096, 4096, dtype=dtype, device="cuda")
    b = torch.randn(4096, 4096, dtype=dtype, device="cuda")

    c = op(a, b)

    assert c.dtype == dtype
    assert c.shape == (4096, 4096)

    ref = torch.mm(a.float(), b.float().t())
    if epilogue is not None:
        ref = epilogue(ref)
    ref = ref.to(dtype)

    print(f"\n{op} c[0:64, 0:64]:")
    print(c[0:64, 0:64])
    print(f"\n{op} ref[0:64, 0:64]:")
    print(ref[0:64, 0:64])

    # The bf16 tolerance is sized for a raw 4096-deep accumulation. An
    # epilogue bounds the output -- sigmoid to [0, 1] -- so that tolerance
    # would accept anything; what is left is the epilogue's own error.
    if epilogue is not None:
        atol = 2e-2
    else:
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
