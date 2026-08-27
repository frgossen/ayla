"""Correctness tests for all GEMM kernels in the collection."""

import ayla.gemm_collection  # noqa: F401  (registers torch.ayla.ayla.*)
import pytest
import torch
from torch.ops import ayla

from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
from ayla.gpu_dialect import TmaDescType
from ayla.ptx_codegen import make_ptx_kernel


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

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
        problem_m=512,
        problem_n=19968,
        problem_k=6656,
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
        variant_suffix="output_ef",
    ))
    assert kernel.name.endswith("_split2_dbuf2_output_ef")
    assert kernel.ptx.count("createpolicy.fractional.L2::evict_first") == 1
    assert kernel.ptx.count("bytes.L2::cache_hint") == 28
    assert kernel.ptx.count("bulk_group.L2::cache_hint") == 8


def test_four_tile_full_output_uses_one_store_per_output_tile():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=1,
        output_store_buffers=1,
        operand_move_attrs={
            "lhs": {"l2_cache_hint": "evict_first"},
            "dst": {"l2_cache_hint": "evict_first"},
        },
        variant_suffix="full_output_store",
    ))
    assert kernel.name.endswith("_4tiles_full_output_store")
    assert kernel.ptx.count("cp.async.bulk.tensor.3d.global.shared") == 4
    assert kernel.ptx.count("cp.async.bulk.wait_group") == 4
    assert kernel.ptx.count("stmatrix.") == 64


def test_four_tile_batches_startup_mbarrier_initialization():
    common = dict(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=1,
        output_store_buffers=1,
        operand_move_attrs={
            "lhs": {"l2_cache_hint": "evict_first"},
            "dst": {"l2_cache_hint": "evict_first"},
        },
    )
    baseline = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        **common, variant_suffix="full_output_store_test",
    ))
    batched = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        **common, batch_mbarrier_init=True,
        variant_suffix="full_output_batched_barriers_test",
    ))
    assert batched.ptx.count("mbarrier.init") == baseline.ptx.count("mbarrier.init")
    assert batched.ptx.count("fence.mbarrier_init") == 1
    assert batched.ptx.count("bar.sync 0;") == baseline.ptx.count("bar.sync 0;") - 3


def test_four_tile_reorders_startup_tma_stages():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=1,
        output_store_buffers=1,
        batch_mbarrier_init=True,
        startup_tma_stage_order=(0, 2, 1),
        variant_suffix="full_output_stage021_test",
    ))
    startup_loads = [
        kernel.ptx.index(f"mm_lhs_smem_iter{stage}_async_iter0")
        for stage in range(3)
    ]
    assert startup_loads[0] < startup_loads[2] < startup_loads[1]


def test_four_tile_uses_lane_zero_tma_guards():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=1,
        output_store_buffers=1,
        batch_mbarrier_init=True,
        startup_tma_stage_order=(0, 2, 1),
        lane_zero_tma_guard=True,
        variant_suffix="full_output_lane0_test",
    ))
    assert "%laneid" in kernel.ptx
    assert "setp.ge.u32" in kernel.ptx
    assert "%tid.y" not in kernel.ptx
    assert "%tid.z" not in kernel.ptx


def test_four_tile_compact_sass_reuses_scalar_setup():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=1,
        output_store_buffers=1,
        batch_mbarrier_init=True,
        startup_tma_stage_order=(0, 2, 1),
        lane_zero_tma_guard=True,
        compact_sass=True,
        variant_suffix="full_output_compact_sass_test",
    ))
    assert kernel.ptx.count("%laneid") == 1
    assert kernel.ptx.count("%tid.x") == 1
    assert "0x989680" not in kernel.ptx
    assert kernel.ptx.count("mbarrier.try_wait") == 28
    assert kernel.ptx.count("stmatrix.") == 64
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
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2_output_ef,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_output_ef",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_store,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_store",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_batched_barriers,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_batched_barriers",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_stage021,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_stage021",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_lane0,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_lane0",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_compact_sass,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_compact_sass",
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
