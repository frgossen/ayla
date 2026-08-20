"""Correctness tests for all GEMM kernels in the collection."""

import re

import ayla.gemm_collection  # noqa: F401  (registers torch.ayla.ayla.*)
import pytest
import torch
from torch.ops import ayla

from ayla.base.ir import find_ops
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
    best_auto_matmul_16384_wgmma256_wg_tma_store_module,
    best_auto_matmul_2048_wgmma256_wg_tma_store_module,
    best_auto_matmul_4096_wgmma256_wg_tma_store_module,
    best_auto_matmul_laguna_down_wgmma256_wg_tma_store_module,
    best_auto_matmul_laguna_up_wgmma256_wg_tma_store_module,
    best_auto_matmul_muse_down_wgmma256_wg_tma_store_module,
    best_auto_matmul_muse_intermediate_wgmma256_wg_tma_store_module,
    best_auto_matmul_muse_up_wgmma256_wg_tma_store_module,
)
from ayla.gpu_dialect import MoveOp, TmaDescType
from ayla.ptx_codegen import make_ptx_kernel


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

F32 = torch.float32
BF16 = torch.bfloat16


def _bf16_op(name: str):
    return pytest.param(getattr(ayla, name), BF16, id=name)


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
    output_desc = next(
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx == 2
    )
    assert sum(isinstance(typ, TmaDescType) for typ in kernel.signature) == 3
    assert output_desc.physical_layout == {
        "kind": "blocked_padding", "block": 128, "padding": 8,
    }
    assert len(stores) == len(waits) == 2
    store_templates = [
        section for section in kernel.ptx.split("// PTX template")
        if "tma_move" in section
        and "cp.async.bulk.tensor.3d.global.shared" in section
    ]
    assert store_templates
    assert all("div.u32" not in section and "rem.u32" not in section
               for section in store_templates)
    assert "%tid.y" not in kernel.ptx and "%tid.z" not in kernel.ptx
    assert "%ntid.x" not in kernel.ptx
    assert "bar.sync 1, 256" in kernel.ptx
    gather_templates = [
        section for section in kernel.ptx.split("// PTX template")
        if "stmatrix." in section
    ]
    assert gather_templates
    assert all("rem.u32" not in section for section in gather_templates)
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
    assert len(gathers) == 32
    assert waits[0] < gathers[len(gathers) // 2] < stores[1] < waits[1]


def test_pre_stsm_variant_uses_scalar_shared_stores():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2, output_store="scalar",
    ))
    assert kernel.name.endswith("_2tiles_pre_stsm")
    assert "st.shared.b32" in kernel.ptx
    assert "stmatrix." not in kernel.ptx
    assert "cp.async.bulk.tensor.2d.global.shared" in kernel.ptx


def test_four_output_tiles_reuse_the_existing_pipeline_resources():
    kernel = make_ptx_kernel(
        best_auto_matmul_4096_wgmma256_wg_tma_store_module()
    )
    assert kernel.name.endswith("_k3_4tiles_full_output_store")
    assert kernel.grid_dim == (8, 16, 1)
    assert kernel.block_dim == (256, 1, 1)
    assert kernel.ptx.count("cp.async.bulk.tensor.3d.global.shared") == 4
    assert kernel.ptx.count("cp.async.bulk.wait_group") == 4
    assert kernel.ptx.count("stmatrix.") == 64
    # The lane/warp-to-shared-memory offset is invariant across all eight
    # STMatrix groups and should be calculated once, then added to each arena.
    # Every lane consumer shares one kernel-scoped special-register read.
    assert kernel.ptx.count("%laneid") == 1
    assert kernel.ptx.count("createpolicy.fractional.L2::evict_first") == 1
    assert kernel.ptx.count("bytes.L2::cache_hint") == 28
    assert kernel.ptx.count("bulk_group.L2::cache_hint") == 4
    assert kernel.ptx.count("elect.sync") == 0
    assert kernel.ptx.count(".reqntid 256, 1, 1") == 1
    assert "shfl.sync.idx" not in kernel.ptx
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }
    assert input_descs[0].l2_promotion == "256B"
    assert input_descs[1].l2_promotion is None
    startup_loads = [
        kernel.ptx.index(f"mm_lhs_smem_iter{stage}_async_iter0")
        for stage in range(3)
    ]
    assert startup_loads[0] < startup_loads[2] < startup_loads[1]
    # Every completed output-arena store that is later reused must cross a
    # full CTA barrier before the next cooperative STMatrix refill. The first
    # three waits have a later refill; the final store is never reused.
    lines = kernel.ptx.splitlines()
    waits = [i for i, line in enumerate(lines)
             if "cp.async.bulk.wait_group" in line]
    gathers = [i for i, line in enumerate(lines) if "stmatrix." in line]
    for wait in waits:
        next_gather = next((i for i in gathers if i > wait), None)
        if next_gather is not None:
            assert any(line.strip() == "bar.sync 0;"
                       for line in lines[wait:next_gather])


def test_retained_16384_kernel_uses_wide_grouped_ctas_and_ab_promotion():
    kernel = make_ptx_kernel(
        best_auto_matmul_16384_wgmma256_wg_tma_store_module()
    )
    assert kernel.name.endswith(
        "_16384_wgmma256_wg_tma_store_k3_m256_n192_wg192_split3_"
        "ctagroupm8_3d_best"
    )
    assert kernel.grid_dim == (8, 86, 8)
    assert kernel.block_dim == (512, 1, 1)
    assert ".reqntid 512, 1, 1" in kernel.ptx
    input_descs = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    }
    assert input_descs[0].l2_promotion == "256B"
    assert input_descs[1].l2_promotion == "256B"
    assert kernel.ptx.count("createpolicy.fractional.L2::evict_first") == 1


def test_retained_2048_kernel_uses_cache_balanced_tile_and_rhs_policy():
    kernel = make_ptx_kernel(
        best_auto_matmul_2048_wgmma256_wg_tma_store_module()
    )
    assert kernel.name.endswith(
        "_2048_wgmma256_wg_tma_store_k4_split2_ctagroupm4_3d_best"
    )
    assert kernel.grid_dim == (4, 8, 4)
    assert kernel.block_dim == (256, 1, 1)
    assert ".reqntid 256, 1, 1" in kernel.ptx
    rhs_desc = next(
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx == 1
    )
    assert rhs_desc.l2_promotion == "256B"
    assert kernel.ptx.count("createpolicy.fractional.L2::evict_last") == 1


def test_retained_muse_intermediate_uses_grouped_192_square_tile():
    kernel = make_ptx_kernel(
        best_auto_matmul_muse_intermediate_wgmma256_wg_tma_store_module()
    )
    assert kernel.name.endswith(
        "_k3_m192_n192_wg192_split3_ctagroupm13_3d_"
        "muse_intermediate_best"
    )
    assert kernel.grid_dim == (13, 104, 8)
    assert kernel.block_dim == (384, 1, 1)
    assert ".reqntid 384, 1, 1" in kernel.ptx


def test_lhs_descriptor_precompute_is_an_opt_in_ptx_variant():
    baseline = make_ptx_kernel(
        best_auto_matmul_4096_wgmma256_wg_tma_store_module(
            precompute_wgmma_lhs_descriptors=False,
        )
    ).ptx
    precomputed = make_ptx_kernel(
        best_auto_matmul_4096_wgmma256_wg_tma_store_module(
            precompute_wgmma_lhs_descriptors=True,
        )
    ).ptx

    def additions_around_first_mma(ptx: str) -> tuple[int, int]:
        first_fence = ptx.index("wgmma.fence.sync.aligned")
        first_mma = ptx.index("wgmma.mma_async", first_fence)
        second_mma = ptx.index("wgmma.mma_async", first_mma + 1)
        return (
            ptx[first_fence:first_mma].count("add.u64"),
            ptx[first_mma:second_mma].count("add.u64"),
        )

    assert additions_around_first_mma(baseline) == (0, 2)
    assert additions_around_first_mma(precomputed) == (3, 1)


def test_rhs_descriptor_precompute_is_an_opt_in_ptx_variant():
    precomputed = make_ptx_kernel(
        best_auto_matmul_4096_wgmma256_wg_tma_store_module(
            precompute_wgmma_lhs_descriptors=False,
            precompute_wgmma_rhs_descriptors=True,
        )
    ).ptx
    first_fence = precomputed.index("wgmma.fence.sync.aligned")
    first_mma = precomputed.index("wgmma.mma_async", first_fence)
    second_mma = precomputed.index("wgmma.mma_async", first_mma + 1)
    assert precomputed[first_fence:first_mma].count("add.u64") == 3
    assert precomputed[first_mma:second_mma].count("add.u64") == 1

    both = make_ptx_kernel(
        best_auto_matmul_4096_wgmma256_wg_tma_store_module(
            precompute_wgmma_lhs_descriptors=True,
            precompute_wgmma_rhs_descriptors=True,
        )
    ).ptx
    first_fence = both.index("wgmma.fence.sync.aligned")
    first_mma = both.index("wgmma.mma_async", first_fence)
    second_mma = both.index("wgmma.mma_async", first_mma + 1)
    assert both[first_fence:first_mma].count("add.u64") == 6
    assert both[first_mma:second_mma].count("add.u64") == 0


def test_regrouped_stmatrix_rows_use_dense_block_major_swizzled_output():
    kernel = make_ptx_kernel(
        best_auto_matmul_4096_wgmma256_wg_tma_store_module()
    )
    output_descs = [
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx == 2
    ]
    assert len(output_descs) == 1
    assert output_descs[0].swizzle == "128B"
    assert output_descs[0].physical_layout == {
        "kind": "blocked_padding", "block": 64, "padding": 0,
        "order": "block_major", "outer_size": 128,
    }
    assert kernel.ptx.count("stmatrix.") == 64
    # The lane swizzle is computed once and all eight 32-column bases are
    # derived statically; regrouping does not add stores.
    assert kernel.ptx.count(", 896;") == 1
    assert kernel.ptx.count("bar.sync") == 33


def test_small_wgmma_output_tiles_are_supported():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            problem_size=512,
            output_m_tile=64,
            output_n_tile=64,
            operand_stages=2,
            output_slices=1,
            output_tma_inner_padding=0,
            regroup_stmatrix_block_major=True,
        )
    )
    assert kernel.grid_dim == (8, 8, 1)
    assert kernel.block_dim == (128, 1, 1)
    assert kernel.ptx.count("wgmma.mma_async") == 16
    assert kernel.ptx.count("stmatrix.") == 4


def test_rectangular_problem_dimensions_are_preserved():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            problem_m=3072,
            problem_n=1024,
            problem_k=3072,
            output_m_tile=256,
            output_n_tile=128,
            operand_stages=2,
            output_slices=2,
            output_tma_inner_padding=0,
            regroup_stmatrix_block_major=True,
        )
    )
    assert kernel.name.startswith("auto_matmul_m3072_n1024_k3072_")
    assert kernel.grid_dim == (12, 8, 1)
    descriptors = {
        typ.arg_idx: typ for typ in kernel.signature
        if isinstance(typ, TmaDescType)
    }
    assert descriptors[0].gmem_type.sizes == [3072, 3072]
    assert descriptors[1].gmem_type.sizes == [1024, 3072]
    assert descriptors[2].gmem_type.sizes == [3072, 1024]


def test_retained_model_shape_helpers_preserve_launch_geometry():
    expected = (
        (best_auto_matmul_laguna_up_wgmma256_wg_tma_store_module,
         (16, 8, 1), (384, 1, 1)),
        (best_auto_matmul_laguna_down_wgmma256_wg_tma_store_module,
         (8, 16, 1), (256, 1, 1)),
        (best_auto_matmul_muse_up_wgmma256_wg_tma_store_module,
         (52, 78, 1), (256, 1, 1)),
        (best_auto_matmul_muse_intermediate_wgmma256_wg_tma_store_module,
         (13, 104, 8), (384, 1, 1)),
        (best_auto_matmul_muse_down_wgmma256_wg_tma_store_module,
         (13, 35, 6), (512, 1, 1)),
    )
    for build, grid, block in expected:
        kernel = make_ptx_kernel(build())
        assert kernel.grid_dim == grid
        assert kernel.block_dim == block


def test_four_output_tiles_support_n_major_cta_raster():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4, operand_stages=3, output_slices=2,
        output_tma_inner_padding=16, output_store_buffers=2,
        cta_raster="n",
    ))
    assert kernel.name.endswith("_k3_4tiles_split2_dbuf2_raster_n")
    assert kernel.grid_dim == (16, 8, 1)
    assert kernel.ptx.count("wgmma.mma_async") == 112


def test_tma_l2_promotion_reaches_input_descriptors():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_move_attrs={
            "lhs": {"l2_promotion": "256B"},
            "rhs": {"l2_promotion": "256B"},
        },
    ))
    input_descs = [
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    ]
    output_descs = [
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx == 2
    ]
    assert input_descs and output_descs
    assert all(desc.l2_promotion == "256B" for desc in input_descs)
    assert all(desc.l2_promotion is None for desc in output_descs)


def test_tma_cache_controls_can_vary_per_static_load():
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4, promote_k_tail=False,
        rhs_output_l2_promotions=("256B", "128B", "256B", "128B"),
        k_tail_cache_hints={"rhs": "evict_first"},
        rhs_output_cache_hints=(
            "evict_last", "evict_last", "evict_last", "evict_first",
        ),
        operand_move_attrs={
            "lhs": {"l2_promotion": "256B"},
            "rhs": {"l2_promotion": "256B"},
        },
    )
    moves = find_ops(module, lambda op: isinstance(op, MoveOp))
    tails = [op for op in moves if "_smem_tail" in op.attrs.get("label", "")]
    rhs = [op for op in moves if "_rhs_smem_" in op.attrs.get("label", "")]
    assert tails and rhs
    assert all("l2_promotion" not in op.attrs for op in tails)
    assert all(op.attrs.get("l2_cache_hint") == "evict_first"
               for op in tails if "_rhs_smem_" in op.attrs.get("label", ""))
    assert all(op.attrs.get("l2_promotion") == "256B"
               for op in moves if re.search(
                   r"_lhs_smem_", op.attrs.get("label", ""))
               and op not in tails)
    assert all(op.attrs.get("l2_promotion") == (
                   "256B" if op.attrs.get("label", "").endswith(("_iter0", "_iter2"))
                   else "128B"
               ) for op in rhs if op not in tails)
    assert all(op.attrs.get("l2_cache_hint") == "evict_first"
               for op in rhs if op.attrs.get("label", "").endswith("_iter3"))
    assert all(op.attrs.get("l2_cache_hint") == "evict_last"
               for op in rhs if op not in tails
               and not op.attrs.get("label", "").endswith("_iter3"))


def test_k3_tail_reuses_stage_zero_mbarrier():
    common = dict(
        output_tiles_per_cta=4, operand_stages=3, output_slices=2,
        output_tma_inner_padding=16, output_store_buffers=2,
    )
    baseline = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(**common)
    )
    reused = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        **common, reuse_k_tail_mbarrier=True,
    ))
    assert reused.ptx.count("mbarrier.init") == baseline.ptx.count("mbarrier.init") - 1
    # All startup mbarriers share one publication fence/rendezvous, so deleting
    # the tail lock now removes only its init; the batch barrier is still needed
    # for the three operand-stage barriers.
    assert reused.ptx.count("fence.mbarrier_init") == 1
    assert reused.ptx.count("bar.sync 0;") == 7
    assert reused.ptx.count("bar.sync") == baseline.ptx.count("bar.sync")
    assert reused.ptx.count("mbarrier.try_wait") == baseline.ptx.count("mbarrier.try_wait")
    assert reused.ptx.count("cp.async.bulk.tensor") == baseline.ptx.count(
        "cp.async.bulk.tensor"
    )


def test_four_output_tiles_support_2x2_grouped_schedule():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4, output_tile_group=(2, 2),
        operand_stages=3, output_slices=2,
        output_tma_inner_padding=16, output_store_buffers=2,
    ))
    assert kernel.name.endswith("_k3_4tiles_group2x2_split2_dbuf2")
    assert kernel.grid_dim == (16, 8, 1)
    assert kernel.ptx.count("wgmma.mma_async") == 112
    assert kernel.ptx.count("cp.async.bulk.tensor.3d.global.shared") == 8


def test_four_output_tiles_support_2x2_m_outer_schedule():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4, output_tile_group=(2, 2),
        output_group_outer="m", operand_stages=3, output_slices=2,
        output_tma_inner_padding=16, output_store_buffers=2,
    ))
    assert kernel.name.endswith("_k3_4tiles_group2x2_mouter_split2_dbuf2")
    assert kernel.grid_dim == (16, 8, 1)
    assert kernel.ptx.count("wgmma.mma_async") == 112


def test_k4_split2_uses_four_operand_stages_and_half_output_arena():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2, operand_stages=4, output_slices=2,
    ))
    assert kernel.name.endswith("_k4_2tiles_split2")
    assert kernel.ptx.count("[16384]") == 4
    assert kernel.ptx.count("[32768]") == 4
    assert kernel.ptx.count("[34816]") == 1
    assert kernel.ptx.count("cp.async.bulk.tensor.3d.global.shared") == 4
    assert kernel.ptx.count("stmatrix.") == 32
    assert kernel.ptx.count("cvt.rn.bf16x2.f32") == 128
    assert "cvt.rn.bf16.f32" not in kernel.ptx


def test_k128_operand_tile_uses_block_major_3d_tma():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2, operand_stages=2, operand_k_tile=128,
        output_slices=2,
    ))
    assert kernel.name.endswith("_k2_ktile128_2tiles_split2")
    input_descs = [
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx in (0, 1)
    ]
    assert len(input_descs) == 2
    assert all(desc.tile_sizes[-1] == 128 for desc in input_descs)
    assert all(desc.physical_layout["order"] == "block_major"
               for desc in input_descs)
    assert kernel.ptx.count("tensor.3d.shared::cluster.global") == 16
    assert "tensor.2d.shared::cluster.global" not in kernel.ptx
    assert kernel.ptx.count("[32768]") == 2
    assert kernel.ptx.count("[65536]") == 2


def test_k128_startup_tma_stage_order_ignores_pipelined_label_reuse():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        problem_size=1024,
        output_m_tile=128,
        output_n_tile=64,
        operand_stages=4,
        operand_k_tile=128,
        output_tma_inner_padding=0,
        regroup_stmatrix_block_major=True,
        startup_tma_stage_order=(0, 1, 3, 2),
    ))
    startup_loads = [
        kernel.ptx.index(f'label: "mm_lhs_smem_iter{stage}_async"')
        for stage in range(4)
    ]
    assert (
        startup_loads[0] < startup_loads[1]
        < startup_loads[3] < startup_loads[2]
    )


def test_k3_keeps_warpgroup_descriptor_arithmetic_uniform():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=2, operand_stages=3, output_slices=2,
        output_tma_inner_padding=16,
    ))
    shared_tid = re.search(r"mov\.u32 (%r\d+), %tid\.x;", kernel.ptx)
    assert shared_tid is not None
    tid = re.escape(shared_tid.group(1))
    warpgroup_id = re.search(
        rf"mov\.u32 (%r\d+), {tid};\n"
        r"\s+shfl\.sync\.idx\.b32 \1, \1, 0, 0x1f, 0xffffffff;\n"
        r"\s+shr\.u32 \1, \1, 7;",
        kernel.ptx,
    )
    derived_descriptors = re.findall(
        r"add\.u64 (%rd\d+), (%rd\d+), ([246]);", kernel.ptx,
    )
    assert warpgroup_id is not None
    # 14 wgmma groups (12 pipelined + one `k` unroll tail per output tile),
    # each advancing its lhs/rhs base descriptors three times.  Updating in
    # place avoids six one-use 64-bit descriptor names per group.
    assert len(derived_descriptors) == 84
    assert all(dst == src and increment == "2"
               for dst, src, increment in derived_descriptors)
    # The tail copy left by the `k` unroll must be unrolled and fused like the
    # pipelined copies: one fence/commit/wait_group per group of four wgmmas,
    # never a rolled `kk` loop draining after every single wgmma.
    assert kernel.ptx.count("wgmma.mma_async") == 56
    assert kernel.ptx.count("wgmma.wait_group") == 14
    assert kernel.ptx.count("wgmma.fence") == 14


def test_n384_split3_uses_three_warpgroups_and_sliced_tma():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_n_tile=384, operand_stages=2, output_slices=3,
    ))
    assert kernel.name.endswith("_k2_n384_split3")
    assert kernel.grid_dim == (32, 11, 1)
    assert kernel.block_dim == (384, 1, 1)
    assert "setp.eq.u32" in kernel.ptx
    assert kernel.ptx.count("[34816]") == 1
    rhs_descs = [
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx == 1
    ]
    assert rhs_descs
    assert all(typ.tile_sizes == (128, 64) for typ in rhs_descs)


def test_m192_split2_uses_three_warpgroups_and_partial_m_tile():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_m_tile=192, operand_stages=3, output_slices=2,
    ))
    assert kernel.name.endswith("_k3_m192_split2")
    assert kernel.grid_dim == (22, 16, 1)
    assert kernel.block_dim == (384, 1, 1)
    assert kernel.ptx.count("[24576]") == 3
    assert kernel.ptx.count("[32768]") == 3
    assert kernel.ptx.count("[52224]") == 1


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
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_pre_stsm,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_pre_stsm",
        ),
        pytest.param(
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles,
            BF16,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles",
        ),
        *(_bf16_op(name) for name in (
            "auto_matmul_4096_wgmma256_wg_tma_store_k2_n384_split3",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_m192_split2",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_full_output_store",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2_promote_256b",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2_raster_n",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_group2x2_split2_dbuf2",
            "auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_group2x2_mouter_split2_dbuf2",
            "auto_matmul_4096_wgmma256_wg_tma_store_k2_ktile128_2tiles_split2",
            "auto_matmul_4096_wgmma256_wg_tma_store_k4_2tiles_split2",
        )),
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
