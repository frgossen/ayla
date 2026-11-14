"""TMEM lifecycle and Blackwell GEMM scheduling regressions."""

import math
import re

import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.base.pipeline import Pipeline
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gemm_collection.gemm import schedule_matmul
from ayla.gemm_jit import _module
from ayla.gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    CollectiveAxis,
    CollectiveLevel,
    DeallocOp,
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    MatmulOp,
    MoveOp,
    TmemTensorType,
    gpu_dialect,
    is_side_effect_free,
    memory_effects,
    root_buffer,
)
from ayla.passes.promote_buffers import PromoteBuffers
from ayla.passes.licm import LICM
from ayla.passes.schedule_async import ScheduleAsync
from ayla.passes.tile_matmul import TileMatmul
from ayla.pipelines.canonicalize import Canonicalize
from ayla.pipelines.tile_matmul_pipeline import schedule_register_matmul
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.test_blackwell_paired import CONFIG as PAIRED_CONFIG
from tests.test_mma_queue_depth import CLUSTER as QUEUE_CLUSTER, PAIRED as QUEUE_PAIRED
from tests.testing_utils import DTYPES, LARGE_K, LOOSE, check_gemm

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)


def _tmem_gemm(dst: str = "gmem<128x64xbf16>") -> str:
    return f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @gemm(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<64x16xbf16 swizzle="128B">, %c: {dst}) block_dim=[128, 1, 1] {{
            %zero: rmem<f32> = constant 0.0
            matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<64x16xbf16 swizzle="128B">, %c: {dst}, %zero: rmem<f32>) collective=xyz.block
          }}
        }}
    """


def _tmem_loads(module):
    return [
        op
        for op in find_ops_of_type(module, MoveOp)
        if isinstance(root_buffer(op.src).type, TmemTensorType)
    ]


def test_tmem_promotion_and_lifetime():
    module = gpu_dialect.parse_module(_tmem_gemm())
    promoted = PromoteBuffers("tmem", "f32", operands=["dst"]).run(module)
    [alloc] = [
        op
        for op in find_ops_of_type(promoted, AllocOp)
        if isinstance(op.result().type, TmemTensorType)
    ]
    [fill] = [op for op in find_ops_of_type(promoted, ElementwiseOp) if op.fn == "fill"]
    [mm] = find_ops_of_type(promoted, MatmulOp)
    [free] = find_ops_of_type(promoted, DeallocOp)
    assert alloc.result().type.columns == 64
    assert fill.dst is mm.dst is free.value is alloc.result()
    assert mm.init is None
    # The accumulator drains through bounded register fragments, which the
    # ordinary register epilogue converts and stores.
    loads = _tmem_loads(promoted)
    assert [load.dst.type.layout for load in loads] == ["tcgen05_16x32bx2"] * 2
    assert all(list(load.dst.type.sizes) == [128, 32] for load in loads)
    stores = [
        op
        for op in find_ops_of_type(promoted, MoveOp)
        if isinstance(op.dst.type, GmemTensorType)
    ]
    assert [store.src.type.dtype.name for store in stores] == ["bf16"] * 2
    assert not is_side_effect_free(alloc)
    assert memory_effects(free)[0].roots == {alloc.result()}
    assert repr(gpu_dialect.parse_module(repr(promoted))) == repr(promoted)


@pytest.mark.parametrize(
    "dst,layout",
    [
        ("gmem<128x64xbf16>", "tcgen05_16x32bx2"),
        ("gmem<128x64xf32>", "tcgen05_32x32b"),
        # Transposed STMatrix stores M-major 16-bit output from m16n8 blocks.
        ("gmem<128x64xbf16 strides=[1, 128]>", "tcgen05_16x256b"),
        ("gmem<128x64xf32 strides=[1, 128]>", "tcgen05_32x32b"),
    ],
)
def test_tmem_fragment_layout_follows_the_output(dst, layout):
    module = gpu_dialect.parse_module(_tmem_gemm(dst))
    promoted = PromoteBuffers("tmem", "f32", operands=["dst"]).run(module)
    assert {load.dst.type.layout for load in _tmem_loads(promoted)} == {layout}


def test_tmem_fragment_layout_override():
    module = gpu_dialect.parse_module(_tmem_gemm())
    promoted = PromoteBuffers(
        "tmem", "f32", operands=["dst"], layout="tcgen05_16x256b"
    ).run(module)
    assert {load.dst.type.layout for load in _tmem_loads(promoted)} == {
        "tcgen05_16x256b"
    }


def test_tmem_reuse_moves_lifetime_and_retains_initialization():
    module = blackwell_tcgen05_gemm_module(
        m=256,
        n=264,
        k=80,
        dtype="f16",
        output_n_tile=128,
        operand_k_tile=64,
        operand_stages=2,
        cyclic_operand_pipeline=True,
        output_grid_size=2,
        allow_partial_tiles=True,
        reuse_tmem_allocation=True,
    )
    [alloc] = [
        op
        for op in find_ops_of_type(module, AllocOp)
        if isinstance(op.result().type, TmemTensorType)
    ]
    [release] = find_ops_of_type(module, DeallocOp)
    [kernel] = module.body.ops
    assert alloc.parent_block is release.parent_block is kernel.body
    boundaries = [
        op
        for op in find_ops_of_type(module, BarrierOp)
        if op.attrs.get("resource_lifetime_reuse")
    ]
    assert len(boundaries) == 1 and boundaries[0].parent_block is not kernel.body
    assert all(
        op.parent_block is not kernel.body for op in find_ops_of_type(module, MatmulOp)
    )
    assert make_ptx_kernel(module).ptx.count("tcgen05.alloc") == 1


def test_tmem_reuse_keeps_multiple_resource_lifetimes_separate():
    module = gpu_dialect.parse_module("""
      module attrs={target: "sm_100a"} {
        kernel @k() block_dim=[128, 1, 1] {
          for (%i: index = 0 to 3) {
            %a: tmem<128x256xf32> = alloc() attrs={id: 1}
            dealloc(%a: tmem<128x256xf32>)
            %b: tmem<128x256xf32> = alloc() attrs={id: 2}
            dealloc(%b: tmem<128x256xf32>)
          }
        }
      }
    """)
    before = repr(module)
    assert repr(LICM(reuse_tmem_allocation=True).run(module)) == before


@pytest.mark.parametrize(
    "n,columns", [(8, 32), (24, 32), (64, 64), (136, 256), (256, 256)]
)
def test_tmem_allocation_granularity(n, columns):
    typ = TmemTensorType.build([128, n], F32Type())
    assert typ.columns == columns


@pytest.mark.parametrize(
    "sizes,columns", [([32, 64], 64), ([128, 65], 64), ([128, 64], 48)]
)
def test_invalid_tmem_geometry(sizes, columns):
    with pytest.raises(AssertionError):
        TmemTensorType.build(sizes, F32Type(), columns=columns)


def test_m64_tmem_promotion_preserves_register_budget():
    module = gpu_dialect.parse_module(_tmem_gemm().replace("128x", "64x"))
    promoted = PromoteBuffers("tmem", "f32", operands=["dst"]).run(module)
    [load] = _tmem_loads(promoted)
    assert list(load.dst.type.sizes) == [64, 64]
    assert load.dst.type.num_elements_per_thread == 32
    assert load.dst.type.warps == 4
    ptx = make_ptx_kernel(
        blackwell_tcgen05_gemm_module(
            m=64,
            n=64,
            k=64,
            output_m_tile=64,
            output_n_tile=64,
            operand_stages=1,
        )
    ).ptx
    assert "tcgen05.ld.sync.aligned.16x32bx2" in ptx


@requires_blackwell
def test_m64_nonzero_initialization_uses_the_mma_datapaths():
    from ayla.pipelines.tile_matmul_pipeline import _schedule_matmul

    module = gpu_dialect.parse_module("""
      module attrs={target: "sm_100a"} {
        kernel @k(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<64x64xf32>) {
          %init: rmem<f32> = constant 7.0
          matmul(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<64x64xf32>, %init: rmem<f32>) collective=xyz.grid attrs={label: "mm"}
        }
      }
    """)
    module = _schedule_matmul(
        module,
        output_m_tile=64,
        output_n_tile=64,
        operand_stages=1,
        operand_k_tile=64,
        output_store="global",
        output_tma_inner_size=None,
    )
    check_gemm(make_ptx_launcher(make_ptx_kernel(module)), 64, 64, 64, out_dtype="f32",
               bias=7, atol=2e-5, rtol=1e-6, runs=1)


def test_blackwell_uses_existing_async_pipeline():
    module = blackwell_tcgen05_gemm_module(problem_size=512)
    groups = [
        op for op in find_ops_of_type(module, AsyncOp) if op.lock.type.kind == "tcgen05"
    ]
    assert groups and all(
        any(wait.lock is group.lock for wait in find_ops_of_type(module, AwaitOp))
        for group in groups
    )
    assert any(len(group.body.ops) >= 4 for group in groups)
    kernel = make_ptx_kernel(module)
    ptx = kernel.ptx
    assert ".target sm_100a" in ptx
    assert "tcgen05.mma.cta_group::1.kind::f16" in ptx
    assert "tcgen05.commit.cta_group::1" in ptx
    assert "mbarrier.try_wait.parity.acquire.cta" in ptx
    assert "cp.async.bulk.tensor" in ptx
    assert "wgmma" not in ptx
    assert (
        ptx.index("tcgen05.alloc")
        < ptx.index("tcgen05.mma")
        < ptx.index("tcgen05.ld")
        < ptx.index("tcgen05.dealloc")
    )
    assert kernel.block_dim == (128, 1, 1)


@pytest.mark.parametrize("stages,probed", [(4, False), (6, True)])
def test_deep_operand_rings_probe_waits_one_mma_batch_ahead(stages, probed):
    ptx = make_ptx_kernel(
        blackwell_tcgen05_gemm_module(
            m=256, n=256, k=1024, output_n_tile=128, operand_stages=stages,
            cyclic_operand_pipeline=True,
        )
    ).ptx
    probe = "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
    assert (probe in ptx) == probed
    if probed:
        # The probe is issued before the MMA batch whose completion it overlaps, and a ready
        # probe branches past the blocking wait loop.
        assert ptx.index(probe) < ptx.index("tcgen05.mma")
        pred = re.search(re.escape(probe) + r" (%p\d+),", ptx).group(1)
        skip = re.search(rf"@{re.escape(pred)} bra (\S+);", ptx)
        wait = ptx.index("mbarrier.try_wait", skip.end())
        assert ptx.index(f"{skip.group(1)}:", skip.end()) > wait


@pytest.mark.parametrize(
    "config",
    [
        {"m": 129},
        {"n": 255},
        {"k": 127},
        {"output_n_tile": 0},
        {"operand_k_tile": 0},
        {"operand_k_tile": 128, "operand_stages": 3},
        {"operand_stages": 0},
        {"dtype": "f32"},
    ],
)
def test_invalid_blackwell_schedule(config):
    with pytest.raises(ValueError):
        blackwell_tcgen05_gemm_module(problem_size=512, **config)


@requires_blackwell
def test_tmem_fill_and_sequential_allocation():
    module = gpu_dialect.parse_module("""
        module attrs={target: "sm_100a"} {
          kernel @fill(%c: gmem<128x64xf32>, %d: gmem<128x64xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %value: rmem<f32> = constant 3.25
            %first: tmem<128x64xf32> = alloc() attrs={id: 100}
            elementwise fill(%first: tmem<128x64xf32>, %value: rmem<f32>)
            %rows: wgrmem<128x64xf32 warps=4 layout="tcgen05_32x32b"> = alloc() attrs={id: 102}
            move(%rows: wgrmem<128x64xf32 warps=4 layout="tcgen05_32x32b">, %first: tmem<128x64xf32>) collective=xyz.block
            move(%d: gmem<128x64xf32>, %rows: wgrmem<128x64xf32 warps=4 layout="tcgen05_32x32b">) collective=xyz.block
            dealloc(%first: tmem<128x64xf32>)
            %second: tmem<128x64xf32> = alloc() attrs={id: 101}
            elementwise fill(%second: tmem<128x64xf32>, %value: rmem<f32>)
            %blocks: wgrmem<128x64xf32 warps=4 layout="tcgen05_16x256b"> = alloc() attrs={id: 103}
            move(%blocks: wgrmem<128x64xf32 warps=4 layout="tcgen05_16x256b">, %second: tmem<128x64xf32>) collective=xyz.block
            move(%c: gmem<128x64xf32>, %blocks: wgrmem<128x64xf32 warps=4 layout="tcgen05_16x256b">) collective=xyz.block
            dealloc(%second: tmem<128x64xf32>)
          }
        }
    """)
    kernel = make_ptx_kernel(module)
    assert kernel.ptx.count("tcgen05.relinquish_alloc_permit") == 1
    assert kernel.ptx.rindex("tcgen05.dealloc") < kernel.ptx.index(
        "tcgen05.relinquish_alloc_permit"
    )
    launch = make_ptx_launcher(kernel)
    c = torch.full((128, 64), float("nan"), device="cuda")
    d = torch.empty_like(c)
    launch(c, d)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.full_like(c, 3.25), rtol=0, atol=0)
    torch.testing.assert_close(d, torch.full_like(d, 3.25), rtol=0, atol=0)


def test_grid_stride_tmem_writeback_retains_tiling_scheme():
    module = blackwell_tcgen05_gemm_module(problem_size=512, output_grid_size=7)
    stores = [
        op
        for op in find_ops_of_type(module, MoveOp)
        if isinstance(op.dst.type, GmemTensorType)
    ]
    assert stores and all(
        op.attrs.get("dst_tiling_scheme") is not None for op in stores
    )
    assert make_ptx_kernel(module).grid_dim == (7, 1, 1)


def _sweep(name, rows, make, dtypes=DTYPES, **check):
    """`(module kwargs, check_gemm options)` params with ids `name-<row>-<dtype>`.

    `make(*row, dtype=...)` builds the kwargs; dict rows are passed whole and
    labelled `config<i>`.
    """
    params = []
    for i, row in enumerate(rows):
        args, label = (row, row) if isinstance(row, tuple) else ((row,), (f"config{i}",))
        params += [
            pytest.param(make(*args, dtype=d), check, id="-".join(map(str, (name, *label, d))))
            for d in dtypes
        ]
    return params


def _wide(nt, warps, budget, kt=64, store="stmatrix", slices=1, buffers=1, tail=0,
          axis="n", stages=None, *, dtype):
    stages = (2 if kt == 64 else 1) if stages is None else stages
    return dict(m=384, n=nt * 2 + tail, k=512, dtype=dtype, output_m_tile=128,
                output_n_tile=nt, operand_k_tile=kt, operand_stages=stages,
                cyclic_operand_pipeline=stages > 1, tmem_writeback_warps=warps,
                tmem_writeback_register_budget=budget, output_store=store,
                output_slices=slices, output_store_buffers=buffers, output_grid_size=3,
                output_grid_fast_axis=axis, allow_partial_tiles=bool(tail),
                reuse_tmem_allocation=True)


def _wide_cta(m, n, k, mt, nt, warps, stages, store, budget, slices, grid, kt, *, dtype):
    return dict(m=m, n=n, k=k, dtype=dtype, output_m_tile=mt, output_n_tile=nt,
                tmem_writeback_warps=warps, operand_stages=stages,
                cyclic_operand_pipeline=stages > 1, operand_k_tile=kt, output_store=store,
                output_slices=slices, output_store_buffers=slices,
                tmem_writeback_register_budget=budget, output_grid_size=grid,
                reuse_tmem_allocation=grid is not None, allow_partial_tiles=True)


def _eight_warp(m, n, k, nt, stages, store, layout, budget, grid, reuse, slices, axis, *, dtype):
    return dict(m=m, n=n, k=k, dtype=dtype, output_n_tile=nt, operand_stages=stages,
                cyclic_operand_pipeline=stages > 1, operand_k_tile=64,
                allow_partial_tiles=True, tmem_writeback_warps=8, output_store=store,
                output_layout=layout, tmem_writeback_register_budget=budget,
                output_grid_size=grid, reuse_tmem_allocation=reuse, output_slices=slices,
                output_store_buffers=slices, output_grid_fast_axis=axis)


def _m64_eight_warp(nt, layout, store, budget, slices, buffers, stages, kt, k, n, *, dtype):
    return dict(m=384, n=n, k=k, dtype=dtype, output_m_tile=64, output_n_tile=nt,
                operand_k_tile=kt, operand_stages=stages, cyclic_operand_pipeline=stages > 1,
                tmem_writeback_warps=8, tmem_writeback_register_budget=budget,
                output_store=store, output_layout=layout, output_slices=slices,
                output_store_buffers=buffers, output_grid_size=3, output_grid_fast_axis="n",
                reuse_tmem_allocation=True, allow_partial_tiles=True)


# Paired 1000x960 GEMMs on a (4, 1, 1) cluster with partial tiles on both axes.
_CLUSTER_1000 = dict(m=1000, n=960, k=1088, output_n_tile=64, operand_stages=4,
                     cyclic_operand_pipeline=True, output_store="stmatrix",
                     allow_partial_tiles=True, cta_group=2, cluster_n_ctas=2,
                     cluster_dim=(4, 1, 1))
_QUEUE_LAYOUTS = dict(single={}, pair=QUEUE_PAIRED, cluster=QUEUE_CLUSTER)
_EIGHT_WARP = [
    (384, 512, 128, 256, 2, "global", None, 32, 3, True, 1, "n"),
    (384, 520, 80, 256, 2, "global", None, 128, 3, True, 1, "m"),
    (384, 408, 320, 136, 3, "global", "tcgen05_16x256b", 32, 3, False, 1, "n"),
    (128, 72, 64, 24, 1, "global", None, 32, None, False, 1, "m"),
    (384, 768, 512, 256, 4, "stmatrix", None, 64, 3, True, 1, "n"),
    (384, 384, 320, 128, 3, "stmatrix", "tcgen05_16x256b", 32, 3, True, 2, "m"),
    (256, 256, 128, 256, 2, "stmatrix", "tcgen05_16x256b", 128, None, False, 1, "m"),
    (128, 8, 16, 8, 1, "global", None, 32, None, False, 1, "n"),
]
_WIDE_CTA = [
    (128, 512, 64, 128, 512, 4, 1, "global", 32, 1, None, 64),
    (128, 512, 128, 128, 512, 8, 2, "stmatrix", 64, 1, None, 64),
    (384, 520, 80, 128, 512, 4, 2, "global", 32, 1, 3, 64),
    (384, 520, 80, 128, 512, 8, 2, "global", 64, 1, 3, 64),
    (384, 768, 512, 128, 384, 4, 3, "stmatrix", 64, 2, 3, 64),
    (384, 768, 512, 128, 384, 8, 3, "global", 64, 1, 3, 64),
    (192, 512, 512, 64, 512, 4, 3, "stmatrix", 32, 1, 2, 64),
    (192, 384, 512, 64, 384, 4, 3, "global", 32, 1, 2, 64),
    (384, 1024, 512, 128, 512, 4, 1, "global", 64, 1, 3, 128),
    (384, 1024, 512, 128, 512, 8, 1, "global", 64, 1, 3, 128),
]

SWEEP = [
    *_sweep("square", [
        {"output_grid_size": 7},
        {"output_grid_size": 7, "output_grid_fast_axis": "n"},
        {"output_grid_fast_axis": "n"},
        {"operand_move_attrs": {"lhs": {"l2_cache_hint": "evict_first"},
                                "rhs": {"l2_cache_hint": "evict_last"}}},
        {"output_n_tile": 8, "operand_k_tile": 64, "operand_stages": 8},
        {"output_n_tile": 16, "operand_k_tile": 512, "operand_stages": 1},
    ], lambda c, dtype: dict(problem_size=1024, dtype=dtype, **c), dtypes=("bf16",), runs=3),
    # Partial tiles must not write outside the output, for either grid order.
    *_sweep("partial_tiles", [
        (reuse, axis, layout)
        for layout in (None, "tcgen05_32x32b") for axis in "mn" for reuse in (False, True)
    ], lambda reuse, axis, layout, dtype: dict(
        m=256, n=264, k=80, dtype=dtype, output_n_tile=128, operand_k_tile=64,
        operand_stages=2, cyclic_operand_pipeline=True, allow_partial_tiles=True,
        output_grid_size=2, output_grid_fast_axis=axis, output_layout=layout,
        reuse_tmem_allocation=reuse)),
    # M64 tiles use all four TMEM partitions.
    *_sweep("m64_partitions", [
        (8, 16, 8, 1, "global", 1),
        (72, 64, 24, 1, "global", 1),
        (408, 320, 136, 3, "global", 1),
        (768, 512, 256, 4, "global", 1),
        (264, 80, 128, 2, "global", 1),
        (768, 512, 256, 4, "stmatrix", 1),
        (384, 320, 128, 3, "stmatrix", 2),
    ], lambda n, k, nt, stages, store, slices, dtype: dict(
        m=192, n=n, k=k, dtype=dtype, output_m_tile=64, output_n_tile=nt,
        operand_stages=stages, operand_k_tile=64, cyclic_operand_pipeline=stages > 1,
        output_store=store, output_slices=slices, output_store_buffers=slices,
        output_grid_size=2, allow_partial_tiles=True, reuse_tmem_allocation=True)),
    *_sweep("wide_writeback", [
        (store, n, budget, mt)
        for mt in (64, 128) for budget in (64, 128)
        for store, n in (("global", 520), ("stmatrix", 512))
    ], lambda store, n, budget, mt, dtype: dict(
        m=384, n=n, k=80, dtype=dtype, output_m_tile=mt, output_n_tile=256,
        operand_k_tile=64, operand_stages=2, cyclic_operand_pipeline=True,
        output_store=store, output_grid_size=3, allow_partial_tiles=True,
        reuse_tmem_allocation=True, tmem_writeback_register_budget=budget),
        dtypes=("f16",)),
    # TMEM reuse orders the barrier reset between output tiles.
    *_sweep("barrier_reset", [(64,), (128,)], lambda mt, dtype: dict(
        m=384, n=72, k=64, dtype=dtype, output_m_tile=mt, output_n_tile=24,
        operand_k_tile=64, operand_stages=1, output_grid_size=2,
        reuse_tmem_allocation=True), dtypes=("f16",)),
    # Repeated launches must reset the accumulator and release its TMEM.
    *_sweep("pipeline", [
        (8, 64, 1, 64), (24, 64, 2, 320), (64, 64, 3, 512),
        (128, 128, 2, 640), (256, 128, 2, 512), (128, 64, 6, 1024),
    ], lambda nt, kt, stages, k, dtype: dict(
        m=256, n=nt * 3, k=k, output_n_tile=nt, operand_k_tile=kt, operand_stages=stages,
        dtype=dtype), dtypes=("bf16", "f16"), atol=0.125, rtol=0.01, runs=20),
    # Larger K batches use blocked TMA and queued MMA scheduling.
    *_sweep("large_k", [
        (128, 64, 512, 1, 1536), (128, 64, 512, 1, 1280), (64, 32, 1024, 1, 1280),
        (64, 64, 768, 1, 1536), (64, 64, 384, 2, 1280),
    ], lambda mt, nt, kt, stages, k, dtype: dict(
        m=384, n=256, k=k, dtype=dtype, output_m_tile=mt, output_n_tile=nt,
        operand_k_tile=kt, operand_stages=stages, cyclic_operand_pipeline=stages > 1,
        output_grid_size=3, reuse_tmem_allocation=True, allow_partial_tiles=k % kt != 0),
        seed=8139, **LARGE_K),
    # Wide TMEM geometries and single-buffer writeback.
    *_sweep("wide_geometry", [
        (320, 4, 64, 64), (320, 8, 64, 64), (448, 4, 64, 64), (448, 8, 64, 64),
        (384, 4, 128, 64), (384, 8, 128, 64), (512, 4, 128, 64), (512, 8, 128, 64),
        (384, 8, 128, 128), (512, 8, 128, 128),
    ], _wide, seed=8137, **LARGE_K),
    *_sweep("wide_staging", [
        (384, 8, 128, "stmatrix", 4, 4, 0, 2), (512, 8, 128, "stmatrix", 1, 1, 0, 2),
        (384, 8, 128, "global", 1, 1, 8, 3), (512, 4, 128, "stmatrix", 2, 2, 0, 1),
    ], lambda nt, warps, budget, store, slices, buffers, tail, stages, dtype: _wide(
        nt, warps, budget, 64, store, slices, buffers, tail, "m", stages, dtype=dtype),
        seed=8137, **LARGE_K),
    # CTA tiles assembled from MMA subtiles: exact while K fits one batch.
    *_sweep("wide_cta", _WIDE_CTA[:4], _wide_cta, seed=7281),
    *_sweep("wide_cta", _WIDE_CTA[4:], _wide_cta, seed=7281, **LARGE_K),
    *_sweep("eight_warp", [r for r in _EIGHT_WARP if r[2] <= 128], _eight_warp, seed=571),
    *_sweep("eight_warp", [r for r in _EIGHT_WARP if r[2] > 128], _eight_warp, seed=571,
            **LARGE_K),
    # Register budgets slice writeback through the declared staging buffers.
    *_sweep("register_budget", [(4,), (8,)], lambda warps, dtype: dict(
        m=384, n=768, k=512, dtype=dtype, output_n_tile=256, operand_stages=3,
        cyclic_operand_pipeline=True, output_store="stmatrix", output_slices=2,
        output_store_buffers=2, tmem_writeback_register_budget=64,
        tmem_writeback_warps=warps, output_layout="tcgen05_16x32bx2",
        output_grid_size=3, reuse_tmem_allocation=True), seed=972, **LARGE_K),
    # Eight warps sharing one M64 accumulator by columns.
    *_sweep("m64_eight_warp", [
        (16, None, "global", 32, 1, 1, 1, 64, 128, 8),
        (80, None, "global", 64, 1, 1, 3, 64, 1024, 168),
        (16, None, "global", 32, 1, 1, 1, 64, 128, 32),
        (48, None, "global", 32, 1, 1, 2, 64, 512, 104),
        (48, "tcgen05_16x256b", "stmatrix", 32, 1, 1, 2, 64, 512, 96),
        (80, None, "stmatrix", 64, 1, 1, 3, 64, 1024, 160),
        (144, "tcgen05_16x256b", "stmatrix", 64, 1, 1, 3, 64, 1024, 288),
        (192, "tcgen05_16x256b", "global", 32, 1, 1, 3, 64, 1024, 392),
        (192, None, "stmatrix", 64, 2, 2, 3, 64, 1024, 384),
        (256, None, "global", 64, 2, 1, 5, 64, 2048, 520),
        (384, None, "stmatrix", 128, 1, 1, 2, 64, 1024, 768),
        (512, None, "stmatrix", 128, 1, 1, 2, 64, 1024, 1024),
        (512, None, "stmatrix", 64, 2, 2, 2, 64, 1024, 1024),
        (512, "tcgen05_16x256b", "global", 128, 1, 1, 2, 64, 1024, 1032),
        (128, "tcgen05_16x256b", "global", 32, 1, 1, 1, 256, 1280, 264),
        (64, None, "global", 32, 1, 1, 10, 64, 2048, 128),
        (128, None, "stmatrix", 64, 2, 2, 8, 64, 4096, 256),
        (128, None, "global", 128, 1, 1, 2, 192, 1280, 512),
    ], _m64_eight_warp, seed=26495, **LOOSE),
    # Paired MMAs see changed operands on every launch.
    *_sweep("paired", [(512, 512, 384, 6), (384, 264, 80, 2), (256, 256, 784, 6)],
            lambda m, n, k, stages, dtype: dict(PAIRED_CONFIG, m=m, n=n, k=k, dtype=dtype,
                                                operand_stages=stages),
            mutate=True, runs=3, atol=4, rms=0.002),
    # Two queued MMA batches, for single, paired and clustered CTAs.
    *_sweep("queue_depth", [
        (m, n, k, layout) for layout in _QUEUE_LAYOUTS
        for m, n, k in ((1024, 1024, 4096), (1000, 960, 1088))
    ], lambda m, n, k, layout, dtype: dict(
        m=m, n=n, k=k, dtype=dtype, output_n_tile=64, operand_stages=8,
        cyclic_operand_pipeline=2, allow_partial_tiles=True, output_store="stmatrix",
        **_QUEUE_LAYOUTS[layout]), dtypes=("f16",), mutate=True, runs=2, atol=math.inf,
        rms=1e-3),
    # Interleaved and hoisted MMA descriptors keep partial cluster outputs.
    *_sweep("interleaved_descriptors", [(64,), (128,)], lambda mt, dtype: dict(
        _CLUSTER_1000, dtype=dtype, output_m_tile=mt, interleave_mma_descriptors=True,
        **(dict(output_layout="tcgen05_16x256b", tmem_writeback_register_budget=32)
           if mt == 64 else {})), mutate=True, runs=3, atol=math.inf, rms=0.01),
    *_sweep("hoisted_descriptors", [(64,), (128,)], lambda mt, dtype: dict(
        _CLUSTER_1000, dtype=dtype, output_m_tile=mt, hoist_mma_descriptors=True),
        graph=True, runs=3, atol=math.inf, rms=0.01),
]


@requires_blackwell
@pytest.mark.parametrize("config,check", SWEEP)
def test_blackwell_sweep_schedules(config, check):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(**config))
    assert kernel.block_dim == (32 * config.get("tmem_writeback_warps", 4), 1, 1)
    m, n, k = (config.get(dim, config.get("problem_size")) for dim in "mnk")
    check_gemm(make_ptx_launcher(kernel), m, n, k, config["dtype"], **check)


def test_measured_square_schedule_is_copied():
    from ayla.gemm_collection.blackwell_tcgen05 import select_blackwell_square_schedule

    schedule = select_blackwell_square_schedule(16384)
    schedule["operand_move_attrs"]["rhs"]["l2_cache_hint"] = "evict_first"
    assert (
        select_blackwell_square_schedule(16384)["operand_move_attrs"]["rhs"][
            "l2_cache_hint"
        ]
        == "evict_last"
    )
    with pytest.raises(ValueError, match="no measured B200"):
        select_blackwell_square_schedule(768)


@requires_blackwell
@pytest.mark.parametrize("size", [512, 1024, 2048, 4096, 8192, 16384])
def test_measured_square_heuristic_correctness(size):
    from ayla.gemm_collection.blackwell_tcgen05 import blackwell_square_gemm_module

    launch = make_ptx_launcher(
        make_ptx_kernel(blackwell_square_gemm_module(problem_size=size))
    )
    check_gemm(launch, size, size, size, runs=1, seed=4)


def _tmem_layout_gemm(m, n, k, n_tile, layout, tma):
    module = gpu_dialect.parse_module(f"""
        module attrs={{target: "sm_100a"}} {{
          kernel @gemm(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) block_dim=[128, 1, 1] {{
            %zero: rmem<f32> = constant 0.0
            matmul(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>, %zero: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
          }}
        }}
    """)
    pipeline = Pipeline()
    pipeline.add_pass(
        TileMatmul(
            op_dims=["M", "N"],
            tile_sizes=[128, n_tile],
            iteration_bindings=[
                (CollectiveLevel.BLOCK, CollectiveAxis.X),
                (CollectiveLevel.BLOCK, CollectiveAxis.Y),
            ],
            for_loop_labels=[None, None],
            op_label_filter="mm",
        )
    )
    pipeline.add_pass(
        PromoteBuffers(
            "tmem",
            "f32",
            operands=["dst"],
            layout=layout,
            tma=tma,
            physical_layout=(
                {"swizzle": "128B", "block_width": 64, "order": "block_major"}
                if tma
                else None
            ),
            label_filter="mm",
        )
    )
    pipeline.add_pass(
        schedule_register_matmul(label="mm", operand_stages=2, operand_k_tile=64)
    )
    pipeline.add_pass(ScheduleAsync())
    pipeline.add_pass(Canonicalize())
    return pipeline.run(module)


@requires_blackwell
@pytest.mark.parametrize("tma", [False, True], ids=["global", "tma"])
@pytest.mark.parametrize(
    "layout,load",
    [
        ("tcgen05_32x32b", "32x32b"),
        ("tcgen05_16x256b", "16x256b"),
        ("tcgen05_16x32bx2", "16x32bx2"),
    ],
)
@pytest.mark.parametrize("n,n_tile,k", [(768, 256, 512), (72, 24, 320)])
def test_tmem_fragment_layouts_store_exactly(layout, load, tma, n, n_tile, k):
    if (tma or layout == "tcgen05_16x32bx2") and n_tile % 16:
        pytest.skip("TMA staging uses 16-column blocks")
    m = 256
    kernel = make_ptx_kernel(_tmem_layout_gemm(m, n, k, n_tile, layout, tma))
    assert f"tcgen05.ld.sync.aligned.{load}" in kernel.ptx
    assert (".global.shared::cta.bulk_group" in kernel.ptx) == tma
    check_gemm(make_ptx_launcher(kernel), m, n, k, runs=1)


@requires_blackwell
@pytest.mark.parametrize("slices,buffers", [(1, 1), (8, 1), (8, 8)])
def test_blackwell_tma_store_output(slices, buffers):
    kernel = make_ptx_kernel(
        blackwell_tcgen05_gemm_module(
            problem_size=1024,
            output_store="stmatrix",
            output_slices=slices,
            output_store_buffers=buffers,
        )
    )
    # Row fragments reach the swizzled staging buffer as 16-byte vectors.
    assert "st.shared.v4.b32" in kernel.ptx
    check_gemm(make_ptx_launcher(kernel), 1024, 1024, 1024, runs=1)


@requires_blackwell
@pytest.mark.parametrize(
    "transpose,lhs_column_major,rhs_column_major,output_store,load",
    [
        (True, False, False, "stmatrix", "16x256b"),
        (False, True, False, "global", "16x32bx2"),
        (False, False, True, "stmatrix", "16x32bx2"),
    ],
    ids=["transposed_output", "mn_major_lhs", "mn_major_rhs"],
)
def test_blackwell_operand_and_output_orientations(
    transpose, lhs_column_major, rhs_column_major, output_store, load
):
    m, n, k = 256, 512, 256
    module = _module(
        "mm", "bfloat16", 1, m, n, k, transpose, lhs_column_major, rhs_column_major
    )
    module.attrs["target"] = "sm_100a"
    kernel = make_ptx_kernel(
        schedule_matmul(
            module,
            output_n_tile=128,
            operand_stages=2,
            output_store=output_store,
        )
    )
    assert f"tcgen05.ld.sync.aligned.{load}" in kernel.ptx
    launch = make_ptx_launcher(kernel)
    check_gemm(lambda a, b, c: launch(
        a.T.contiguous().T if lhs_column_major else a,
        b.T.contiguous().T if rhs_column_major else b,
        c.T if transpose else c,
    ), m, n, k, runs=1, seed=1)
