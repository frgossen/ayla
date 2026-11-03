"""TMEM lifecycle and Blackwell GEMM scheduling regressions."""

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
from ayla.passes.schedule_async import ScheduleAsync
from ayla.passes.tile_matmul import TileMatmul
from ayla.pipelines.canonicalize import Canonicalize
from ayla.pipelines.tile_matmul_pipeline import schedule_register_matmul
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

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


@pytest.mark.parametrize(
    "n,columns", [(8, 32), (24, 32), (64, 64), (136, 256), (256, 256)]
)
def test_tmem_allocation_granularity(n, columns):
    typ = TmemTensorType.build([128, n], F32Type())
    assert typ.columns == columns


@pytest.mark.parametrize(
    "sizes,columns", [([64, 64], 64), ([128, 65], 64), ([128, 64], 48)]
)
def test_invalid_tmem_geometry(sizes, columns):
    with pytest.raises(AssertionError):
        TmemTensorType.build(sizes, F32Type(), columns=columns)


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
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize(
    "n_tile,k_tile,stages,k",
    [
        (8, 64, 1, 64),
        (24, 64, 2, 320),
        (64, 64, 3, 512),
        (128, 128, 2, 640),
        (256, 128, 2, 512),
    ],
)
def test_blackwell_pipeline_correctness(dtype, n_tile, k_tile, stages, k):
    m, n = 256, n_tile * 3
    kernel = make_ptx_kernel(
        blackwell_tcgen05_gemm_module(
            m=m,
            n=n,
            k=k,
            output_n_tile=n_tile,
            operand_k_tile=k_tile,
            operand_stages=stages,
            dtype=dtype,
        )
    )
    launch = make_ptx_launcher(kernel)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    for seed in range(3):
        torch.manual_seed(seed)
        a = torch.randn(m, k, device="cuda", dtype=torch_dtype)
        b = torch.randn(n, k, device="cuda", dtype=torch_dtype)
        c = torch.full((m, n), float("nan"), device="cuda", dtype=torch_dtype)
        # Repeated launches must reset the accumulator and release its TMEM.
        for _ in range(20):
            launch(a, b, c)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a @ b.T, rtol=0.01, atol=0.125)


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


@requires_blackwell
@pytest.mark.parametrize(
    "config",
    [
        {"output_grid_size": 7},
        {"output_grid_size": 7, "output_grid_fast_axis": "n"},
        {"output_grid_fast_axis": "n"},
        {
            "operand_move_attrs": {
                "lhs": {"l2_cache_hint": "evict_first"},
                "rhs": {"l2_cache_hint": "evict_last"},
            }
        },
        {"output_n_tile": 8, "operand_k_tile": 64, "operand_stages": 8},
        {"output_n_tile": 16, "operand_k_tile": 512, "operand_stages": 1},
    ],
)
def test_blackwell_sweep_schedules(config):
    kernel = make_ptx_kernel(blackwell_tcgen05_gemm_module(problem_size=1024, **config))
    launch = make_ptx_launcher(kernel)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.full_like(a, float("nan"))
    for _ in range(3):
        launch(a, b, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b.T, rtol=0, atol=0)


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

    torch.manual_seed(4)
    launch = make_ptx_launcher(
        make_ptx_kernel(blackwell_square_gemm_module(problem_size=size))
    )
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.full_like(a, float("nan"))
    launch(a, b, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b.T, rtol=0, atol=0)


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
    "layout,load", [("tcgen05_32x32b", "32x32b"), ("tcgen05_16x256b", "16x256b"), ("tcgen05_16x32bx2", "16x32bx2")]
)
@pytest.mark.parametrize("n,n_tile,k", [(768, 256, 512), (72, 24, 320)])
def test_tmem_fragment_layouts_store_exactly(layout, load, tma, n, n_tile, k):
    if (tma or layout == "tcgen05_16x32bx2") and n_tile % 16:
        pytest.skip("TMA staging uses 16-column blocks")
    m = 256
    kernel = make_ptx_kernel(_tmem_layout_gemm(m, n, k, n_tile, layout, tma))
    assert f"tcgen05.ld.sync.aligned.{load}" in kernel.ptx
    assert (".global.shared::cta.bulk_group" in kernel.ptx) == tma
    launch = make_ptx_launcher(kernel)
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    launch(a, b, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b.T, rtol=0, atol=0)


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
    launch = make_ptx_launcher(kernel)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.full_like(a, float("nan"))
    launch(a, b, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b.T, rtol=0, atol=0)


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
    torch.manual_seed(1)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    launch(
        a.T.contiguous().T if lhs_column_major else a,
        b.T.contiguous().T if rhs_column_major else b,
        c.T if transpose else c,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a @ b.T, rtol=0, atol=0)
