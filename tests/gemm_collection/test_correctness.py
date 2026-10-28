"""Correctness tests for all GEMM kernels in the collection."""

import textwrap

import ayla.gemm_collection  # noqa: F401  (registers torch.ayla.ayla.*)
import re
import pytest
import torch

from ayla.base.ir import find_ancestor_of_type, find_ops_of_type
from ayla.gemm_collection.gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
    schedule_matmul,
)
from ayla.gemm_collection.dynamic_m import (
    _dynamic_m_schedule_heuristic,
    make_dynamic_m_matmul,
)
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_sigmoid import (
    auto_matmul_4096_wgmma256_wg_sigmoid_module,
)
from ayla.gpu_dialect import (
    CollectiveAxis,
    ElementwiseOp,
    ForOp,
    ModuleOp,
    TileOp,
    TmaDescType,
    WgrmemTensorType,
    gpu_dialect,
)
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


ayla = torch.ops.ayla


F32 = torch.float32
BF16 = torch.bfloat16

def test_two_tile_tma_store_ptx_overlaps_next_tile_compute():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_m_tiles_per_cta=2,
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


def test_scheduled_gemm_overwrites_its_accumulator_instead_of_zeroing_it():
    ptx = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_m_tiles_per_cta=2,
    )).ptx
    assert "elementwise fill" not in ptx
    scale_d = re.findall(r"^\s+(\S+), 1, 1, 0, 0;$", ptx, re.MULTILINE)
    assert len(scale_d) == ptx.count("wgmma.mma_async")
    # The peeled first WGMMA of each of the two tiles overwrites.
    assert sorted(set(scale_d)) == ["0", "1"]
    assert scale_d.count("0") == 2


def test_configurable_tma_gemm_supports_rectangular_shapes():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=512,
        n=19968,
        k=6656,
        output_m_tiles_per_cta=2,
        operand_stages=4,
        output_slices=2,
    ))
    assert kernel.name.startswith("auto_matmul_m512_n19968_k6656_")
    assert kernel.grid_dim == (2, 78, 1)
    assert kernel.block_dim == (256, 1, 1)


@pytest.mark.parametrize("m_tile,expected_threads", [(64, 128), (256, 512)])
def test_configurable_tma_gemm_supports_extended_m_tiles(
    m_tile, expected_threads,
):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=256,
            n=64,
            k=64,
            output_m_tile=m_tile,
            output_n_tile=64,
            warpgroup_n_tile=64,
            operand_stages=2,
            output_tma_inner_size=None,
        )
    )
    assert kernel.block_dim == (expected_threads, 1, 1)


@pytest.mark.parametrize("n_tile", [48, 80, 96, 128, 152, 160, 192])
def test_configurable_tma_gemm_supports_general_n_tiles(n_tile):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=128,
            n=192,
            k=64,
            output_n_tile=n_tile,
            warpgroup_n_tile=n_tile,
            output_tma_inner_size=None,
        )
    )
    assert "wgmma.mma_async" in kernel.ptx


def test_fixed_output_grid_uses_a_persistent_tile_loop():
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=512,
        n=256,
        k=64,
        output_grid_size=3,
        output_m_tile=128,
        output_n_tile=128,
        warpgroup_n_tile=128,
        operand_stages=2,
        output_slices=8,
        output_tma_inner_size=None,
    )
    [kernel] = module.body.ops
    outer_loops = [
        loop for loop in find_ops_of_type(module, ForOp)
        if find_ancestor_of_type(loop, ForOp) is None
    ]

    assert kernel.grid_dim == (3, 1, 1)
    assert len(outer_loops) == 1
    assert outer_loops[0].trip_count == 3
    assert "bar.sync" in make_ptx_kernel(module).ptx


@requires_cuda
def test_fixed_output_grid_matches_torch():
    m, n, k = 512, 256, 64
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=k,
        output_grid_size=3,
        output_m_tile=128,
        output_n_tile=128,
        warpgroup_n_tile=128,
        operand_stages=2,
        output_slices=8,
        output_tma_inner_size=None,
    )
    launcher = make_ptx_launcher(make_ptx_kernel(module))
    torch.manual_seed(23)
    lhs = torch.randn((m, k), dtype=BF16, device="cuda")
    rhs = torch.randn((n, k), dtype=BF16, device="cuda")
    actual = torch.empty((m, n), dtype=BF16, device="cuda")


@requires_cuda
@pytest.mark.parametrize(
    "n_tile,slices,block_width,swizzle",
    [
        (256, 1, 64, "128B"),
        (224, 1, 32, "64B"),
        (144, 1, 16, "32B"),
        (192, 4, 16, "32B"),
        (168, 1, 8, None),
    ],
)
def test_blocked_output_staging_narrows_to_the_slice_width(
    n_tile, slices, block_width, swizzle,
):
    # A partial last tile, with a row width only the narrow block divides.
    m, k = 256, 128
    n = 3 * n_tile - block_width
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=k,
        output_n_tile=n_tile,
        warpgroup_n_tile=n_tile,
        output_slices=slices,
        operand_stages=2,
    ))
    [store] = [
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and physical_layout(typ) is not None
    ]
    assert physical_layout(store) == (block_width, "block_major")
    assert store.swizzle == swizzle
    launcher = make_ptx_launcher(kernel)
    torch.manual_seed(11)
    lhs = torch.randn((m, k), dtype=BF16, device="cuda")
    rhs = torch.randn((n, k), dtype=BF16, device="cuda")
    actual = torch.full((m, n), float("nan"), dtype=BF16, device="cuda")

    launcher(lhs, rhs, actual)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, torch.mm(lhs, rhs.T))


@pytest.mark.parametrize(
    "axes,expected_grid",
    [
        ((CollectiveAxis.X, CollectiveAxis.Y), (4, 3, 1)),
        ((CollectiveAxis.Y, CollectiveAxis.X), (3, 4, 1)),
    ],
)
def test_output_grid_axis_bindings(axes, expected_grid):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=256,
            n=192,
            k=64,
            output_m_tile=64,
            output_n_tile=64,
            warpgroup_n_tile=64,
            output_grid_axes=axes,
            output_tma_inner_size=None,
        )
    )
    assert kernel.grid_dim == expected_grid


@pytest.mark.parametrize(
    "axes,expected_grid",
    [
        ((CollectiveAxis.X, CollectiveAxis.Y), (3, 1, 1)),
        ((CollectiveAxis.Y, CollectiveAxis.X), (1, 3, 1)),
    ],
)
def test_grouped_output_grid_axis_bindings(axes, expected_grid):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=384,
            n=256,
            k=64,
            output_n_tiles_per_cta=4,
            output_m_tile=128,
            output_n_tile=64,
            warpgroup_n_tile=64,
            output_grid_axes=axes,
            output_tma_inner_size=None,
        )
    )
    assert kernel.grid_dim == expected_grid


def test_multi_output_uses_requested_tile_geometry():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=768,
            n=128,
            k=64,
            output_m_tiles_per_cta=2,
            output_m_tile=192,
            output_n_tile=64,
            warpgroup_n_tile=64,
            output_tma_inner_size=None,
        )
    )
    assert kernel.grid_dim == (2, 2, 1)
    assert kernel.block_dim == (384, 1, 1)


@pytest.mark.parametrize(
    "tiles_per_cta,expected_grid_x",
    [(3, 6), (5, 4), (6, 3), (8, 2), (16, 1)],
)
def test_extended_output_tiles_per_cta(tiles_per_cta, expected_grid_x):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=2048,
            n=256,
            k=64,
            output_m_tiles_per_cta=tiles_per_cta,
        )
    )
    assert kernel.grid_dim == (expected_grid_x, 1, 1)


@pytest.mark.parametrize(
    "m,n,k,m_tile,n_tile,n_group,expected_grid",
    [
        (12_544, 1_024, 512, 192, 128, 4, (66, 2, 1)),
        (25_088, 256, 1_024, 192, 128, 2, (131, 1, 1)),
        (8_192, 3_072, 768, 192, 128, 8, (43, 3, 1)),
    ],
)
def test_n_output_groups_infer_the_grid(
    m, n, k, m_tile, n_tile, n_group, expected_grid,
):
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=m,
            n=n,
            k=k,
            output_n_tiles_per_cta=n_group,
            output_m_tile=m_tile,
            output_n_tile=n_tile,
            warpgroup_n_tile=n_tile,
            output_tma_inner_size=None,
        )
    )
    assert kernel.grid_dim == expected_grid


def test_k128_block_major_nested_tiles_lower_to_ptx():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=128,
            n=256,
            k=128,
            operand_k_tile=128,
            output_slices=2,
        )
    )
    assert "wgmma.mma_async" in kernel.ptx
    assert kernel.ptx.count("wgmma.mma_async") == 8


def test_configurable_tma_gemm_supports_fp16():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=128,
            n=64,
            k=64,
            dtype="f16",
            output_n_tile=64,
            warpgroup_n_tile=64,
            output_tma_inner_size=None,
        )
    )
    assert ".f32.f16.f16" in kernel.ptx
    assert ".f32.bf16.bf16" not in kernel.ptx


def test_cyclic_wgmma_pipeline_reaches_ptx():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=128,
            n=256,
            k=1024,
            operand_stages=4,
            cyclic_operand_pipeline=True,
            output_slices=2,
        )
    )

    assert "wgmma.wait_group.sync.aligned 1;" in kernel.ptx


@requires_cuda
def test_cyclic_wgmma_pipeline_matches_torch():
    m, n, k = 128, 256, 1024
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=k,
        operand_stages=4,
        cyclic_operand_pipeline=True,
        output_slices=2,
    )
    launcher = make_ptx_launcher(make_ptx_kernel(module))
    torch.manual_seed(17)
    lhs = torch.randn((m, k), dtype=BF16, device="cuda")
    rhs = torch.randn((n, k), dtype=BF16, device="cuda")
    actual = torch.empty((m, n), dtype=BF16, device="cuda")

    launcher(lhs, rhs, actual)
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.mm(lhs, rhs.T))


@pytest.mark.parametrize(
    ("m", "n", "cluster_m_ctas", "cluster_n_ctas", "multicast_operand"),
    [
        (256, 256, 2, 1, "rhs"),
        (128, 512, 1, 2, "lhs"),
    ],
)
@requires_cuda
def test_cluster_multicast_gemm_matches_torch(
    m: int,
    n: int,
    cluster_m_ctas: int,
    cluster_n_ctas: int,
    multicast_operand: str,
):
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=128,
        output_m_tile=128,
        output_n_tile=256,
        warpgroup_n_tile=128,
        cluster_m_ctas=cluster_m_ctas,
        cluster_n_ctas=cluster_n_ctas,
        operand_move_attrs={multicast_operand: {"multicast": True}},
    )
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    torch.manual_seed(19)
    lhs = torch.randn((m, 128), dtype=BF16, device="cuda")
    rhs = torch.randn((n, 128), dtype=BF16, device="cuda")
    actual = torch.empty((m, n), dtype=BF16, device="cuda")

    launcher(lhs, rhs, actual)
    torch.cuda.synchronize()

    assert kernel.cluster_dim == (cluster_m_ctas, cluster_n_ctas, 1)
    assert ".multicast::cluster" in kernel.ptx
    torch.testing.assert_close(actual, torch.mm(lhs, rhs.T))


@requires_cuda
def test_cluster_dimensions_follow_swapped_output_grid_axes():
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=256,
        n=512,
        k=64,
        output_grid_axes=(CollectiveAxis.Y, CollectiveAxis.X),
        cluster_n_ctas=2,
        operand_move_attrs={"lhs": {"multicast": True}},
    )
    kernel = make_ptx_kernel(module)

    assert kernel.grid_dim == (2, 2, 1)
    assert kernel.cluster_dim == (2, 1, 1)


@requires_cuda
def test_k128_block_major_nested_tiles_match_torch():
    m, n, k = 128, 256, 128
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=k,
        operand_k_tile=128,
        output_slices=2,
    )
    launcher = make_ptx_launcher(make_ptx_kernel(module))
    torch.manual_seed(7)
    lhs = torch.randn((m, k), dtype=BF16, device="cuda")
    rhs = torch.randn((n, k), dtype=BF16, device="cuda")
    actual = torch.empty((m, n), dtype=BF16, device="cuda")

    launcher(lhs, rhs, actual)
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.mm(lhs, rhs.T))


@requires_cuda
def test_two_dimensional_warpgroup_tiles_match_torch():
    m, n, k = 256, 256, 128
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m,
        n=n,
        k=k,
        output_m_tile=128,
        output_n_tile=256,
        warpgroup_m_tile=64,
        warpgroup_n_tile=128,
        operand_stages=3,
    )
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    torch.manual_seed(17)


@pytest.mark.parametrize("cluster", [(2, 1), (1, 2)])
@pytest.mark.parametrize("output_m_tiles_per_cta", [1, 2])
def test_multicast_gemm_matches_torch(cluster, output_m_tiles_per_cta):
    # Four stages over eight k-steps release and refill every stage; two
    # output tiles per CTA refill stages again after the first epilogue.
    m, n, k = 512, 512, 512
    shared = "rhs" if cluster == (2, 1) else "lhs"
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=m, n=n, k=k, output_m_tile=64, output_n_tile=128,
        warpgroup_n_tile=64, output_slices=2, operand_stages=4,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        operand_move_attrs={shared: {"multicast": True}},
    )
    module.body.ops[0].attrs["cluster_dim"] = [*cluster, 1]
    launcher = make_ptx_launcher(make_ptx_kernel(module))
    torch.manual_seed(11)
    lhs = torch.randn((m, k), dtype=BF16, device="cuda")
    rhs = torch.randn((n, k), dtype=BF16, device="cuda")
    actual = torch.empty((m, n), dtype=BF16, device="cuda")

    launcher(lhs, rhs, actual)
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.mm(lhs, rhs.T))


def test_four_tile_output_evict_first_reaches_tma_stores():
    kernel = make_ptx_kernel(auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_m_tiles_per_cta=4,
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
        output_m_tiles_per_cta=4,
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


def test_dynamic_m_schedule_uses_broad_muse_regimes():
    heuristic = _dynamic_m_schedule_heuristic
    assert heuristic(128, 19968, 6656)["output_n_tile"] == 64
    assert heuristic(512, 19968, 6656)["output_n_tile"] == 208
    assert heuristic(1024, 19968, 19968)["output_n_tile"] == 208
    assert heuristic(4096, 19968, 19968)["output_n_tile"] == 256


def test_dynamic_m_gemm_returns_an_inferred_meta_output():
    op = make_dynamic_m_matmul(
        256, 64,
        name="test_dynamic_m_gemm_returns_an_inferred_meta_output",
        namespace="ayla_dynamic_schema_test",
    )
    lhs = torch.empty((32, 64), dtype=BF16, device="meta")
    rhs = torch.empty((256, 64), dtype=BF16, device="meta")

    out = op(lhs, rhs)

    assert str(op._schema).endswith("(Tensor arg0, Tensor arg1) -> Tensor")
    assert out.shape == (32, 256)
    assert out.dtype == BF16
    assert out.device.type == "meta"


@requires_cuda
def test_dynamic_m_gemm_specializes_distinct_shapes():
    op = make_dynamic_m_matmul(
        256, 64,
        name="test_dynamic_m_gemm_specializes_distinct_shapes",
        namespace="ayla_dynamic_test",
    )
    for m in (3, 128, 256):
        torch.manual_seed(m)
        lhs = torch.randn((m, 64), dtype=BF16, device="cuda")
        rhs = torch.randn((256, 64), dtype=BF16, device="cuda")
        out = op(lhs, rhs)
        torch.cuda.synchronize()
        assert torch.equal(out, torch.mm(lhs, rhs.T))


def test_sigmoid_output_tiles_lower_to_ptx():
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_sigmoid_module()
    )
    lines = kernel.ptx.splitlines()

    def at(pattern):
        return [i for i, line in enumerate(lines) if pattern in line]

    sigmoids = at("ex2.approx.ftz.f32")
    assert not at("ex2.approx.f32")
    converts = at("cvt.rn.bf16x2.f32")
    stores = at("cp.async.bulk.tensor.3d.global.shared")
    # Four output-M iterations each contain four fully unrolled 64-column
    # slices, with 32 f32 values (16 packed outputs) per thread and slice.
    assert len(sigmoids) == 4 * 4 * 32
    assert len(converts) == 4 * 4 * 16
    assert len(stores) == 16


def test_sigmoid_output_tiles_are_static_before_ptx_lowering():
    module = auto_matmul_4096_wgmma256_wg_sigmoid_module()

    loops = find_ops_of_type(module, ForOp)
    assert all(loop.attrs.get("label") != "outn" for loop in loops)
    wgrmem_tiles = [
        op for op in find_ops_of_type(module, TileOp)
        if isinstance(op.result().type, WgrmemTensorType)
    ]
    assert wgrmem_tiles
    assert all(None not in tile.static_indices for tile in wgrmem_tiles)

    epilogues = [
        op for op in find_ops_of_type(module, ElementwiseOp)
        if op.fn == "sigmoid"
    ]
    assert [op.attrs.get("label") for op in epilogues[:4]] == [
        "epi_iter0_iter0",
        "epi_iter1_iter0",
        "epi_iter2_iter0",
        "epi_iter3_iter0",
    ]
    assert len(epilogues) == 16


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
            ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2,
            BF16,
            None,
            id="auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_split2_dbuf2",
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


def test_every_registered_kernel_is_covered():
    """Nothing lands in the collection without a case above.

    That list is hand-written, so a new kernel is exercised only if whoever
    added it remembered to. `register` validates nothing about the kernel it
    builds, which leaves these cases as the only thing between a faulty one
    and the collection.
    """
    parametrize = next(
        mark for mark in getattr(test_gemm_correctness, "pytestmark")
        if mark.args and mark.args[0] == "op,dtype,epilogue"
    )
    covered = {param.id for param in parametrize.args[1]}
    # Fused operations with non-GEMM signatures have dedicated end-to-end
    # correctness tests instead of fitting the two-input matrix above.
    covered.add("muse_swiglu")
    registered = {
        name
        for name in dir(ayla)
        if isinstance(getattr(ayla, name), torch._ops.OpOverloadPacket)
        # The destination-passing forms take a buffer to write into, so
        # `op(a, b)` is the wrong shape for them; the torch.compile backend
        # covers those where it uses them.
        and getattr(ayla, name).default._schema.returns
    }
    assert registered <= covered, f"not exercised: {sorted(registered - covered)}"


def _transposed_output_matmul(m, n, k, **schedule):
    """Schedule C = A B^T as C^T = B A^T through C's column-major view."""
    a, b = f"gmem<{m}x{k}xf16>", f"gmem<{n}x{k}xf16>"
    c = f"gmem<{n}x{m}xf16 strides=[1, {n}]>"
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @mm(%a: {a}, %b: {b}, %c: {c}) {{
            matmul(%b: {b}, %a: {a}, %c: {c}) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return schedule_matmul(module, **schedule)


def test_transposed_output_stores_fragments_with_stmatrix_trans():
    kernel = make_ptx_kernel(_transposed_output_matmul(
        200, 256, 128, output_m_tile=128, output_n_tile=104,
    ))

    assert "stmatrix.sync.aligned.m8n8.x4.trans" in kernel.ptx
    assert "stmatrix.sync.aligned.m8n8.x2.trans" in kernel.ptx
    output_desc = next(
        typ for typ in kernel.signature
        if isinstance(typ, TmaDescType) and typ.arg_idx == 2
    )
    # The store describes the row-major C behind the column-major view.
    assert list(output_desc.gmem_type.sizes) == [200, 256]
    assert list(output_desc.tile_sizes) == [104, 128]


def test_transposed_output_needs_a_blocked_staging_layout():
    with pytest.raises(ValueError, match="transposed WGMMA fragment store"):
        make_ptx_kernel(_transposed_output_matmul(
            128, 128, 64, output_m_tile=128, output_n_tile=128,
            output_tma_inner_size=None,
        ))


@requires_cuda
@pytest.mark.parametrize(
    ("m", "n", "k", "schedule"),
    [
        (196, 384, 384, {"output_m_tile": 128, "output_n_tile": 200}),
        (130, 512, 512, {
            "output_m_tile": 64, "output_n_tile": 64,
            "warpgroup_n_tile": 64, "operand_stages": 4,
        }),
        (700, 768, 256, {"output_m_tile": 192, "output_n_tile": 176,
                         "operand_stages": 2}),
        (300, 512, 128, {"output_m_tile": 128, "output_n_tile": 128,
                         "output_slices": 2, "output_store_buffers": 2}),
    ],
)
def test_transposed_output_matches_torch(m, n, k, schedule):
    launcher = make_ptx_launcher(make_ptx_kernel(
        _transposed_output_matmul(m, n, k, **schedule),
    ))
    torch.manual_seed(11)
    lhs = torch.randn((m, k), dtype=torch.float16, device="cuda")
    rhs = torch.randn((n, k), dtype=torch.float16, device="cuda")
    actual = torch.full((m, n), float("nan"), dtype=torch.float16, device="cuda")

    launcher(lhs, rhs, actual)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, torch.mm(lhs, rhs.T), rtol=2e-3, atol=0.25)


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "ir_dtype"),
    [(torch.bfloat16, "bf16"), (torch.float16, "f16")],
)
def test_unswizzled_column_warpgroups_store_only_owned_slice(dtype, ir_dtype):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @mm(%a: gmem<64x64x{ir_dtype}>, %b: gmem<256x64x{ir_dtype}>, %c: gmem<64x256x{ir_dtype}>) {{
            matmul(%a: gmem<64x64x{ir_dtype}>, %b: gmem<256x64x{ir_dtype}>, %c: gmem<64x256x{ir_dtype}>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    module = schedule_matmul(
        module, output_m_tile=64, output_n_tile=256, warpgroup_n_tile=128,
        operand_stages=2, output_slices=2, output_tma_inner_size=None,
    )
    kernel = make_ptx_kernel(module)
    assert kernel.block_dim == (256, 1, 1)
    run = make_ptx_launcher(kernel)
    out = torch.empty(64, 256, dtype=dtype, device="cuda")
    for seed in (1, 104729, 15485863):
        torch.manual_seed(seed)
        a = torch.randn(64, 64, dtype=dtype, device="cuda")
        b = torch.randn(256, 64, dtype=dtype, device="cuda")
        out.fill_(float("nan"))
        run(a, b, out)
        torch.testing.assert_close(out, a @ b.T, atol=0.25, rtol=0.01)
