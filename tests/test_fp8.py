"""FP8 types, native instruction selection, scaling and GPU numerics."""

import itertools

import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.fp8 import scaled_mm
from ayla.gemm_collection.gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module as build,
)
from ayla.gpu_dialect import AllocOp, E4M3Type, E5M2Type, TmaDescType, gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from ayla.torch_custom_call import from_torch_dtype, to_torch_dtype
from tests.testing_utils import requires_cuda

FORMATS = [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]


@pytest.mark.parametrize(
    "typ,dtype", [(E4M3Type, torch.float8_e4m3fn), (E5M2Type, torch.float8_e5m2)]
)
def test_fp8_types(typ, dtype):
    assert typ.byte_size == 1
    assert to_torch_dtype(typ()) == dtype
    assert from_torch_dtype(dtype) == typ()
    module = gpu_dialect.parse(
        f"module {{ kernel @types(%a: gmem<64x128x{typ().name}>) {{}} }}"
    )
    assert module.body.ops[0].body.arguments[0].type.dtype == typ()


@pytest.mark.parametrize("da,db", list(itertools.product(FORMATS, repeat=2)))
@pytest.mark.parametrize("kt", [64, 128])
def test_fp8_instruction_and_transfer_layout(da, db, kt):
    p = make_ptx_kernel(
        build(m=128, n=256, k=256, dtype=da[0], rhs_dtype=db[0], operand_k_tile=kt)
    )
    assert f"m64n256k32.f32.{da[0]}.{db[0]}" in p.ptx
    assert "k16.f32" not in p.ptx
    desc = next(t for t in p.signature if isinstance(t, TmaDescType) and t.arg_idx == 0)
    assert desc.swizzle == ("64B" if kt == 64 else "128B")


@requires_cuda
@pytest.mark.parametrize("da,db", list(itertools.product(FORMATS, repeat=2)))
@pytest.mark.parametrize("kt", [64, 128])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp8_scaled_matches_quantized_reference(da, db, kt, output_dtype):
    # Partial M/N tiles and multiple K iterations exercise TMA zero filling.
    torch.manual_seed(17)
    m, n, k = 80, 152, 256
    a = torch.randn(m, k, device="cuda").to(da[1])
    b = torch.randn(n, k, device="cuda").to(db[1])
    sa = torch.tensor(0.5, device="cuda")
    sb = torch.tensor(0.125, device="cuda")
    out = torch.empty(m, n, device="cuda", dtype=output_dtype)
    output_name = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}[
        output_dtype
    ]
    module = build(
        m=m,
        n=n,
        k=k,
        dtype=da[0],
        rhs_dtype=db[0],
        scaled=True,
        output_dtype=output_name,
        operand_k_tile=kt,
    )
    launch = make_ptx_launcher(make_ptx_kernel(module))
    launch(a, b, out, sa, sb)
    torch.cuda.synchronize()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (a.float() @ b.float().T) * sa * sb
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert torch.isfinite(out).all()
    assert ((out.float() - reference).norm() / reference.norm()).item() < 0.005
    before = out.clone()
    # Scales remain device data and must be re-read on graph replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(a, b, out, sa, sb)
    sa.fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, before * 0.5, atol=0, rtol=0)


@requires_cuda
def test_public_scaled_mm_and_compile():
    a = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
    bt = torch.randn(128, 128, device="cuda").to(torch.float8_e5m2)
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.5, 0.25))
    eager = scaled_mm(a, bt.T, sa, sb)
    compiled = torch.compile(scaled_mm, fullgraph=True)
    torch.testing.assert_close(compiled(a, bt.T, sa, sb), eager)
    with pytest.raises(ValueError, match="column-major"):
        scaled_mm(a, bt, sa, sb)


@requires_cuda
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_public_output_types_long_k(output_dtype):
    a = torch.randn(64, 4096, device="cuda").to(torch.float8_e4m3fn)
    bt = torch.randn(128, 4096, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.tensor(0.7, device="cuda")
    sb = torch.tensor(0.3, device="cuda")
    out = scaled_mm(a, bt.T, sa, sb, out_dtype=output_dtype)
    assert out.dtype == output_dtype
    torch.cuda.synchronize()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (a.float() @ bt.float().T) * sa * sb
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert torch.isfinite(out).all()
    assert ((out.float() - reference).norm() / reference.norm()).item() < 0.005


@requires_cuda
@pytest.mark.parametrize(
    "options",
    [dict(output_slices=2), dict(cyclic_operand_pipeline=True), dict(cluster_n_ctas=2)],
)
def test_fp8_scaling_with_schedules(options):
    a = torch.randn(128, 512, device="cuda").to(torch.float8_e4m3fn)
    bt = torch.randn(512, 512, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.tensor(0.7, device="cuda")
    sb = torch.tensor(0.3, device="cuda")
    out = torch.empty(128, 512, device="cuda", dtype=torch.bfloat16)
    module = build(
        m=128, n=512, k=512, dtype="e4m3", scaled=True, operand_k_tile=128, **options
    )
    make_ptx_launcher(make_ptx_kernel(module))(a, bt, out, sa, sb)
    torch.cuda.synchronize()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (a.float() @ bt.float().T) * sa * sb
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert torch.isfinite(out).all()
    assert ((out.float() - reference).norm() / reference.norm()).item() < 0.005


@requires_cuda
@pytest.mark.parametrize("shape", [(128, 4096, 4096), (2048, 11008, 4096)])
def test_public_tuned_schedule_and_compile(shape):
    m, n, k = shape
    torch.manual_seed(121)
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.7, 0.3))
    compiled = torch.compile(scaled_mm, fullgraph=True)
    actual = compiled(a, b.T, sa, sb)
    assert actual.shape == (m, n)
    assert actual.is_contiguous()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (a.float() @ b.float().T) * sa * sb
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert torch.isfinite(actual).all()
    assert ((actual.float() - reference).norm() / reference.norm()).item() < 0.005


@requires_cuda
@pytest.mark.parametrize("cyclic", [False, True])
def test_fp8_independent_operand_rings(cyclic):
    # Cross two complete A3/B4 ring periods, with partial output tiles.
    m, n, k = 257, 384, 4096
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e5m2)
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.5, 0.125))
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    module = build(
        m=m, n=n, k=k, dtype="e4m3", rhs_dtype="e5m2", scaled=True,
        output_m_tile=256, warpgroup_m_tile=128,
        output_n_tile=168, warpgroup_n_tile=168, operand_k_tile=128,
        operand_stages=3, operand_stage_overrides={"mm_rhs_smem": 4},
        cyclic_operand_pipeline=cyclic, unsafe_reuse_smem=True,
        output_tma_inner_size=None,
    )
    allocations = find_ops_of_type(module, AllocOp)
    assert sum(op.attrs.get("label") == "mm_lhs_smem" for op in allocations) == 3
    assert sum(op.attrs.get("label") == "mm_rhs_smem" for op in allocations) == 4
    kernel = make_ptx_kernel(module)
    assert kernel.block_dim == (256, 1, 1)
    assert "setmaxnreg" not in kernel.ptx
    launch = make_ptx_launcher(kernel)
    launch(a, b, out, sa, sb)
    torch.cuda.synchronize()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = (a.float() @ b.float().T) * sa * sb
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert ((out.float() - reference).norm() / reference.norm()).item() < 0.005
    before = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(a, b, out, sa, sb)
    sa.mul_(0.5)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, before * 0.5, atol=0, rtol=0)


@pytest.mark.parametrize("mt,wg_m", [(64, 64), (128, 64), (128, 128), (256, 128)])
def test_cooperative_launch_has_only_compute_warpgroups(mt, wg_m):
    kernel = make_ptx_kernel(
        build(
            m=128,
            n=256,
            k=512,
            dtype="e4m3",
            scaled=True,
            output_m_tile=mt,
            output_n_tile=128,
            warpgroup_m_tile=wg_m,
            warpgroup_n_tile=128,
            operand_k_tile=128,
        )
    )
    assert kernel.block_dim == (128 * (mt // wg_m), 1, 1)
    assert "setmaxnreg" not in kernel.ptx


@requires_cuda
@pytest.mark.parametrize("da,db", list(itertools.product(FORMATS, repeat=2)))
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("tile", [(128, 64, 256, 128), (256, 128, 168, 128), (256, 128, 168, None)])
def test_fp8_cooperative_shared_arena_reuse(da, db, cyclic, tile):
    # Four FP8 operand stages fit only when BF16 writeback reuses the arena.
    mt, wg_m, nt, output_inner = tile
    m, n, k = 257, 256 if nt == 256 else 384, 1024
    a = torch.randn(m, k, device="cuda").to(da[1])
    b = torch.randn(n, k, device="cuda").to(db[1])
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.7, 0.3))
    kernel = make_ptx_kernel(
        build(
            m=m,
            n=n,
            k=k,
            dtype=da[0],
            rhs_dtype=db[0],
            scaled=True,
            output_m_tile=mt,
            output_n_tile=nt,
            warpgroup_m_tile=wg_m,
            warpgroup_n_tile=nt,
            operand_k_tile=128,
            operand_stages=4,
            output_grid_size=2,
            output_tma_inner_size=output_inner,
            unsafe_reuse_smem=True,
            cyclic_operand_pipeline=cyclic,
        )
    )
    assert kernel.block_dim == (256, 1, 1)
    assert "setmaxnreg" not in kernel.ptx
    launch = make_ptx_launcher(kernel)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(a, b, out, sa, sb)
    for factor in (0.7, 0.4):
        sa.fill_(factor)
        graph.replay()
        torch.cuda.synchronize()
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            reference = (a.float() @ b.float().T) * sa * sb
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        assert torch.isfinite(out).all()
        assert ((out.float() - reference).norm() / reference.norm()).item() < 0.005
