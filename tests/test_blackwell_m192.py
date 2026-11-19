"""M192 compositions must share operands and preserve both row regions."""

import re

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    AllocOp,
    MatmulOp,
    SmemTensorType,
    TmemTensorType,
    root_buffer,
    static_tmem_columns,
)
from ayla.ptx_codegen import make_ptx_kernel


def build(**changes):
    options = dict(
        m=576,
        n=192,
        k=192,
        dtype="f16",
        output_m_tile=192,
        output_n_tile=192,
        operand_stages=3,
        output_store="stmatrix",
        allow_partial_tiles=True,
    )
    options.update(changes)
    return blackwell_tcgen05_gemm_module(**options)


def test_native_accumulators_share_one_operand_panel_per_stage():
    module = build()
    allocations = find_ops_of_type(module, AllocOp)
    tmem = [
        op.result().type
        for op in allocations
        if isinstance(op.result().type, TmemTensorType)
    ]
    assert len(tmem) == 1
    assert list(tmem[0].sizes) == [128, 384] and tmem[0].columns == 512
    matmuls = find_ops_of_type(module, MatmulOp)
    assert {op.dst.type.sizes[0] for op in matmuls} == {64, 128}
    assert {
        static_tmem_columns(op.dst) for op in matmuls if op.dst.type.sizes[0] == 128
    } == {(0, 192)}
    assert {
        static_tmem_columns(op.dst) for op in matmuls if op.dst.type.sizes[0] == 64
    } == {(192, 384)}
    for rows in (64, 128):
        assert sum(op.dst.type.sizes[0] == rows for op in matmuls) == 12
        # Initialization must overwrite each accumulator on its first K16.
        assert (
            sum(not op.accumulate for op in matmuls if op.dst.type.sizes[0] == rows)
            == 1
        )
    shared = [
        op.result()
        for op in allocations
        if isinstance(op.result().type, SmemTensorType)
    ]
    lhs = {root_buffer(op.lhs) for op in matmuls}
    rhs = {root_buffer(op.rhs) for op in matmuls}
    assert len(lhs) == len(rhs) == 3
    assert lhs.isdisjoint(rhs)
    assert all(
        value in shared and list(value.type.sizes) == [192, 64] for value in lhs | rhs
    )


def test_scoped_promotion_rejects_overlapping_unequal_row_regions():
    from ayla.gpu_dialect import gpu_dialect
    from ayla.passes.compose_matmul_rows import ComposeMatmulRows
    from ayla.passes.promote_buffers import PromoteBuffers
    from ayla.passes.tile_matmul import TileMatmul

    module = gpu_dialect.parse_module("""module attrs={target:"sm_100a"} {
      kernel @overlap(%a:smem<192x128xf16>,%b:smem<192x128xf16>,%c:gmem<192x192xf16>) {
        %zero:rmem<f32> = constant 0.0
        matmul(%a:smem<192x128xf16>,%b:smem<192x128xf16>,%c:gmem<192x192xf16>,%zero:rmem<f32>) collective=xyz.block attrs={label:"mm"}
      }
    }""")
    module = TileMatmul(
        op_dims=["K"],
        tile_sizes=[64],
        iteration_bindings=[None],
        for_loop_labels=["k"],
        op_label_filter="mm",
    ).run(module)
    module = ComposeMatmulRows(label_filter="mm").run(module)
    head, tail = find_ops_of_type(module, MatmulOp)
    # Move the 64-row tail to [64,128), which overlaps the 128-row head.
    tail.dst.definition.attrs["static_indices"] = [1, 0]
    assert head.dst.type.sizes[0] == 128 and tail.dst.type.sizes[0] == 64
    with pytest.raises(ValueError, match="overlap"):
        PromoteBuffers(
            target_type="tmem",
            operands=["dst"],
            target_dtype="f32",
            scope_loop_label="k",
            label_filter="mm",
            warps=4,
        ).run(module)


@pytest.mark.parametrize("marker", [1, None, "true"])
def test_native_tmem_view_requires_an_explicit_boolean(marker):
    from ayla.gpu_dialect import TileOp
    from ayla.passes.verify_gpu_dialect import VerifyGpuDialect

    module = build()
    native = next(
        op
        for op in find_ops_of_type(module, TileOp)
        if op.attrs.get("native_tmem_view")
    )
    native.attrs["native_tmem_view"] = marker
    with pytest.raises(ValueError, match="native_tmem_view must be a boolean"):
        VerifyGpuDialect().run(module)


def test_composed_arena_retires_at_its_final_tmem_read():
    from ayla.gpu_dialect import DeallocOp, ElementwiseOp, MoveOp

    module = build()
    release = next(
        op
        for op in find_ops_of_type(module, DeallocOp)
        if isinstance(op.value.type, TmemTensorType)
    )
    ops = release.parent_block.ops
    before, after = ops[: ops.index(release)], ops[ops.index(release) + 1 :]
    assert any(
        isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType)
        for op in before
    )
    assert not any(
        isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType)
        for op in after
    )
    assert any(isinstance(op, ElementwiseOp) and op.fn == "convert" for op in after)


@pytest.mark.parametrize(
    "options",
    [
        dict(k=64, operand_stages=1),
        dict(k=576, cyclic_operand_pipeline=True),
        dict(k=576, rolled_operand_pipeline=True),
        dict(k=576, output_grid_size=2, reuse_tmem_allocation=True),
        dict(m=50176, cluster_m_ctas=4),
        dict(n=384, output_n_tile=192, cluster_n_ctas=2),
        dict(output_store="global"),
        dict(dtype="bf16", output_layout="tcgen05_16x256b"),
        dict(output_column_major=True),
        dict(cta_group=2, tmem_writeback_register_budget=128),
        dict(cta_group=2, cluster_m_ctas=4, tmem_writeback_register_budget=128),
    ],
)
def test_m192_lowers_to_only_legal_native_mmas(options):
    kernel = make_ptx_kernel(build(**options))
    descriptors = re.findall(
        r"mov.b32 %r\d+, (\d+);\n  mov.pred %p\d+, [01];\n  @%p\d+ tcgen05.mma",
        kernel.ptx,
    )
    assert descriptors
    group = options.get("cta_group", 1)
    assert {((int(value) >> 24) & 31) * 16 for value in descriptors} == {
        64 * group,
        128 * group,
    }
    assert "wgmma.mma" not in kernel.ptx


@pytest.mark.parametrize(
    "options",
    [
        dict(cta_group=2),
        dict(output_n_tile=384),
        dict(persistent_output_pipeline=True),
        dict(batch_output_m_operands=True),
    ],
)
def test_unsupported_m192_compositions_fail_before_scheduling(options):
    with pytest.raises(ValueError, match="M192 composition|shared ring requires"):
        build(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(m=576, k=64, operand_stages=1),
        dict(m=769, k=192),
        dict(m=50176, k=192, cluster_m_ctas=4),
        dict(m=769, k=576, cyclic_operand_pipeline=True, output_grid_size=2),
        dict(m=769, k=576, rolled_operand_pipeline=True),
        dict(m=769, k=576, output_grid_size=2, reuse_tmem_allocation=True),
        dict(m=769, k=192, dtype="bf16", output_layout="tcgen05_16x256b"),
        dict(m=576, k=192, output_store="global"),
        dict(m=776, k=192, output_column_major=True),
        dict(m=288, k=144, operand_k_tile=16, output_column_major=True),
        dict(m=769, k=192, cta_group=2, tmem_writeback_register_budget=128),
        dict(
            m=769,
            k=192,
            cta_group=2,
            cluster_m_ctas=4,
            tmem_writeback_register_budget=128,
        ),
    ],
)
def test_m192_changed_input_graph_replays_preserve_rows_and_canaries(options):
    import torch
    from ayla.ptx_runtime import assemble_ptx, make_ptx_launcher

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires a Blackwell GPU")
    kernel = make_ptx_kernel(build(**options))
    launch = make_ptx_launcher(
        kernel, assemble_ptx(kernel, "/usr/local/cuda-13.0/bin/ptxas")
    )
    m, k = options["m"], options["k"]
    dtype = torch.bfloat16 if options.get("dtype") == "bf16" else torch.float16
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(192, k, device="cuda", dtype=dtype)
    if options.get("rhs_column_major"):
        b = b.T.contiguous().T
    storage = torch.full((m * 192 + 256,), 71.0, device="cuda", dtype=dtype)
    c = (
        storage[128:-128].view(192, m).T
        if options.get("output_column_major")
        else storage[128:-128].view(m, 192)
    )
    bias = torch.randn(192, device="cuda", dtype=dtype) if options.get("bias") else None
    args = (bias.expand(m, 192), a, b, c) if bias is not None else (a, b, c)
    launch(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(*args)
    for seed in (17, 43, 71):
        torch.manual_seed(seed)
        a.normal_()
        b.normal_()
        c.fill_(float("nan"))
        expected = a.float() @ b.float().T
        if bias is not None:
            bias.normal_()
            expected += bias.float()
        expected = expected.to(dtype)
        graph.replay()
        torch.cuda.synchronize()
        # Check head and tail independently: a missing M64 component cannot
        # hide in a whole-matrix RMS measurement.
        for origin, rows in ((0, 128), (128, 64)):
            mask = torch.arange(m, device="cuda") % 192
            mask = (mask >= origin) & (mask < origin + rows)
            error = (c[mask].float() - expected[mask].float()).square().mean()
            scale = expected[mask].float().square().mean()
            assert float((error / scale).sqrt()) < 0.002
        assert bool((storage[:128] == 71.0).all() & (storage[-128:] == 71.0).all())
