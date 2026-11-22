"""Owned warp fragments, tail coordinates and completion before shared reuse."""

from collections import Counter

import pytest
import torch

from ayla.base.ir import Block, Value, clone_type, find_ops_of_type
from ayla.analysis.sync import SyncAnalysis
from ayla.gemm_collection.gemm import make_fp8_gemm_kernel
from ayla.gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    CollectiveAxis,
    CollectiveLevel,
    F32Type,
    GmemTensorType,
    IndexConstantOp,
    KernelOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    TmaMoveOp,
    WgrmemTensorType,
    WarpIdsOp,
    ReinterpretOp,
)
from ayla.passes.folding import Folding
from ayla.passes.lower_to_async import LowerToAsync
from ayla.passes.lower_to_tma_move import LowerToTmaMove
from ayla.passes.schedule_async import ScheduleAsync
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_runtime import make_ptx_launcher


@pytest.mark.parametrize("rows,columns", [(64, 168), (128, 168), (128, 192)])
def test_warp_views_partition_every_accumulator_register(rows, columns):
    root = AllocOp.build_with_fresh_id(
        WgrmemTensorType.build(
            [rows, columns], F32Type(), warps=4, layout="wgmma_layout_0"
        )
    ).result()
    selected = []
    for stream in range(rows // 64):
        for column, start in enumerate(range(0, columns, 64)):
            for half in range(2):
                typ = WgrmemTensorType.build(
                    [8, min(64, columns - start)],
                    F32Type(),
                    warps=1,
                    layout="wgmma_warp",
                )
                view = TileOp.build(
                    root, [8 * stream + half, column], typ, index_steps=[8, 64]
                ).result()
                layout = typ.register_layout(view)
                selected.extend(typ.register_indices(view))
                coverage = Counter(
                    layout.coordinates_for_thread(e, lane)
                    for lane in range(32)
                    for e in layout.elements
                )
                assert coverage == Counter(
                    {
                        (r, c): 1
                        for r in range(8)
                        for c in range(min(64, columns - start))
                    }
                )
    assert Counter(selected) == Counter(root.type.register_indices(root))


def test_smaller_tail_retains_the_regular_iteration_stride():
    gmem = Value("g", GmemTensorType.build([64, 168], F32Type()))
    kernel = KernelOp.build(
        "tail", Block(arguments=[gmem]), grid_dim=[1, 1, 1], block_dim=[128, 1, 1]
    )
    shared = AllocOp.build_with_fresh_id(SmemTensorType.build([64, 40], F32Type()))
    tile = TileOp.build(
        gmem, [0, 2], clone_type(gmem.type, {"sizes": [64, 40]}), index_steps=[64, 64]
    )
    store = MoveOp.build(
        tile.result(),
        shared.result(),
        collective={axis: CollectiveLevel.BLOCK for axis in CollectiveAxis},
        attrs={"tma": True},
    )
    kernel.body.ops = [shared, tile, store]
    module = ModuleOp.build(Block(ops=[kernel]))
    LowerToTmaMove().run(module)
    Folding().run(module)
    [copy] = find_ops_of_type(module, TmaMoveOp)
    assert copy.dst.type.tile_sizes == (64, 40)
    assert isinstance(copy.indices[1].definition, IndexConstantOp)
    assert copy.indices[1].definition.value == 128


@pytest.mark.parametrize(
    "stores,arenas,overlap", [(3, 2, False), (10, 8, False), (3, 2, True)]
)
def test_bulk_store_completion_stays_before_shared_source_reuse(
    stores, arenas, overlap
):
    output = Value("g", GmemTensorType.build([64, stores * 64], F32Type()))
    source = Value(
        "r",
        WgrmemTensorType.build([64, 64], F32Type(), warps=4, layout="wgmma_layout_0"),
    )
    kernel = KernelOp.build(
        "queue",
        Block(arguments=[output, source]),
        grid_dim=[1, 1, 1],
        block_dim=[128, 1, 1],
    )
    collective = {axis: CollectiveLevel.BLOCK for axis in CollectiveAxis}
    buffers = [
        AllocOp.build_with_fresh_id(SmemTensorType.build([64, 64], F32Type()))
        for _ in range(arenas)
    ]
    ops = list(buffers)
    for index in range(stores):
        tile = TileOp.build(
            output,
            [0, 0 if overlap and index == 1 else index],
            clone_type(output.type, {"sizes": [64, 64]}),
        )
        gather = MoveOp.build(
            buffers[index % arenas].result(),
            source,
            collective=collective,
            attrs={"label": f"publish{index}"},
        )
        store = MoveOp.build(
            tile.result(),
            buffers[index % arenas].result(),
            collective=collective,
            attrs={"tma": True, "label": f"store{index}"},
        )
        ops.extend([tile, gather, store])
    kernel.body.ops = ops
    module = ModuleOp.build(Block(ops=[kernel]))
    ScheduleAsync().run(LowerToAsync().run(module))
    issues = find_ops_of_type(module, AsyncOp)
    waits = {w.lock: w for w in find_ops_of_type(module, AwaitOp)}
    first = waits[issues[0].lock]
    assert first.attrs.get("pending", 0) == (0 if overlap else arenas - 1)
    body = kernel.body.ops
    for index in range(arenas, stores):
        reuse = next(op for op in body if op.attrs.get("label") == f"publish{index}")
        assert body.index(waits[issues[index - arenas].lock]) < body.index(reuse)
    assert all(0 <= wait.attrs.get("pending", 0) <= 7 for wait in waits.values())
    if overlap:
        assert body.index(first) < body.index(issues[1])
    else:
        assert body.index(issues[arenas - 1]) < body.index(first)
        LowerToTmaMove().run(module)
        analysis = SyncAnalysis()
        analysis.analyze(module)
        effects = [
            effect
            for effects in analysis.live_out(first).concurrent_memory_effects.values()
            for effect in effects
        ]
        assert all(effect.pending_await is not issues[0].lock for effect in effects)
        assert any(
            effect.lock is issues[1].lock and effect.completed_after is None
            for effect in effects
        )
    assert waits[issues[-1].lock].attrs.get("pending", 0) == 0


@pytest.mark.parametrize("steps", [[0, 64], [64], [64, True]])
def test_rejects_invalid_tile_iteration_strides(steps):
    root = Value("g", GmemTensorType.build([64, 168], F32Type()))
    tile = TileOp.build(
        root, [0, 2], clone_type(root.type, {"sizes": [64, 40]}), index_steps=steps
    )
    module = ModuleOp.build(Block(ops=[tile]))
    with pytest.raises(ValueError, match="index_steps"):
        verify_gpu_dialect(module)


@pytest.mark.parametrize("owned", [False, True])
def test_private_shared_scope_requires_the_issuing_warp_slice(owned):
    source = Value(
        "r", WgrmemTensorType.build([8, 64], F32Type(), warps=1, layout="wgmma_warp")
    )
    kernel = KernelOp.build(
        "owned", Block(arguments=[source]), grid_dim=[1, 1, 1], block_dim=[128, 1, 1]
    )
    warp = WarpIdsOp.build()
    shared = AllocOp.build_with_fresh_id(
        SmemTensorType.build([32, 64], F32Type(), attrs={"scope": "warp"})
    )
    tile = TileOp.build(
        shared.result(),
        [warp.x if owned else 0, 0],
        clone_type(shared.result().type, {"sizes": [8, 64]}),
    )
    move = MoveOp.build(
        tile.result(),
        source,
        collective={axis: CollectiveLevel.BLOCK for axis in CollectiveAxis},
    )
    kernel.body.ops = [warp, shared, tile, move]
    module = ModuleOp.build(Block(ops=[kernel]))
    if owned:
        verify_gpu_dialect(module)
    else:
        with pytest.raises(ValueError, match="issuing warp"):
            verify_gpu_dialect(module)


@pytest.mark.parametrize("width,offset", [(72, 0), (40, 16)])
def test_reinterpret_cannot_escape_the_owned_fragment_arena(width, offset):
    source = Value(
        "r", WgrmemTensorType.build([8, width], F32Type(), warps=1, layout="wgmma_warp")
    )
    kernel = KernelOp.build(
        "escape", Block(arguments=[source]), grid_dim=[1, 1, 1], block_dim=[128, 1, 1]
    )
    warp = WarpIdsOp.build()
    shared = AllocOp.build_with_fresh_id(
        SmemTensorType.build([32, 64], F32Type(), attrs={"scope": "warp"})
    )
    tile = TileOp.build(
        shared.result(),
        [warp.x, 0],
        SmemTensorType.build([8, 64], F32Type(), attrs={"scope": "warp"}),
    )
    tail = ReinterpretOp.build(
        tile.result(),
        SmemTensorType.build([8, width], F32Type(), attrs={"scope": "warp"}),
        byte_offset=offset,
    )
    move = MoveOp.build(
        tail.result(),
        source,
        collective={axis: CollectiveLevel.BLOCK for axis in CollectiveAxis},
    )
    kernel.body.ops = [warp, shared, tile, tail, move]
    with pytest.raises(ValueError, match="reinterpret must fit"):
        verify_gpu_dialect(ModuleOp.build(Block(ops=[kernel])))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "m,rows,n,width,buffers", [(73, 64, 184, 192, 8), (129, 128, 256, 256, 4)]
)
def test_single_warpgroup_with_default_tma_layout(m, rows, n, width, buffers):
    config = dict(
        m=m,
        n=n,
        k=128,
        dtype="e4m3",
        scaled=True,
        output_m_tile=rows,
        warpgroup_m_tile=rows,
        output_n_tile=width,
        warpgroup_n_tile=width,
        operand_k_tile=128,
        unsafe_reuse_smem=True,
    )
    baseline = make_fp8_gemm_kernel(**config, output_tma_inner_size=None)
    fragmented = make_fp8_gemm_kernel(
        **config, output_fragment_cols=64, output_store_buffers=buffers
    )
    assert "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group" in fragmented.ptx
    a = torch.randn(m, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, 128, device="cuda").to(torch.float8_e4m3fn)
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.25, 0.125))
    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    reference, run = make_ptx_launcher(baseline), make_ptx_launcher(fragmented)
    reference(a, b, expected, sa, sb)
    for _ in range(20):
        run(a, b, actual, sa, sb)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("output_dtype", ["bf16", "f16"])
@pytest.mark.parametrize("reuse", [False, True])
def test_swizzled_fragments_keep_padded_tails_owned_across_mutable_replays(
    output_dtype, reuse
):
    config = dict(
        m=257, n=344, k=512, dtype="e4m3", rhs_dtype="e5m2",
        output_dtype=output_dtype, scaled=True, output_m_tile=256,
        warpgroup_m_tile=128, output_n_tile=168, warpgroup_n_tile=168,
        operand_k_tile=128, operand_stages=3, output_tma_inner_size=None,
        unsafe_reuse_smem=reuse,
    )
    baseline = make_ptx_launcher(make_fp8_gemm_kernel(
        **{**config, "unsafe_reuse_smem": True}
    ))
    kernel = make_fp8_gemm_kernel(
        **config, output_fragment_cols=64, output_fragment_swizzle="128B",
        output_store_buffers=2,
    )
    assert "stmatrix.sync.aligned.m8n8.x1" in kernel.ptx
    assert "cp.async.bulk.wait_group.read 1;" in kernel.ptx
    assert "setmaxnreg" not in kernel.ptx
    descriptors = [t for t in kernel.signature if getattr(t, "arg_idx", None) == 2]
    assert {tuple(t.tile_sizes) for t in descriptors} == {(8, 64), (8, 40)}
    assert all(t.swizzle == "128B" for t in descriptors)
    run = make_ptx_launcher(kernel)
    a = torch.randn(257, 512, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(344, 512, device="cuda").to(torch.float8_e5m2)
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.25, 0.125))
    dtype = torch.bfloat16 if output_dtype == "bf16" else torch.float16
    expected = torch.empty(257, 344, device="cuda", dtype=dtype)
    actual = torch.empty_like(expected)
    baseline(a, b, expected, sa, sb)
    run(a, b, actual, sa, sb)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(a, b, actual, sa, sb)
    for scale in (0.5, 0.125, 0.25):
        a.copy_(torch.randn(a.shape, device="cuda").to(a.dtype))
        b.copy_(torch.randn(b.shape, device="cuda").to(b.dtype))
        sa.fill_(scale)
        sb.fill_(scale / 2)
        baseline(a, b, expected, sa, sb)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cols,dtype", [(56, "bf16"), (64, "f32")])
def test_swizzled_fragments_reject_unsupported_physical_rows(cols, dtype):
    with pytest.raises(ValueError, match="N64 and 16-bit"):
        make_fp8_gemm_kernel(
            m=256, n=168, k=128, dtype="e4m3", output_dtype=dtype,
            output_m_tile=256, warpgroup_m_tile=128, output_n_tile=168,
            warpgroup_n_tile=168, operand_k_tile=128,
            output_fragment_cols=cols, output_fragment_swizzle="128B",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("buffers", [1, 2, 3])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "lhs,rhs,out",
    [
        ("e4m3", "e4m3", "bf16"),
        ("e4m3", "e5m2", "f16"),
        ("e5m2", "e4m3", "f32"),
    ],
)
def test_partial_tiles_mixed_fp8_and_changing_graph_scales(
    buffers, reuse, lhs, rhs, out
):
    m, n, k = 257, 384, 512
    config = dict(
        m=m,
        n=n,
        k=k,
        dtype=lhs,
        rhs_dtype=rhs,
        output_dtype=out,
        scaled=True,
        output_m_tile=256,
        warpgroup_m_tile=128,
        output_n_tile=168,
        warpgroup_n_tile=168,
        operand_k_tile=128,
        operand_stages=2 if out == "f32" else 3,
        output_tma_inner_size=None,
        unsafe_reuse_smem=reuse,
    )
    control = make_fp8_gemm_kernel(**{**config, "unsafe_reuse_smem": True})
    fragmented = make_fp8_gemm_kernel(
        **config, output_fragment_cols=64, output_store_buffers=buffers
    )
    assert "wgmma.mma_async" in fragmented.ptx
    assert "stmatrix.sync" in fragmented.ptx or out == "f32"
    assert "cp.async.bulk.wait_group.read 1;" in fragmented.ptx or buffers == 1
    assert "setmaxnreg" not in fragmented.ptx
    dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    a = torch.randn(m, k, device="cuda").to(dtype[lhs])
    b = torch.randn(n, k, device="cuda").to(dtype[rhs])
    sa, sb = (torch.tensor(x, device="cuda") for x in (0.25, 0.125))
    output_dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[
        out
    ]
    expected = torch.empty(m, n, device="cuda", dtype=output_dtype)
    actual = torch.empty_like(expected)
    baseline, run = make_ptx_launcher(control), make_ptx_launcher(fragmented)
    baseline(a, b, expected, sa, sb)
    run(a, b, actual, sa, sb)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(a, b, actual, sa, sb)
    for scale in (0.5, 0.125, 0.25):
        sa.fill_(scale)
        baseline(a, b, expected, sa, sb)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
