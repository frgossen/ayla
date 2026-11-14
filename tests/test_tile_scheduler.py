"""Tile task lowering, cooperative dispatch, and initialized scratch replay."""

from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import ayla.forward_compile as forward
from ayla.base.ir import clone_op, find_ops_of_type
from ayla.fx_compiler import discover_fx_graph, fx_region_to_module
from ayla.gpu_dialect import AllocOp, BarrierOp, GmemTensorType, IfOp, TileSyncOp
from ayla.passes.schedule_persistent_graph import SchedulePersistentGraph
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel


def test_static_retiling_preserves_nonzero_tail_coordinates():
    from ayla.base.ir import Block, Value
    from ayla.gemm_collection.swiglu import schedule_swiglu_projection_wave_tail
    from ayla.gpu_dialect import BF16Type, F32Type, KernelOp, ModuleOp
    from ayla.passes.schedule_tile_graph import _extract_root
    from ayla.analysis.tile_dependency import TileDependencyAnalyzer

    arguments = [Value(name, GmemTensorType.build(shape, BF16Type()))
                 for name, shape in [("x", (256, 512)), ("w", (6144, 512)),
                                     ("out", (256, 3072))]]
    sync = Value("forward_sync", GmemTensorType.build([2], F32Type()))
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "static_tail", Block(arguments=[*arguments, sync]))]))
    schedule_swiglu_projection_wave_tail(
        module, *arguments, blocks=5, grid_sync=sync, n_tile=192,
        tail_n_tile=128, operand_stages=3, tail_operand_stages=3,
        block_store=True, k_tail_tiles=0,
    )
    operations = module.body.ops[0].body.ops
    boundary = next(i for i, op in enumerate(operations)
                    if isinstance(op, BarrierOp) and op.scope == "grid")
    main = _extract_root(operations[:boundary], 5)
    tail = _extract_root(operations[boundary + 1:], 5)
    # The library peels this single tail wave into a guarded direct tile;
    # its fifth lane has an empty access set.
    assert (main.count, tail.count) == (30, 5)
    writes = [access.region for access in TileDependencyAnalyzer().accesses(
        tail.body, tail.tile, 0) if access.write and access.root is arguments[2]]
    assert writes
    assert min(region.intervals[0][0] for region in writes) == 2816 * 2
    assert not any(access.write for access in TileDependencyAnalyzer().accesses(
        tail.body, tail.tile, 4))


def test_static_lookahead_stops_at_final_wave():
    from ayla.base.ir import Block, Op, Value
    from ayla.gemm_collection.swiglu import schedule_swiglu_projection
    from ayla.gpu_dialect import BF16Type, KernelOp, ModuleOp
    from ayla.passes.schedule_tile_graph import _extract_root
    from ayla.analysis.tile_dependency import TileDependencyAnalyzer

    arguments = [Value(name, GmemTensorType.build(shape, BF16Type()))
                 for name, shape in [("x", (256, 512)), ("w", (1536, 512)),
                                     ("out", (256, 768))]]
    module = ModuleOp.build(Block(ops=[KernelOp.build(
        "static_lookahead", Block(arguments=arguments))]))
    schedule_swiglu_projection(
        module, *arguments, blocks=2, n_tile=192, operand_stages=3,
        cyclic_operand_pipeline=False, k_tail_tiles=0, block_store=True,
        output_prefetch_tiles=2,
    )
    root = _extract_root(module.body.ops[0].body.ops, 2)
    futures = [branch for outer in root.body.ops for branch in find_ops_of_type(outer, IfOp)
               if any(str(op.attrs.get("label", "")).startswith("output_prefetch_")
                      for op in branch.body.ops for op in find_ops_of_type(op, Op))]
    # Retain just the immediate guarded prefetch branch, not enclosing guards.
    futures = [branch for branch in futures if not find_ops_of_type(branch.body.ops[0], IfOp)]
    assert futures
    analyzer = TileDependencyAnalyzer()
    assert any(analyzer.accesses(Block(ops=[branch]), root.tile, 0) for branch in futures)
    assert not any(analyzer.accesses(Block(ops=[branch]), root.tile, root.count - 1)
                   for branch in futures)


def _mlp(x, weight, down):
    packed = x @ weight.T
    width = weight.shape[0] // 2
    gate = packed[:, :width].float()
    hidden = (gate / (1 + torch.exp(-gate))).bfloat16() * packed[:, width:]
    return hidden @ down.T


def test_row_chunk_acquires_union_once_and_retains_producer_publications():
    from ayla.analysis.cross_loop_scheduler import schedule_tiles, specialize_static_waits
    from ayla.analysis.tile_dependency import TileTask, TileDependencyGraph
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import IndexType
    from ayla.passes.schedule_tile_graph import _TileRoot, _coalesce_row_entry_waits
    tasks = tuple(TileTask(root, tile, ()) for root, tile in [(0, 0), (0, 1), (1, 0), (1, 1)])
    graph = TileDependencyGraph(tasks, (frozenset(), frozenset(), frozenset({0}), frozenset({1})))
    roots = [_TileRoot([], Block(), Value(None, IndexType()), 2, 2) for _ in range(2)]
    plan = _coalesce_row_entry_waits(schedule_tiles(graph), roots, [[1, 1], [0, 0]])
    plan = specialize_static_waits(plan, 2, workers=(1, 1, 0, 0))
    assert plan.records[2][2:4] == (1, 2)
    assert plan.records[3][2:4] == (0, 0)
    assert plan.events[0].producers == (0, 1)
    assert plan.publications == (1, 1)


def test_independent_gemm_acquires_all_output_tile_inputs_once():
    from ayla.analysis.cross_loop_scheduler import schedule_tiles, specialize_static_waits
    from ayla.analysis.tile_dependency import TileTask, TileDependencyGraph
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import BF16Type, IndexType, MatmulOp
    from ayla.passes.schedule_tile_graph import _TileRoot, _coalesce_row_entry_waits
    values = [Value(name, GmemTensorType.build((128, 128), BF16Type()))
              for name in ("lhs", "rhs", "dst")]
    root = _TileRoot([], Block(ops=[MatmulOp.build(*values, collective={})]), Value(None, IndexType()), 2, 1)
    tasks = tuple(TileTask(r, t, ()) for r, t in ((0, 0), (0, 1), (1, 0), (1, 1)))
    graph = TileDependencyGraph(tasks, (frozenset(), frozenset(), frozenset({0}), frozenset({1})))
    original = schedule_tiles(graph)
    coalesced = _coalesce_row_entry_waits(original, [root, root], [[1, 1], [0, 0]],
                                         eligible_async=frozenset({1}))
    plan = specialize_static_waits(coalesced, 2, workers=(1, 1, 0, 0))
    assert plan.events[plan.records[2][2] - 1].producers == (0, 1)
    assert plan.records[2][3] == 2 and plan.records[3][2:4] == (0, 0)
    assert plan.publications == (1, 1)
    # Early workers must keep the first input tile's narrower acquire so
    # later producer work cannot delay that first tile.
    root.early_workers = (0,)
    assert _coalesce_row_entry_waits(original, [root, root], [[1, 1], [0, 0]],
                                     eligible_async=frozenset({1})) is original




def test_one_wave_gemm_completes_rows_for_successive_head_consumers():
    from ayla.analysis.interval_set import IntervalSet
    from ayla.analysis.tile_dependency import TileAccess, TileTask, TileDependencyAnalyzer
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import BF16Type, CollectiveAxis, CollectiveLevel, IndexType, KernelOp, MatmulOp, ModuleOp
    from ayla.passes.schedule_tile_graph import _TileRoot, _extract_root, _assign_ready_gemms
    from ayla.pipelines.tile_matmul_pipeline import _schedule_matmul
    args = [Value(name, GmemTensorType.build(shape, BF16Type())) for name, shape in
            [("x", (512, 512)), ("w", (1024, 512)), ("y", (512, 1024))]]
    module = ModuleOp.build(Block(ops=[KernelOp.build("one_wave", Block(arguments=args,
        ops=[MatmulOp.build(*args, collective={axis: CollectiveLevel.GRID for axis in CollectiveAxis}, attrs={"label": "mm"})]))]))
    module = _schedule_matmul(module, output_m_tile=64, output_n_tile=256, warpgroup_m_tile=64,
                              warpgroup_n_tile=128, output_grid_size=32, output_grid_fast_axis="n",
                              output_tma_inner_size=128)
    gemm = _extract_root(module.body.ops[0].body.ops, 32)
    assert gemm.count == gemm.grid == 32
    normalizer = _TileRoot([], Block(), Value(None, IndexType()), 512, 36,
                          row_ranges=[(128, 140)] * 32 + [(0, 32), (32, 64), (64, 96), (96, 128)])
    consumers = [_TileRoot([], Block(), Value(None, IndexType()), 512, 36) for _ in range(2)]
    tasks = tuple(TileTask(0, row, (TileAccess(args[0], IntervalSet.build([(row * 1024, (row + 1) * 1024)]), True),))
                  for row in range(512)) + tuple(TileTask(1, tile, TileDependencyAnalyzer().accesses(gemm.body, gemm.tile, tile))
                  for tile in range(32)) + tuple(
        TileTask(root, row, (TileAccess(args[2], IntervalSet.build([(row * 2048 + first, row * 2048 + stop)]), False),))
        for root, first, stop in [(2, 1536, 2048), (3, 0, 1536)] for row in range(512))
    _assign_ready_gemms([normalizer, gemm, *consumers], tasks, 36, 256)
    assert gemm.worker_lanes[-4:] == [0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sequential_k_and_q_rows_replay_after_complete_row_gemm(monkeypatch):
    from ayla.forward_schedule_table import STATIC_GEMM_LAYOUT_OVERRIDES
    from ayla.torch_custom_call import to_torch_dtype
    original_schedule, original_emit = forward.SchedulePersistentGraph, forward.make_ptx_kernel
    captured = []

    def schedule(**kwargs):
        return original_schedule(**{**kwargs, "blocks": 36})

    def emit(module):
        captured.append(module.body.ops[0].attrs.copy())
        return original_emit(module)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    monkeypatch.setattr(forward, "make_ptx_kernel", emit)
    for n, k, wg_n in [(2048, 512, 64), (1024, 2048, 128)]:
        monkeypatch.setitem(STATIC_GEMM_LAYOUT_OVERRIDES, (36, 256, 512, n, k),
                            dict(output_m_tile=64, output_n_tile=2 * wg_n, warpgroup_m_tile=64,
                                 warpgroup_n_tile=wg_n, output_grid_size=32, output_grid_fast_axis="n",
                                 output_tma_inner_size=None))
    STATIC_GEMM_LAYOUT_OVERRIDES[36, 256, 512, 2048, 512]["output_grid_fast_axis"] = "m"

    def compute(x, w0, w1):
        h = x @ w0.T
        h = (h.float() * (h.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16()
        y = h @ w1.T
        k = y[:, 768:].clone().reshape(-1, 128).float()
        q = y[:, :768].clone().reshape(-1, 128).float()
        return ((k * (k.square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16(),
                (q * (q.square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16())

    inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .05
              for shape in [(512, 512), (2048, 512), (1024, 2048)]]
    fx = make_fx(compute)(*inputs)
    match = discover_fx_graph(fx.graph)
    module = fx_region_to_module(match, name="sequential_ready_heads")
    placeholders = [node for node in fx.graph.nodes if node.op == "placeholder"]
    args = [inputs[placeholders.index(node)] for node in match.inputs]
    outputs = [torch.empty_strided(typ.static_shape, typ.strides, device="cuda", dtype=to_torch_dtype(typ.dtype))
               for typ in module.body.ops[0].signature[len(args):]]
    runner = forward.compile_forward_module(module, tile_scheduler="static")()
    [attrs] = captured
    # A large producer M tile must not monopolize the four idle CTAs.
    # Sequential K/Q consumers keep balanced chunks rather than taking all
    # early head work on those same CTAs.
    for root in (1, 3, 4):
        lengths = [stop - first for first, stop in attrs["tile_row_ranges"][root]]
        assert max(lengths) <= 2 * -(-sum(lengths) // 36)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*args, *outputs)
    for delta in (0., .01, -.02):
        inputs[0].add_(delta)
        graph.replay()
        torch.cuda.synchronize()
        expected = {tuple(t.shape): t for t in compute(*inputs)}
        for actual in outputs:
            torch.testing.assert_close(actual, expected[tuple(actual.shape)], atol=.002, rtol=.02)


def test_row_frontier_does_not_hoist_a_wait_for_a_later_row():
    from ayla.analysis.cross_loop_scheduler import schedule_tiles
    from ayla.analysis.tile_dependency import TileTask, TileDependencyGraph
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import IndexType
    from ayla.passes.schedule_tile_graph import _TileRoot, _coalesce_row_entry_waits
    tasks = (TileTask(0, 0, ()), TileTask(1, 0, ()), TileTask(1, 1, ()))
    graph = TileDependencyGraph(tasks, (frozenset(), frozenset({0}), frozenset({1})))
    roots = [_TileRoot([], Block(), Value(None, IndexType()), count, 2) for count in (1, 2)]
    original = schedule_tiles(graph)
    assert _coalesce_row_entry_waits(original, roots, [[1], [0, 0]]) is original


def test_projection_acquires_current_rows_before_lookahead_rows():
    from ayla.analysis.cross_loop_scheduler import schedule_tiles
    from ayla.analysis.interval_set import IntervalSet
    from ayla.analysis.tile_dependency import TileAccess, TileTask, TileDependencyAnalyzer, build_tile_dependency_graph
    from ayla.base.ir import Block, Value
    from ayla.gemm_collection.swiglu import schedule_swiglu_projection
    from ayla.gpu_dialect import AsyncOp, BF16Type, F32Type, ForOp, IndexType, KernelOp, ModuleOp
    from ayla.passes.schedule_tile_graph import ScheduleTileGraph, _TileRoot, _extract_root, _assign_ready_gemms

    arguments = [Value(name, GmemTensorType.build(shape, BF16Type())) for name, shape in
                 [("normalized", (512, 512)), ("weight", (2304, 512)), ("out", (512, 1152))]]
    sync = Value("sync", GmemTensorType.build([2], F32Type()))
    module = ModuleOp.build(Block(ops=[KernelOp.build("lookahead", Block(arguments=[*arguments, sync]))]))
    schedule_swiglu_projection(module, *arguments, blocks=6, n_tile=192, operand_stages=3,
                               cyclic_operand_pipeline=False, k_tail_tiles=0, block_store=True,
                               output_prefetch_tiles=2)
    projection = _extract_root(module.body.ops[0].body.ops, 6)
    # The last physical lane normalizes the first 128 rows; other lanes own later rows.
    normalizer = _TileRoot([], Block(), Value("row", IndexType()), 512, 6,
                           row_ranges=[(128, 205), (205, 282), (282, 359), (359, 436), (436, 512), (0, 128)])
    tasks = tuple(TileTask(0, row, (TileAccess(arguments[0], IntervalSet(((row * 1024, (row + 1) * 1024),)), True),))
                  for row in range(512)) + tuple(
        TileTask(1, tile, TileDependencyAnalyzer().accesses(projection.body, projection.tile, tile))
        for tile in range(projection.count))
    _assign_ready_gemms([normalizer, projection], tasks, 6, 256)
    assert projection.worker_lanes is not None
    assert sorted(projection.worker_lanes) == list(range(6))
    assert projection.worker_lanes[5] in (0, 4)
    graph = build_tile_dependency_graph(tasks)
    original = schedule_tiles(graph)
    plan, info, tables = ScheduleTileGraph._split_lookahead_readiness(
        [normalizer, projection], tasks, graph, original, 256, sync)
    assert info and info[0]["root"] == 1
    record = next(record for record in plan.records if record[:2] == (1, 4))
    assert plan.events[record[2] - 1].producers == tuple(range(128))
    event, expected = tables["ayla_root_1_lookahead_readiness"][4]
    assert set(original.events[event - 1].producers) == set(range(128)) | set(range(256, 384))
    assert expected == 256
    # The previous tile's deferred acquire covers this tile's current rows.
    assert next(record for record in plan.records if record[:2] == (1, 10))[2:4] == (0, 0)
    assert tables["ayla_root_1_lookahead_readiness"][10][0] != 0
    # Both counters remain published, so the deferred full acquire preserves
    # the ordering proof used to elide checks on later tiles.
    for row in range(128):
        producer = plan.records[row]
        published = plan.publications[producer[4]:producer[4] + producer[5]]
        assert record[2] in published and event in published
    waits = [op for top in projection.body.ops for op in find_ops_of_type(top, TileSyncOp)]
    assert len(waits) == 1
    block = waits[0].parent_block
    while block is not None and block.parent_op is not None:
        assert not isinstance(block.parent_op, ForOp)  # No check in the GEMM K loop.
        block = block.parent_op.parent_block
    future = [op for top in projection.body.ops for op in find_ops_of_type(top, AsyncOp)
              if str(op.attrs.get("label", "")).startswith("output_prefetch_")]
    assert future


def test_early_gemm_lanes_complete_one_head_slice_before_spreading_rows():
    from ayla.analysis.interval_set import IntervalSet
    from ayla.analysis.tile_dependency import TileAccess, TileTask, TileDependencyAnalyzer
    from ayla.base.ir import Block, Value
    from ayla.gemm_collection.swiglu import schedule_swiglu_projection
    from ayla.gpu_dialect import BF16Type, IndexType, KernelOp, ModuleOp
    from ayla.passes.schedule_tile_graph import _TileRoot, _extract_root, _assign_ready_gemms
    arguments = [Value(name, GmemTensorType.build(shape, BF16Type())) for name, shape in
                 [("normalized", (512, 512)), ("weight", (3072, 512)), ("out", (512, 1536))]]
    module = ModuleOp.build(Block(ops=[KernelOp.build("head_priority", Block(arguments=arguments))]))
    schedule_swiglu_projection(module, *arguments, blocks=8, n_tile=192, operand_stages=3,
                               cyclic_operand_pipeline=False, k_tail_tiles=0, block_store=True)
    projection = _extract_root(module.body.ops[0].body.ops, 8)
    normalizer = _TileRoot([], Block(), Value(None, IndexType()), 512, 10,
                           row_ranges=[(256 + 32 * i, 288 + 32 * i) for i in range(8)] + [(0, 128), (128, 256)])
    head = _TileRoot([], Block(), Value(None, IndexType()), 1, 10)
    tasks = (TileTask(0, 0, (TileAccess(arguments[0], IntervalSet(((0, 512 * 512 * 2),)), True),)),) + tuple(
        TileTask(1, tile, TileDependencyAnalyzer().accesses(projection.body, projection.tile, tile))
        for tile in range(projection.count)) + (
        TileTask(2, 0, (TileAccess(arguments[2], IntervalSet(tuple(
            ((row * 1536 + 1152) * 2, (row + 1) * 1536 * 2) for row in range(256))), False),)),)
    _assign_ready_gemms([normalizer, projection, head], tasks, 10, 256)
    assert projection.worker_lanes is not None
    assert projection.worker_lanes[8:] == [0, 4]  # Both last-column lanes for M group zero.
    assert sorted(lane for lane in projection.worker_lanes if lane < 8) == list(range(8))


def _module(device="meta", rows=256):
    inputs = [
        torch.randn(rows, 128, device=device, dtype=torch.bfloat16) * 0.05,
        torch.randn(768, 128, device=device, dtype=torch.bfloat16) * 0.05,
        torch.randn(128, 384, device=device, dtype=torch.bfloat16) * 0.05,
    ]
    module = fx_region_to_module(
        discover_fx_graph(make_fx(_mlp)(*inputs).graph), name="tile_mlp"
    )
    return module, inputs


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_tile_codegen_removes_grid_rendezvous_and_retains_full_store_drain(mode):
    module, _ = _module()
    kernel = module.body.ops[0]
    external = len(kernel.signature)
    for op in list(find_ops_of_type(module, AllocOp)):
        if isinstance(op.result().type, GmemTensorType):
            kernel.body.ops.remove(op)
            kernel.body.append_argument(op.result())
    module = SchedulePersistentGraph(
        blocks=4, external_arguments=external, tile_scheduler=mode
    ).run(module)
    kernel = module.body.ops[0]
    assert kernel.attrs["tile_readiness_events"] > 0
    assert not any(op.scope == "grid" for op in find_ops_of_type(module, BarrierOp))
    syncs = find_ops_of_type(module, TileSyncOp)
    assert any(op.attrs["action"] == "wait" for op in syncs)
    assert any(op.attrs["action"] == "publish" for op in syncs)
    if mode == "static":
        from ayla.gpu_dialect import ForOp
        # Readiness belongs before the output loop; it must not reappear at
        # each tile or inside the K pipeline.
        assert all(op.attrs["action"] != "wait" for loop in find_ops_of_type(module, ForOp)
                   for op in find_ops_of_type(loop, TileSyncOp))
    verify_gpu_dialect(module)
    # Includes the new op in the textual IR protocol.
    for op in find_ops_of_type(module, TileSyncOp):
        assert op.attrs["action"] in ("wait", "publish")
    ptx = make_ptx_kernel(module).ptx
    assert "cp.async.bulk.wait_group 0;" in ptx
    assert "ld.acquire.gpu.global.u32" in ptx
    assert "red.release.gpu.global.add.u32" in ptx
    assert "atom.acq_rel.gpu.global.add.u32" not in ptx
    assert "global.inc.u32" not in ptx
    assert ".global .align 4 .u32 ayla_tile_tasks" in ptx
    assert ("atom.relaxed.gpu.global.add.u32" in ptx) == (mode == "dynamic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode,tile_order",
    [("static", "source"), ("dynamic", "source"), ("dynamic", "memory")],
)
def test_tile_mlp_replay_resets_counters_with_changing_inputs(
    mode, tile_order, monkeypatch
):
    original = forward.SchedulePersistentGraph

    def schedule(**kwargs):
        kwargs["blocks"] = 4
        return original(**kwargs)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    torch.manual_seed(923)
    module, inputs = _module("cuda")
    output = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    plan = forward.compile_forward_module(
        clone_op(module, {}), tile_scheduler=mode, tile_order=tile_order
    )
    runner = plan()
    runner(*inputs, output)
    torch.testing.assert_close(output, _mlp(*inputs), atol=0.001, rtol=0.015)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*inputs, output)
    for delta in (0.0, 0.01, -0.02):
        inputs[0].add_(delta)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, _mlp(*inputs), atol=0.001, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_single_worker_static_orders_payload_without_global_readiness(monkeypatch):
    original = forward.SchedulePersistentGraph

    def schedule(**kwargs):
        kwargs["blocks"] = 1
        return original(**kwargs)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    torch.manual_seed(932)
    module, inputs = _module("cuda")
    output = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    plan = forward.compile_forward_module(clone_op(module, {}), tile_scheduler="static")
    ptx = plan.routes[0].kernel.ptx
    assert "ld.acquire.gpu.global.u32" not in ptx
    assert "red.release.gpu.global.add.u32" not in ptx
    assert "fence.proxy.async.global" in ptx
    runner = plan()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*inputs, output)
    for delta in (0.0, 0.01, -0.02):
        inputs[0].add_(delta)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, _mlp(*inputs), atol=0.001, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_projection_wave_acquires_once_outside_down_k_loop(monkeypatch):
    original_schedule = forward.SchedulePersistentGraph
    original_emit = forward.make_ptx_kernel
    captured = []
    emitted = []

    def schedule(**kwargs):
        kwargs["blocks"] = 3
        return original_schedule(**kwargs)

    def emit(module):
        captured.append(module.body.ops[0].attrs.copy())
        emitted.append(clone_op(module, {}))
        return original_emit(module)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    monkeypatch.setattr(forward, "make_ptx_kernel", emit)
    torch.manual_seed(943)
    module, inputs = _module("cuda")
    output = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    plan = forward.compile_forward_module(module, tile_scheduler="static")
    [attrs] = captured
    assert "tile_streaming_pairs" not in attrs
    assert "static final producer tile flags" not in plan.routes[0].kernel.ptx
    from ayla.gpu_dialect import ForOp
    down = next(branch for branch in find_ops_of_type(emitted[0], IfOp)
                if branch.attrs.get("tile_root") == 1)
    waits = [op for op in find_ops_of_type(emitted[0], TileSyncOp)
             if op.attrs.get("action") == "wait" and op.attrs.get("readiness_root") == 1]
    assert len(waits) == 1
    assert not find_ops_of_type(down, TileSyncOp)
    assert not any(find_ops_of_type(loop, TileSyncOp)
                   for loop in find_ops_of_type(down, ForOp))
    runner = plan()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*inputs, output)
    for delta in (0.0, 0.01, -0.02):
        inputs[0].add_(delta)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, _mlp(*inputs), atol=0.001, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wide_normalization_uses_complete_input_acquire_on_replay(monkeypatch):
    def compute(x, w, residual, gamma1, gamma2):
        first = x @ w.T
        variance = first.float().square().mean(-1, keepdim=True)
        normalized = (first.float() * (variance + 1e-5).rsqrt() * (1 + gamma1.float())).bfloat16()
        mixed = (normalized + residual).bfloat16()
        variance = mixed.float().square().mean(-1, keepdim=True)
        return (mixed.float() * (variance + 1e-5).rsqrt() * (1 + gamma2.float())).bfloat16()

    original_schedule = forward.SchedulePersistentGraph
    original_emit = forward.make_ptx_kernel
    captured = []

    def schedule(**kwargs):
        kwargs["blocks"] = 9
        return original_schedule(**kwargs)

    def emit(module):
        captured.append(module.body.ops[0].attrs.copy())
        return original_emit(module)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    monkeypatch.setattr(forward, "make_ptx_kernel", emit)
    torch.manual_seed(947)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05
        for shape in [(256, 128), (4096, 128), (256, 4096), (4096,), (4096,)]
    ]
    fx = make_fx(compute)(*inputs)
    match = discover_fx_graph(fx.graph)
    placeholders = [node for node in fx.graph.nodes if node.op == "placeholder"]
    arguments = [inputs[placeholders.index(node)] for node in match.inputs]
    module = fx_region_to_module(match, name="wide_norm_prefix")
    output = torch.empty((256, 4096), device="cuda", dtype=torch.bfloat16)
    plan = forward.compile_forward_module(module, tile_scheduler="static")
    [attrs] = captured
    assert "tile_streaming_pairs" not in attrs
    assert "ayla_root_1_prefix_readiness" not in attrs["ptx_global_u32_tables"]
    assert attrs["tile_scalar_preamble_global_reads"][1] > 0
    assert "load deferred row readiness" not in plan.routes[0].kernel.ptx
    runner = plan()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*arguments, output)
    for delta in (0.0, 0.01, -0.02):
        inputs[0].add_(delta)
        inputs[3].add_(delta)
        inputs[4].sub_(delta)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, compute(*inputs), atol=0.025, rtol=0.025)


def test_partial_rows_and_pointwise_tail_compile_as_tile_tasks():
    def compute(x, w, v):
        first = x @ w.T
        mixed = (first + x).bfloat16()
        norm = mixed.float().square().mean(-1, keepdim=True)
        normalized = (mixed.float() * (norm + 1e-5).rsqrt()).bfloat16()
        return normalized @ v.T

    inputs = [
        torch.empty(257, 128, device="meta", dtype=torch.bfloat16),
        torch.empty(128, 128, device="meta", dtype=torch.bfloat16),
        torch.empty(128, 128, device="meta", dtype=torch.bfloat16),
    ]
    module = fx_region_to_module(
        discover_fx_graph(make_fx(compute)(*inputs).graph), name="tile_partial_norm"
    )
    kernel = module.body.ops[0]
    external = len(kernel.signature)
    for op in list(find_ops_of_type(module, AllocOp)):
        if isinstance(op.result().type, GmemTensorType):
            kernel.body.ops.remove(op)
            kernel.body.append_argument(op.result())
    module = SchedulePersistentGraph(
        blocks=4, external_arguments=external, tile_scheduler="dynamic"
    ).run(module)
    ptx = make_ptx_kernel(module).ptx
    assert "global.inc.u32" not in ptx
    assert "tile publish" in ptx


@pytest.mark.parametrize("blocks,rows", [(4, 256), (132, 512)])
def test_static_normalization_and_next_gemm_depend_only_on_their_rows(
    monkeypatch, blocks, rows
):
    import ayla.passes.schedule_tile_graph as tile_graph

    def compute(x, w, v):
        first = x @ w.T
        norm = first.float().square().mean(-1, keepdim=True)
        normalized = (first.float() * (norm + 1e-5).rsqrt()).bfloat16()
        return normalized @ v.T

    inputs = [
        torch.empty(rows, 128, device="meta", dtype=torch.bfloat16),
        torch.empty(512, 128, device="meta", dtype=torch.bfloat16),
        torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
    ]
    module = fx_region_to_module(
        discover_fx_graph(make_fx(compute)(*inputs).graph),
        name="static_row_dependencies",
    )
    kernel = module.body.ops[0]
    external = len(kernel.signature)
    for op in list(find_ops_of_type(module, AllocOp)):
        if isinstance(op.result().type, GmemTensorType):
            kernel.body.ops.remove(op)
            kernel.body.append_argument(op.result())
    graphs = []
    original = tile_graph.build_tile_dependency_graph

    def record(tasks):
        graph = original(tasks)
        graphs.append(graph)
        return graph

    monkeypatch.setattr(tile_graph, "build_tile_dependency_graph", record)
    module = SchedulePersistentGraph(
        blocks=blocks, external_arguments=external, tile_scheduler="static"
    ).run(module)
    assert module.body.ops[0].attrs["tile_static_lowering"] == "root_loops"
    if blocks == 132:
        norm = next(
            op
            for op in find_ops_of_type(module, IfOp)
            if op.attrs.get("tile_root") == 1
        )
        # Four rows per logical tile give 128 active tiles in this fixture;
        # the library's loop pads its capacity to 132 iterations.
        assert module.body.ops[0].attrs["tile_row_affinity"][1] == 1
        assert not find_ops_of_type(norm, TileSyncOp)
    [graph] = graphs
    roots = [
        {i for i, task in enumerate(graph.tasks) if task.root_id == root}
        for root in range(3)
    ]
    assert all(roots)
    # Each row reduction waits for every column tile of its own GEMM row
    # group. Other row groups must remain independent.
    for i in roots[1]:
        assert graph.predecessors[i]
        assert graph.predecessors[i] < roots[0]
    row_dependencies = {graph.predecessors[i] for i in roots[1]}
    assert len(row_dependencies) >= 2
    assert not frozenset.intersection(*row_dependencies)
    # The following GEMM acquires only the normalized rows it consumes.
    for i in roots[2]:
        assert graph.predecessors[i]
        assert graph.predecessors[i] < roots[1]
    assert not any(op.scope == "grid" for op in find_ops_of_type(module, BarrierOp))


def test_normalization_entry_covers_all_rows_and_releases_exact_frontiers(monkeypatch):
    import ayla.passes.schedule_tile_graph as tile_graph

    def compute(x, w):
        first = x @ w.T
        variance = first.float().square().mean(-1, keepdim=True)
        return (first.float() * (variance + 1e-5).rsqrt()).bfloat16()

    inputs = [torch.empty(shape, device="meta", dtype=torch.bfloat16)
              for shape in [(2048, 128), (4096, 128)]]
    module = fx_region_to_module(discover_fx_graph(make_fx(compute)(*inputs).graph),
                                 name="norm_complete_rows")
    kernel = module.body.ops[0]
    external = len(kernel.signature)
    for op in list(find_ops_of_type(module, AllocOp)):
        if isinstance(op.result().type, GmemTensorType):
            kernel.body.ops.remove(op)
            kernel.body.append_argument(op.result())
    captured = []
    original = tile_graph.ScheduleTileGraph._static_root_loops

    def capture(kernel, roots, plan, blocks):
        captured.append((kernel, roots, plan, blocks))
        original(kernel, roots, plan, blocks)

    monkeypatch.setattr(tile_graph.ScheduleTileGraph, "_static_root_loops", staticmethod(capture))
    SchedulePersistentGraph(blocks=132, external_arguments=external,
                            tile_scheduler="static").run(module)
    [(kernel, roots, plan, blocks)] = captured
    analyzer = tile_graph.TileDependencyAnalyzer(threads=256)
    tasks = tuple(tile_graph.TileTask(root, tile, analyzer.accesses(r.body, r.tile, tile))
                  for root, r in enumerate(roots) for tile in range(r.count))
    graph = tile_graph.build_tile_dependency_graph(tasks)
    positions = {task: position for position, task in enumerate(plan.order)}
    assert "tile_streaming_pairs" not in kernel.attrs
    ranges = kernel.attrs["tile_row_ranges"][1]
    lengths = [stop - first for first, stop in ranges]
    assert max(lengths) <= -(-sum(lengths) // blocks) + 1
    for worker, (first, stop) in enumerate(ranges):
        covered = {i for i, task in enumerate(tasks)
                   if task.root_id == 0 and task.tile_id % roots[0].grid == worker}
        for row in range(first, stop):
            task = roots[0].count + row
            record = plan.records[positions[task]]
            if record[2]:
                assert row == first  # one complete-row frontier per chunk
                covered.update(plan.events[record[2] - 1].producers)
            assert graph.predecessors[task] <= covered
    tables = kernel.attrs["ptx_global_u32_tables"]
    publications = tables["ayla_tile_publications"]
    for tile, (offset, length) in enumerate(zip(
            tables.get("ayla_root_0_frontier_publications", [])[::2],
            tables.get("ayla_root_0_frontier_publications", [])[1::2], strict=True)):
        worker = tile % roots[0].grid
        for event in publications[offset:offset + length]:
            contributors = [tasks[t].tile_id for t in plan.events[event - 1].producers
                            if tasks[t].root_id == 0 and tasks[t].tile_id % roots[0].grid == worker]
            assert contributors and max(contributors) == tile


@pytest.mark.parametrize("rows,row_tile,grid,blocks", [
    (2048, 128, 128, 132), (2048, 256, 128, 132),
    (1024, 128, 64, 68), (256, 64, 8, 9),
])
def test_normalization_row_budget_is_independent_of_producer_m_tile(rows, row_tile, grid, blocks):
    from ayla.passes.schedule_tile_graph import _normalization_rows
    ranges = _normalization_rows(rows, row_tile, grid, blocks)
    assert len(ranges) == blocks
    assert sorted(row for first, stop in ranges for row in range(first, stop)) == list(range(rows))
    budget = -(-rows // blocks)
    assert max(stop - first for first, stop in ranges[grid:]) <= budget
    # Keeping each chunk inside one row group can require more work than
    # unconstrained division. Check the optimal feasible maximum, including
    # the small-grid case where 64-row groups require 32-row chunks.
    prefix = ranges[-1][1]
    sizes = [row_tile - prefix, *([row_tile] * (rows // row_tile - 1))]
    optimum = next(limit for limit in range(1, row_tile + 1)
                   if sum(-(-size // limit) for size in sizes) <= grid)
    assert max(stop - first for first, stop in ranges[:grid]) == optimum
    for first, stop in ranges:
        assert first < stop and first // row_tile == (stop - 1) // row_tile


def test_idle_normalization_completes_input_tiles_with_bounded_extra_work():
    from ayla.passes.schedule_tile_graph import _normalization_rows
    for needed, prefix in ((128, 128), (256, 64)):
        ranges = _normalization_rows(2048, 256, 128, 132, ready_rows=needed)
        assert ranges[-1][1] == prefix
        assert max(stop - first for first, stop in ranges[128:]) <= 32
        assert sorted(row for first, stop in ranges for row in range(first, stop)) == list(range(2048))


def test_ready_head_reassignment_caps_idle_work_and_covers_every_row():
    from ayla.analysis.tile_dependency import TileTask, TileDependencyGraph
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import IndexType
    from ayla.passes.schedule_tile_graph import _TileRoot, _assign_early_scalar_rows
    producer = _TileRoot([], Block(), Value(None, IndexType()), 8, 4,
                         worker_lanes=[4, 4, 2, 3, 0, 1])
    consumer = _TileRoot([], Block(), Value(None, IndexType()), 24, 6)
    tasks = tuple(TileTask(0, tile, ()) for tile in range(8)) + tuple(TileTask(1, row, ()) for row in range(24))
    deps = (frozenset(),) * 8 + tuple(frozenset({0, 4}) if row < 20 else frozenset({2, 6})
                                     for row in range(24))
    _assign_early_scalar_rows([producer, consumer], tasks, TileDependencyGraph(tasks, deps), 6)
    ranges = consumer.row_ranges
    assert ranges is not None
    assert max(stop - first for first, stop in ranges[4:]) <= 8
    assert sorted(row for first, stop in ranges for row in range(first, stop)) == list(range(24))
    assert ranges[4:] == [(0, 8), (8, 16)]


@pytest.mark.parametrize("preceding_k", [64, 512])
def test_idle_gemm_workers_complete_a_row_only_with_sufficient_work_budget(preceding_k):
    from ayla.analysis.interval_set import IntervalSet
    from ayla.analysis.tile_dependency import TileAccess, TileTask, TileDependencyAnalyzer
    from ayla.base.ir import Block, Value
    from ayla.gpu_dialect import BF16Type, CollectiveAxis, CollectiveLevel, IndexType, KernelOp, MatmulOp, ModuleOp
    from ayla.passes.schedule_tile_graph import _TileRoot, _extract_root, _assign_ready_gemms, _row_range
    from ayla.pipelines.tile_matmul_pipeline import _schedule_matmul
    args = [Value(name, GmemTensorType.build(shape, BF16Type())) for name, shape in
            [("x", (512, 256)), ("w", (3072, 256)), ("y", (512, 3072))]]
    module = ModuleOp.build(Block(ops=[KernelOp.build("ready_full_row", Block(arguments=args,
        ops=[MatmulOp.build(*args, collective={axis: CollectiveLevel.GRID for axis in CollectiveAxis}, attrs={"label": "mm"})]))]))
    module = _schedule_matmul(module, output_m_tile=64, output_n_tile=128, warpgroup_m_tile=64,
                              warpgroup_n_tile=64, output_grid_size=64, output_grid_fast_axis="n",
                              output_tma_inner_size=128)
    gemm = _extract_root(module.body.ops[0].body.ops, 64)
    gemm.n_fast = True
    raw = Value("raw", GmemTensorType.build((512, preceding_k), BF16Type()))
    weight = Value("previous_weight", GmemTensorType.build((256, preceding_k), BF16Type()))
    previous = _TileRoot([], Block(ops=[MatmulOp.build(raw, weight, args[0], collective={})]),
                         Value(None, IndexType()), 128, 64)
    normalizer = _TileRoot([], Block(), Value(None, IndexType()), 512, 68,
        row_ranges=[(64 + first, 64 + stop) for worker in range(64)
                    for first, stop in [_row_range(448, 64, worker)]]
        + [_row_range(64, 4, worker) for worker in range(4)])
    tasks = (TileTask(0, 0, (TileAccess(args[0], IntervalSet.build([(0, 64 * 256 * 2)]), True),
                            TileAccess(raw, IntervalSet.build([(0, 64 * preceding_k * 2)]), False))),)
    tasks += tuple(TileTask(1, row, (TileAccess(args[0], IntervalSet.build([(row * 512, (row + 1) * 512)]), True),))
                   for row in range(512))
    tasks += tuple(TileTask(2, tile, TileDependencyAnalyzer(threads=256).accesses(gemm.body, gemm.tile, tile))
                   for tile in range(gemm.count))
    _assign_ready_gemms([previous, normalizer, gemm], tasks, 68, 256)
    if preceding_k == 64:
        assert gemm.row_ranges is None
    else:
        assert gemm.row_ranges is not None
        assert gemm.row_ranges[64:] == [(0, 6), (6, 12), (12, 18), (18, 24)]
        assert max(stop - first for first, stop in gemm.row_ranges[:64]) == 3
        assert sorted(tile for first, stop in gemm.row_ranges for tile in range(first, stop)) == list(range(gemm.count))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_residual_dependency_reduction_with_changing_inputs(monkeypatch):
    def compute(x, w, v):
        residual = x @ w.T
        variance = residual.float().square().mean(-1, keepdim=True)
        normalized = (residual.float() * (variance + 1e-5).rsqrt()).bfloat16()
        projected = normalized @ v.T
        mixed = (projected + residual).bfloat16()
        variance = mixed.float().square().mean(-1, keepdim=True)
        return (mixed.float() * (variance + 1e-5).rsqrt()).bfloat16()

    original = forward.SchedulePersistentGraph

    def schedule(**kwargs):
        kwargs["blocks"] = 4
        return original(**kwargs)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    torch.manual_seed(937)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05
        for shape in [(256, 128), (512, 128), (512, 512)]
    ]
    module = fx_region_to_module(
        discover_fx_graph(make_fx(compute)(*inputs).graph), name="static_residual"
    )
    output = torch.empty((256, 512), device="cuda", dtype=torch.bfloat16)
    runner = forward.compile_forward_module(module, tile_scheduler="static")()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*inputs, output)
    for delta in (0.0, 0.01, -0.02):
        inputs[0].add_(delta)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, compute(*inputs), atol=0.015, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_rows_normalization_and_short_phase_grids_on_cuda(monkeypatch):
    from dataclasses import replace
    import ayla.passes.schedule_persistent_graph as persistent

    original_schedule = forward.SchedulePersistentGraph
    original_select = persistent.select_persistent_gemm_schedule

    def schedule(**kwargs):
        kwargs["blocks"] = 4
        return original_schedule(**kwargs)

    def select(*args, **kwargs):
        seed = original_select(*args, **kwargs)
        return replace(
            seed, builder_config={**seed.builder_config, "output_grid_size": 2}
        )

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    monkeypatch.setattr(persistent, "select_persistent_gemm_schedule", select)

    def compute(x, w, v):
        first = x @ w.T
        mixed = (first + x).bfloat16()
        norm = mixed.float().square().mean(-1, keepdim=True)
        normalized = (mixed.float() * (norm + 1e-5).rsqrt()).bfloat16()
        return normalized @ v.T

    inputs = [
        torch.randn(257, 128, device="cuda", dtype=torch.bfloat16) * 0.05,
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.05,
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.05,
    ]
    module = fx_region_to_module(
        discover_fx_graph(make_fx(compute)(*inputs).graph), name="tile_partial_gpu"
    )
    output = torch.empty_like(inputs[0])
    runner = forward.compile_forward_module(module, tile_scheduler="dynamic")()
    runner(*inputs, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, compute(*inputs), atol=0.015, rtol=0.015)


def test_tile_sync_ir_roundtrip_and_invalid_scheduler():
    from ayla.gpu_dialect import gpu_dialect

    module = gpu_dialect.parse_module("""module {
      kernel @tile_sync_protocol(%sync: gmem<64xf32>, %event: index, %count: index)
             grid_dim=[4, 1, 1] block_dim=[256, 1, 1]
             attrs={cooperative: True, tile_scheduler: "static"} {
        tile_sync(%sync: gmem<64xf32>, %event: index, %count: index) attrs={action: "wait"}
        tile_sync(%sync: gmem<64xf32>, %event: index, %count: index) attrs={action: "publish"}
      }
    }""")
    verify_gpu_dialect(gpu_dialect.parse_module(str(module)))
    with pytest.raises(ValueError, match="tile_scheduler"):
        SchedulePersistentGraph(blocks=4, tile_scheduler="invalid")
    with pytest.raises(ValueError, match="resident"):
        forward.compile_forward_module(
            _module()[0], reference=True, tile_scheduler="static"
        )


def test_reused_readiness_scratch_resets_once_per_sequence(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import ayla.jit_compiler as jit
    from ayla.gpu_dialect import F32Type
    from ayla.ptx_codegen import PtxKernel

    typ = GmemTensorType.build([1], F32Type())
    with pytest.raises(ValueError, match="zero initialization"):
        jit.ScratchBuffer(typ, reset_each_launch=True)
    scratch = jit.ScratchBuffer(typ, jit.ScratchInit.ZERO, reset_each_launch=True)
    zeros = []
    observed = []

    def memset(buffer):
        zeros.append(buffer)
        buffer.value = 0

    def launch(source, buffer):
        observed.append(buffer.value)
        buffer.value += 1

    monkeypatch.setattr(jit, "make_ptx_launcher", lambda kernel: launch)
    monkeypatch.setattr(jit, "memset_zero", memset)
    monkeypatch.setattr(jit, "current_capture_id", lambda: 0)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: SimpleNamespace(value=None))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *a: SimpleNamespace(cuda_stream=7)
    )
    routes = [
        jit.PtxRoutePlan(
            PtxKernel(
                ptx="",
                name=name,
                signature=[typ, typ],
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
            ),
            [0, scratch],
        )
        for name in ("producer", "consumer")
    ]
    run = jit.PtxSequencePlan(routes, reuse_scratch=True)()
    source = SimpleNamespace(device=torch.device("cuda:0"))
    run(source)
    run(source)
    assert observed == [0, 1, 0, 1]
    assert len(zeros) == 2 and zeros[0] is zeros[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_single_output_wave_retains_entire_k_reductions(mode, monkeypatch):
    # Folding removes a one-trip output loop. The remaining top-level gate/up
    # K loops must remain reductions inside the tile, never become tasks.
    original = forward.SchedulePersistentGraph

    def schedule(**kwargs):
        kwargs["blocks"] = 2
        return original(**kwargs)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    torch.manual_seed(592)
    inputs = [
        torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16) * 0.05,
        torch.randn(768, 1024, device="cuda", dtype=torch.bfloat16) * 0.05,
        torch.randn(128, 384, device="cuda", dtype=torch.bfloat16) * 0.05,
    ]
    module = fx_region_to_module(
        discover_fx_graph(make_fx(_mlp)(*inputs).graph), name="single_tile_wave"
    )
    output = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    runner = forward.compile_forward_module(module, tile_scheduler=mode)()
    runner(*inputs, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, _mlp(*inputs), atol=0.0005, rtol=0.015)


@pytest.mark.parametrize("rows", [512, 2048])
def test_balanced_row_ownership_keeps_every_resident_lane_busy(rows):
    from ayla.passes.schedule_tile_graph import _row_owner, _row_range

    blocks = 132
    ranges = [_row_range(rows, blocks, worker) for worker in range(blocks)]
    assert all(stop > first for first, stop in ranges)
    assert (
        max(stop - first for first, stop in ranges)
        - min(stop - first for first, stop in ranges)
        <= 1
    )
    assert [tile for first, stop in ranges for tile in range(first, stop)] == list(
        range(rows)
    )
    assert all(
        _row_owner(tile, rows, blocks) == worker
        for worker, (first, stop) in enumerate(ranges)
        for tile in range(first, stop)
    )


@pytest.mark.parametrize("row_batch", [1, 2, 4])
def test_idle_ctas_normalize_first_completed_row_group(monkeypatch, row_batch):
    from dataclasses import replace
    import ayla.passes.schedule_persistent_graph as persistent
    import ayla.passes.schedule_tile_graph as tile_graph
    from ayla.analysis.tile_dependency import TileDependencyAnalyzer
    from ayla.forward_schedule_table import STATIC_NORMALIZATION_ROW_THREADS
    monkeypatch.setitem(STATIC_NORMALIZATION_ROW_THREADS, (68, 256, 1024, 4096), 256 // row_batch)

    def compute(x, w):
        first = x @ w.T
        return (first.float() * (first.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16()

    original_select = persistent.select_persistent_gemm_schedule

    def select(*args, **kwargs):
        seed = original_select(*args, **kwargs)
        return replace(seed, builder_config={**seed.builder_config, "output_grid_size": 64})

    monkeypatch.setattr(persistent, "select_persistent_gemm_schedule", select)
    captured = []
    original_order = tile_graph._order_normalization_producers

    def order(roots, tasks, blocks):
        result = original_order(roots, tasks, blocks)
        captured.append((roots, result))
        return result

    monkeypatch.setattr(tile_graph, "_order_normalization_producers", order)
    inputs = [torch.empty(shape, device="meta", dtype=torch.bfloat16)
              for shape in [(1024, 128), (4096, 128)]]
    module = fx_region_to_module(discover_fx_graph(make_fx(compute)(*inputs).graph), name="early_norm")
    kernel = module.body.ops[0]
    external = len(kernel.signature)
    for op in list(find_ops_of_type(module, AllocOp)):
        if isinstance(op.result().type, GmemTensorType):
            kernel.body.ops.remove(op)
            kernel.body.append_argument(op.result())
    scheduled = SchedulePersistentGraph(blocks=68, external_arguments=external, tile_scheduler="static").run(module)
    [(roots, tasks)] = captured
    producer, consumer = roots
    assert producer.n_fast and consumer.row_ranges is not None
    ranges = consumer.row_ranges
    assert len(ranges) == 68 and all(stop > first for first, stop in ranges)
    assert consumer.row_batch == row_batch
    assert sorted(row for first, stop in ranges for row in range(first, stop)) == list(range(1024 // row_batch))
    first_group_end = ranges[-1][1]
    assert ranges[64][0] == 0 and all(stop <= first_group_end for first, stop in ranges[64:])
    for task in tasks:
        if task.root_id == 0 and task.tile_id in (0, 1, 63, 64, producer.count - 1):
            assert TileDependencyAnalyzer().accesses(producer.body, producer.tile, task.tile_id) == task.accesses
    graph = tile_graph.build_tile_dependency_graph(tasks)
    for task, deps in zip(graph.tasks, graph.predecessors):
        if task.root_id == 1 and task.tile_id < first_group_end:
            assert deps and all(graph.tasks[p].root_id == 0 and graph.tasks[p].tile_id < producer.grid for p in deps)
    kernel = scheduled.body.ops[0]
    assert kernel.attrs["tile_n_fast_roots"] == [0]
    assert "tile_streaming_pairs" not in kernel.attrs
    verify_gpu_dialect(scheduled)
    assert "load static row bounds" in make_ptx_kernel(scheduled).ptx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("blocks,active,rows,following_gemm,row_batch", [(9, 8, 256, False, 1), (68, 64, 1024, False, 1),
                                                              (68, 64, 1024, True, 1), (68, 64, 1024, False, 2),
                                                              (68, 64, 1024, False, 4)])
def test_phase_grid_is_preserved_with_idle_resident_lanes(monkeypatch, blocks, active, rows, following_gemm, row_batch):
    from dataclasses import replace
    import ayla.passes.schedule_persistent_graph as persistent
    from ayla.forward_schedule_table import STATIC_NORMALIZATION_ROW_THREADS
    monkeypatch.setitem(STATIC_NORMALIZATION_ROW_THREADS, (blocks, 256, rows, 4096), 256 // row_batch)

    def normalize(x, w):
        first = x @ w.T
        variance = first.float().square().mean(-1, keepdim=True)
        return (first.float() * (variance + 1e-5).rsqrt()).bfloat16()

    if following_gemm:
        def compute(x, w, next_weight):
            last = normalize(x, w) @ next_weight.T
            return (last.float() * (last.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16()
    else:
        compute = normalize

    original_schedule = forward.SchedulePersistentGraph
    original_select = persistent.select_persistent_gemm_schedule
    original_emit = forward.make_ptx_kernel
    captured = []

    def schedule(**kwargs):
        kwargs["blocks"] = blocks
        return original_schedule(**kwargs)

    def select(*args, **kwargs):
        seed = original_select(*args, **kwargs)
        assert seed is not None and seed.builder_config is not None
        return replace(
            seed, builder_config={**seed.builder_config, "output_grid_size": active}
        )

    def emit(module):
        captured.append(module.body.ops[0].attrs.copy())
        return original_emit(module)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    monkeypatch.setattr(persistent, "select_persistent_gemm_schedule", select)
    monkeypatch.setattr(forward, "make_ptx_kernel", emit)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05
        for shape in [(rows, 128), (4096, 128)]
    ]
    if following_gemm:
        inputs.append(torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.005)
    module = fx_region_to_module(
        discover_fx_graph(make_fx(compute)(*inputs).graph), name="subgrid_norm"
    )
    resident_module = clone_op(module, {}) if following_gemm else None
    output = torch.empty((rows, 4096), device="cuda", dtype=torch.bfloat16)
    plan = forward.compile_forward_module(module, tile_scheduler="static")
    [attrs] = captured
    assert attrs["tile_root_grids"][0] == active
    if active == 64:
        assert 0 in attrs["tile_n_fast_roots"]
        assert "tile_streaming_pairs" not in attrs
        assert attrs["tile_row_ranges"][1][active][0] == 0
    else:
        assert "tile_streaming_pairs" not in attrs
    assert attrs["tile_row_ranges"][1][active][0] < attrs["tile_row_ranges"][1][active][1]
    if following_gemm:
        assert attrs["tile_root_grids"][2] == active
        lanes = attrs["tile_worker_lanes"].get(2, [*range(active), *([active] * (blocks - active))])
        assert sorted(lane for lane in lanes if lane < active) == list(range(active))
        assert len(lanes) == blocks
        # Lane transfers require an entire GEMM input tile to be ready;
        # balanced idle-lane row chunks can be smaller than that tile.
    assert "static root active worker and tile" in plan.routes[0].kernel.ptx
    runner = plan()
    resident = forward.compile_forward_module(resident_module)() if resident_module is not None else None
    resident_output = torch.empty_like(output) if resident is not None else None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*inputs, output)
    for delta in (0.0, 0.01, -0.02):
        inputs[0].add_(delta)
        graph.replay()
        if resident is not None:
            resident(*inputs, resident_output)
        torch.cuda.synchronize()
        if resident_output is not None:
            torch.testing.assert_close(output, resident_output, atol=0, rtol=0)
        # Two BF16 GEMMs/norms accumulate more rounding error near zero than
        # the single-stage fixture; exact resident agreement checks scheduling.
        torch.testing.assert_close(output, compute(*inputs), atol=0.002 if following_gemm else 0.001, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_head_norm_retains_subwarp_packing_and_replay(monkeypatch):
    import ayla.passes.schedule_persistent_graph as persistent

    def compute(x, weight, cosine, sine):
        head = (x @ weight.T)[:, :256].clone().reshape(-1, 128).float()
        normalized = (head * (head.square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16()
        lo, hi = (normalized.reshape(-1, 2, 128) * 3.87).split(64, dim=-1)
        return torch.cat((lo * cosine[:, None, :] - hi * sine[:, None, :],
                          hi * cosine[:, None, :] + lo * sine[:, None, :]), dim=-1)

    original_schedule, original_emit = forward.SchedulePersistentGraph, forward.make_ptx_kernel
    captured = []

    def schedule(**kwargs):
        kwargs["blocks"] = 4
        return original_schedule(**kwargs)

    def emit(module):
        captured.append(module.body.ops[0].attrs.copy())
        return original_emit(module)

    monkeypatch.setattr(forward, "SchedulePersistentGraph", schedule)
    monkeypatch.setattr(forward, "make_ptx_kernel", emit)
    monkeypatch.setitem(persistent.PERSISTENT_WARP_ROW_THREADS, (4, 256, 256, 128), 16)
    inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .05
              for shape in [(128, 128), (384, 128), (128, 64), (128, 64)]]
    module = fx_region_to_module(discover_fx_graph(make_fx(compute)(*inputs).graph), name="static_packed_norm")
    plan = forward.compile_forward_module(module, tile_scheduler="static")
    assert any(row["warp_row_threads"] == 16 for row in captured[0]["forward_row_schedules"])
    output = torch.empty_like(compute(*inputs))
    runner = plan()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner(*inputs, output)
    for delta in (0., .125, -.25):
        inputs[0].add_(delta)
        inputs[2].add_(.03125)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, compute(*inputs), atol=.02, rtol=.02)
