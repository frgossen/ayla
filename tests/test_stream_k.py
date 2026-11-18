"""Stream-K ownership, generated schedules, and scratch-safe replay."""
import math

import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.gemm_jit import _module
from ayla.gemm_collection.gemm import schedule_stream_k
from ayla.gpu_dialect import CollectiveAxis, MatmulOp, ReduceOp, gpu_dialect
from ayla.passes.stream_k import StreamKPlan, partition_stream_k
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.ptx_codegen import make_ptx_kernel


def test_partition_coverage_balance_and_slots():
    # Exhaust small grids to cover every remainder, boundary and oversubscription.
    for tiles in range(1, 10):
        for k_tiles in range(1, 12):
            for workers in range(1, 18):
                plan = StreamKPlan(tiles, k_tiles, workers)
                intervals = [plan.interval(w) for w in range(plan.workers)]
                assert intervals[0][0] == 0 and intervals[-1][1] == plan.total
                assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
                lengths = [end - start for start, end in intervals]
                assert min(lengths) >= 1 and max(lengths) - min(lengths) <= 1
                for tile in range(tiles):
                    first, last = tile * k_tiles, (tile + 1) * k_tiles
                    owners = [w for w, (s, e) in enumerate(intervals) if s < last and e > first]
                    assert owners == list(range(plan.owner(first), plan.owner(last - 1) + 1))
                    assert len(owners) <= plan.slots
                    if plan.uniform_contributors is not None:
                        assert len(owners) == plan.slots == plan.uniform_contributors


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_invalid_workers(workers):
    with pytest.raises(ValueError, match='positive integer'):
        StreamKPlan(2, 4, workers)


@pytest.mark.parametrize('operation,batch,m,n,k,workers', [
    ('mm', 1, 128, 128, 256, 3),     # work crosses output tiles
    ('addmm', 1, 65, 72, 320, 9),   # ragged rows/columns and uneven K ranges
    ('bmm', 2, 64, 64, 128, 8),     # oversubscribed workers, independent batches
])
def test_lowering(operation, batch, m, n, k, workers):
    module = _module(operation, 'bfloat16', batch, m, n, k, False)
    original = repr(module)
    raw = partition_stream_k(module, workers=workers, output_m_tile=64,
                             output_n_tile=64)
    assert repr(module) == original
    VerifyGpuDialect().run(raw.partials)
    VerifyGpuDialect().run(gpu_dialect.parse_module(repr(raw.partials)))
    plan = StreamKPlan(math.ceil(m / 64) * math.ceil(n / 64), k // 64, workers)
    kernel = raw.partials.body.ops[0]
    assert kernel.grid_dim == (plan.workers, 1, batch)
    assert raw.workspace_shape[0] == plan.slots
    assert raw.workspace_shape[-1] % 8 == 0
    for hero in find_ops_of_type(raw.partials, MatmulOp):
        assert hero.collective[CollectiveAxis.X].name == 'BLOCK'
        lo, hi = hero.attrs['reduction_range']
        assert 0 <= lo < hi <= k // 64
    assert not find_ops_of_type(raw.partials, ReduceOp)
    stages = schedule_stream_k(module, workers=workers, output_m_tile=64,
                              output_n_tile=64, operand_stages=2)
    for phase in (stages.partials, stages.reduction):
        ptx = make_ptx_kernel(phase)
        assert '.entry ' in ptx.ptx
    assert repr(module) == original


def test_many_complete_tiles_stay_in_rolled_loop():
    module = _module('mm', 'bfloat16', 1, 65536, 65536, 256, False)
    stages = partition_stream_k(module, workers=1, output_m_tile=64, output_n_tile=64)
    assert len(find_ops_of_type(stages.partials, MatmulOp)) == 1
    assert stages.workspace_shape[0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('operation,dtype,batch,m,n,k,workers,partial_slices,reduction_tile', [
    ('mm', 'bfloat16', 1, 128, 128, 256, 3, None, 1),
    ('addmm', 'float16', 1, 65, 72, 320, 9, None, 1),
    ('bmm', 'bfloat16', 2, 64, 64, 128, 8, None, 1),
    ('addmm', 'float16', 1, 65, 72, 256, 8, None, 1),  # uniform split with ragged tiles and bias
    ('addmm', 'float16', 1, 65, 72, 256, 8, 2, 1),  # sliced F32 TMA with ragged tiles
    ('addmm', 'float16', 1, 65, 72, 256, 8, 2, 16),  # grouped finalization, partial column tile
    ('bmm', 'bfloat16', 2, 64, 64, 128, 8, None, 8),  # grouped batched finalization
])
def test_numerical_and_graph_replay(operation, dtype, batch, m, n, k, workers, partial_slices, reduction_tile):
    from ayla.ptx_runtime import make_ptx_launcher

    stages = schedule_stream_k(_module(operation, dtype, batch, m, n, k, False),
                               workers=workers, output_m_tile=64,
                               output_n_tile=64, operand_stages=2,
                               partial_output_slices=partial_slices,
                               reduction_n_tile=reduction_tile)
    producer, final = [make_ptx_launcher(make_ptx_kernel(p))
                       for p in (stages.partials, stages.reduction)]
    tdtype = getattr(torch, dtype)
    prefix = (batch,) if operation == 'bmm' else ()
    a = torch.randn((*prefix, m, k), device='cuda', dtype=tdtype) * .1
    b = torch.randn((*prefix, n, k), device='cuda', dtype=tdtype) * .1
    out = torch.empty((*prefix, m, n), device='cuda', dtype=tdtype)
    bias = torch.randn(n, device='cuda', dtype=tdtype) * .1
    scratch = torch.full(stages.workspace_shape, float('nan'), device='cuda')
    args = ([bias] if operation == 'addmm' else []) + [a, b, out, scratch]

    def run():
        producer(*args)
        final(*args)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        a.normal_(std=.1)
        scratch.fill_(float('nan'))
        out.fill_(float('nan'))
        graph.replay()
        expected = a.float() @ b.float().transpose(-1, -2)
        if operation == 'addmm':
            expected += bias.float()
        torch.testing.assert_close(out, expected.to(tdtype), atol=.008, rtol=.02)


@pytest.mark.parametrize('slices', [0, -1, True, 1.5, 4, 5])
def test_partial_slices_preserve_lane_groups(slices):
    with pytest.raises(ValueError, match='8-column groups'):
        schedule_stream_k(_module('mm', 'bfloat16', 1, 256, 144, 256, False),
                          workers=2, output_m_tile=256, output_n_tile=144,
                          partial_output_slices=slices)


def test_managed_launch_plan_shares_scratch():
    from ayla.jit_compiler import ScratchBuffer

    stages = schedule_stream_k(_module('mm', 'bfloat16', 1, 64, 64, 128, False),
                               workers=2, output_m_tile=64, output_n_tile=64,
                               operand_stages=1)
    plan = stages.launch_plan()
    assert plan.reuse_scratch
    assert len(plan.routes) == 2
    scratch = plan.routes[0].bindings[-1]
    assert isinstance(scratch, ScratchBuffer)
    assert plan.routes[1].bindings[-1] is scratch
    assert scratch.tensor_type.static_shape == stages.workspace_shape
    assert plan.routes[0].bindings[:-1] == (0, 1, 2)


def test_jit_schedule_accepts_stream_k_and_builds_managed_phases():
    from ayla.gemm_jit import CONFIGS, _DEFAULTS, _seed, build_gemm_module
    from ayla.passes.tile_reduce import ReductionModules

    captured = {**_DEFAULTS, **CONFIGS[0], 'stream_k_workers': 3,
                'output_m_tile': 64, 'output_n_tile': 64,
                'warpgroup_m_tile': None, 'warpgroup_n_tile': 64,
                'output_cta_count': None, 'operand_stages': 2,
                'output_tiles_per_cta': 1, 'output_n_tiles_per_cta': 1,
                'cluster_m_ctas': 1, 'cluster_n_ctas': 1, 'multicast': False}
    seed = _seed(captured, 'mm', 128, 128, 256, (1, 128, 128, 256), exact=True)
    assert seed is not None and seed.stream_k_workers == 3
    phases = build_gemm_module('mm', 'bfloat16', 1, 128, 128, 256, seed)
    assert isinstance(phases, ReductionModules)
    for phase in (phases.partials, phases.reduction):
        assert '.entry ' in make_ptx_kernel(phase).ptx
    for bad in ({'stream_k_workers': 0}, {'split_k_parts': 2},
                {'transpose_output': True}, {'cluster_m_ctas': 2},
                {'output_cta_count': 4}):
        assert _seed({**captured, **bad}, 'mm', 128, 128, 256,
                     (1, 128, 128, 256), exact=True) is None


def test_barrier_and_shared_memory_resources_do_not_grow_with_segments():
    import re

    stages = schedule_stream_k(_module('mm', 'bfloat16', 1, 256, 256, 512, False),
                               workers=31, output_m_tile=64, output_n_tile=64,
                               operand_stages=2)
    ptx = make_ptx_kernel(stages.partials).ptx
    # Every specialized segment shares one compute/writeback arena; allocations
    # inside LowerToTmaMove also add small completion-event storage.
    allocations = [int(size) for size in re.findall(r'\.shared \.align \d+ \.b8 \w+\[(\d+)\]', ptx)]
    assert max(allocations) <= 64 * 1024
    assert sum(allocations) <= 64 * 1024
    assert 'wgmma.mma_async' in ptx


@pytest.mark.parametrize('initialization_threads', [32, 128, 256])
@pytest.mark.parametrize('fast_axis', ['m', 'n'])
@pytest.mark.parametrize('k,workers,operand_tile', [(128, 5, 64), (320, 9, 64), (384, 7, 128),
                                                  (256, 8, 64)])  # exact two contributors per tile
def test_generated_partials_write_every_logical_slot_once_on_cpu(k, workers, operand_tile, fast_axis, initialization_threads):
    """Execute the partition IR's indexing and stores, including unused slots.

    This validates generated addresses independently of the planner and GPU
    code generation; matrix contractions use NumPy as the CPU reference.
    """
    import numpy as np
    from ayla.gpu_dialect import (
        AllocOp, BlockIdOp, ConstantOp, ForOp, IfOp, IndexAddOp, IndexConstantOp,
        IndexDivOp, IndexEqOp, IndexLtOp, IndexModOp, IndexMulOp, MoveOp,
        ReshapeOp, ThreadIdOp, TileOp,
    )

    m, n = 65, 72
    module = _module('bmm', 'float16', 2, m, n, k, False)
    stages = partition_stream_k(module, workers=workers, output_m_tile=64,
                               output_n_tile=64, operand_k_tile=operand_tile,
                               output_grid_fast_axis=fast_axis,
                               initialization_threads=initialization_threads)
    kernel = stages.partials.body.ops[0]
    rng = np.random.default_rng(12)
    a = rng.normal(size=(2, m, k)).astype(np.float32)
    b = rng.normal(size=(2, n, k)).astype(np.float32)
    scratch = np.full(stages.workspace_shape, np.nan, dtype=np.float32)
    writes = np.zeros(stages.workspace_shape, dtype=np.int32)
    roots = dict(zip(kernel.body.arguments, (a, b, np.empty((2, m, n)), scratch), strict=True))
    env = {}
    hardware = {}
    operations = {IndexAddOp: lambda a, b: a + b, IndexMulOp: lambda a, b: a * b,
                  IndexDivOp: lambda a, b: a // b, IndexModOp: lambda a, b: a % b,
                  IndexLtOp: lambda a, b: a < b, IndexEqOp: lambda a, b: a == b}

    def index(value):
        if isinstance(value, int):
            return value
        if value in env:
            return env[value]
        op = value.definition
        if isinstance(op, IndexConstantOp):
            return op.value
        if isinstance(op, (BlockIdOp, ThreadIdOp)):
            return hardware[(type(op), op.axis)]
        return operations[type(op)](*(index(v) for v in op.operands))

    def address(value, coordinates, valid=True):
        if value in roots:
            return value, coordinates, valid
        op = value.definition
        if isinstance(op, TileOp):
            coordinates = [c + index(i) * s for c, i, s in
                           zip(coordinates, op.indices, value.type.static_shape, strict=True)]
            if op.maybe_partial:
                for c, size in zip(coordinates, op.arg.type.static_shape, strict=True):
                    valid = valid & (c >= 0) & (c < size)
        elif isinstance(op, ReshapeOp):
            flat = 0
            for c, size in zip(coordinates, value.type.static_shape, strict=True):
                flat = flat * size + c
            shape = op.arg.type.static_shape
            coordinates = [(flat // math.prod(shape[d + 1:])) % size
                           for d, size in enumerate(shape)]
        else:
            raise AssertionError(type(op))
        return address(op.arg, coordinates, valid)

    def load(value):
        coordinates = np.indices(value.type.static_shape)
        root, coords, valid = address(value, list(coordinates))
        safe = [np.where(valid, c, 0) for c in coords]
        return np.where(valid, roots[root][tuple(safe)], 0.)

    def store(value, data):
        root, coords, valid = address(value, list(np.indices(value.type.static_shape)))
        coords = tuple(np.broadcast_to(c, value.type.static_shape)[np.broadcast_to(valid, value.type.static_shape)] for c in coords)
        roots[root][coords] = np.broadcast_to(data, value.type.static_shape)[np.broadcast_to(valid, value.type.static_shape)]
        assert roots[root] is scratch
        writes[coords] += 1

    def execute(block):
        for op in block.ops:
            if isinstance(op, ForOp):
                for i in range(op.lb, op.ub, op.step):
                    env[op.iv] = i
                    execute(op.body)
            elif isinstance(op, IfOp):
                if index(op.condition):
                    execute(op.body)
            elif isinstance(op, MatmulOp):
                # All CTA threads jointly perform the contraction; execute it
                # once, while scalar initialization stores run on each lane.
                if hardware[(ThreadIdOp, CollectiveAxis.X)] == 0:
                    lo, hi = op.attrs['reduction_range']
                    left, right = load(op.lhs), load(op.rhs)
                    store(op.dst, left[:, lo * operand_tile:hi * operand_tile] @ right[:, lo * operand_tile:hi * operand_tile].T)
            elif isinstance(op, MoveOp):
                assert isinstance(op.src.definition, ConstantOp)
                store(op.dst, op.src.definition.value)
            elif isinstance(op, AllocOp):
                # The input BMM only allocates no tensors in its prologue.
                raise AssertionError('unexpected allocation')

    for batch in range(2):
        hardware[(BlockIdOp, CollectiveAxis.Z)] = batch
        for worker in range(kernel.grid_dim[0]):
            hardware[(BlockIdOp, CollectiveAxis.X)] = worker
            for lane in range(initialization_threads):
                hardware[(ThreadIdOp, CollectiveAxis.X)] = lane
                execute(kernel.body)
    assert np.all(writes[..., :n] == 1)
    np.testing.assert_allclose(scratch[..., :n].sum(0), a @ b.transpose(0, 2, 1),
                               atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize('fast_axis', ['m', 'n'])
def test_balanced_route_has_identical_ptx_and_no_workspace(fast_axis):
    from ayla.base.ir import clone_op
    from ayla.gpu_dialect import ModuleOp
    from ayla.gemm_collection.gemm import schedule_matmul
    from ayla.passes.tile_reduce import ReductionModules

    module = _module('mm', 'bfloat16', 1, 128, 256, 256, False)
    options = dict(output_m_tile=64, output_n_tile=64, operand_stages=2,
                   output_grid_fast_axis=fast_axis)
    direct = schedule_stream_k(module, workers=4, **options)
    assert isinstance(direct, ModuleOp)
    baseline = schedule_matmul(clone_op(module, {}), output_grid_size=4, **options)
    assert make_ptx_kernel(direct).ptx == make_ptx_kernel(baseline).ptx
    forced = schedule_stream_k(module, workers=4, direct_when_balanced=False, **options)
    assert isinstance(forced, ReductionModules)
    assert forced.workspace_shape[0] <= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_managed_launch_plan_graph_replay():
    """Exercise the public launcher, including repeated independent captures."""
    stages = schedule_stream_k(_module('mm', 'bfloat16', 1, 128, 128, 256, False),
                               workers=3, output_m_tile=64, output_n_tile=64,
                               operand_stages=2)
    launch = stages.launch_plan()()
    a = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16) * .1
    b = torch.randn_like(a) * .1
    outputs = [torch.empty(128, 128, device='cuda', dtype=a.dtype) for _ in range(2)]
    graphs = []
    for out in outputs:
        launch(a, b, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(a, b, out)
        graphs.append(graph)
    for _ in range(3):
        a.normal_(std=.1)
        expected = (a.float() @ b.float().T).bfloat16()
        for graph, out in zip(graphs, outputs, strict=True):
            out.fill_(float('nan'))
            graph.replay()
            torch.testing.assert_close(out, expected, atol=.008, rtol=.02)
