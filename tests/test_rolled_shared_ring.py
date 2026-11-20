"""Continuous dynamic banks, accumulator ownership and partial job tails."""

import math
import pytest

from ayla.base.ir import clone_op, find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    AllocOp, AsyncOp, AwaitOp, BarrierOp, BlockIdOp, ClusterBlockIdsOp, ForOp, IfOp,
    IndexAddOp, IndexConstantOp, IndexDivOp, IndexLtOp, IndexModOp,
    IndexMulOp, LockType, LockViewOp, MatmulOp, MoveOp, PersistentPipelineOp,
    TileOp, TmemTensorType, SmemTensorType, gpu_dialect, memory_effects, physical_layout,
    PredicateOp,
)
from ayla.passes.pipeline_rolled_shared_ring import LowerRolledSharedRing
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.ptx_codegen import make_ptx_kernel
from tests.testing_utils import assert_dominance, check_gemm


def build(**kwargs):
    cfg = dict(m=2049, n=520, k=384, dtype="f16", cta_group=2,
               output_n_tile=128, output_grid_size=4, output_grid_fast_axis="m",
               operand_stages=4, rolled_operand_pipeline=True,
               persistent_output_pipeline=True,
               output_store="stmatrix", output_slices=2,
               tmem_writeback_register_budget=64, allow_partial_tiles=True)
    cfg.update(kwargs)
    return blackwell_tcgen05_gemm_module(**cfg)


def check_protocol(mod, block_index, steps, queue_samples=None):
    values, pending, generations, banks = {}, {}, {}, {}
    counts = dict(load=0, mma=0, read=0, store=0)
    slot_owners, slot_released = {}, {}
    bank_readers, completion_ordinals = {}, {}
    completed_through = -1
    depth = next(x.result().type.attrs["count"] for x in find_ops_of_type(mod, AllocOp)
                 if x.attrs.get("label") == "operand_bank_ready")

    def coordinates(value):
        indices = [0, 0]
        while isinstance(value.definition, TileOp):
            tile = value.definition
            for i, index in enumerate(tile.indices):
                indices[i] += (index if isinstance(index, int) else values[index]) * value.type.sizes[i]
            value = tile.arg
        return value, indices

    def bank(value):
        root, indices = coordinates(value)
        layout = physical_layout(root.type)
        return (root, indices[1] // 64 if layout and layout[1] == "block_major"
                else indices[0] // (root.type.sizes[0] // depth))

    def identity(value):
        if isinstance(value.definition, LockViewOp):
            arena, index, phase = value.definition.operands
            return (arena, values[index]), values[phase]
        return (value, 0), None

    def observe(value):
        nonlocal completed_through
        key, phase = identity(value)
        assert pending.get(key), ("observed an unissued epoch", key)
        if phase is not None:
            assert phase == (generations[key] - 1) % 2, ("wrong phase", key, phase)
        pending[key] = False
        if value.type.kind == "tcgen05":
            completed_through = max(completed_through, completion_ordinals[key])

    def walk(block):
        for op in block.ops:
            if isinstance(op, IndexConstantOp):
                values[op.result()] = op.value
            elif isinstance(op, BlockIdOp):
                values[op.result()] = block_index
            elif isinstance(op, ClusterBlockIdsOp):
                values.update(zip(op.results, (block_index % 2, 0, 0)))
            elif isinstance(op, (IndexAddOp, IndexDivOp, IndexLtOp, IndexModOp, IndexMulOp)):
                a, b = values[op.lhs], values[op.rhs]
                values[op.result()] = {IndexAddOp: lambda: a+b, IndexDivOp: lambda: a//b,
                                       IndexLtOp: lambda: a<b, IndexModOp: lambda: a%b,
                                       IndexMulOp: lambda: a*b}[type(op)]()
            elif isinstance(op, PredicateOp):
                args = [values[x] for x in op.operands]
                values[op.result()] = {"and": lambda: all(args), "or": lambda: any(args),
                                      "not": lambda: not args[0]}[op.attrs['kind']]()
            elif isinstance(op, IfOp):
                if values[op.condition]:
                    walk(op.body)
            elif isinstance(op, ForOp):
                for i in range(op.lb, op.ub, op.step):
                    values[op.iv] = i
                    walk(op.body)
            elif isinstance(op, AsyncOp):
                kind = op.lock.type.kind
                if kind == "mbar":
                    for move in op.body.ops:
                        _, src = coordinates(move.src)
                        assert src[1] == counts["load"] % steps * 64
                        key = bank(move.dst)
                        assert key not in bank_readers or bank_readers[key] <= completed_through, "operand overwritten before MMA completion"
                        banks[key] = counts["load"]
                    counts["load"] += 1
                elif kind == "tcgen05":
                    for mm in op.body.ops:
                        assert isinstance(mm, MatmulOp)
                        assert banks[bank(mm.lhs)] == counts["mma"]
                        assert banks[bank(mm.rhs)] == counts["mma"]
                        bank_readers[bank(mm.lhs)] = counts["mma"]
                        bank_readers[bank(mm.rhs)] = counts["mma"]
                    assert op.body.ops[0].accumulate == (counts["mma"] % steps != 0)
                    _, dst = coordinates(op.body.ops[0].dst)
                    slot = dst[1] // 128
                    if counts["mma"] % steps == 0:
                        assert slot not in slot_owners or slot_released[slot], "TMEM overwritten before last read"
                        slot_owners[slot] = counts["mma"] // steps
                        slot_released[slot] = False
                    counts["mma"] += 1
                else:
                    counts["store"] += 1
                for lock in op.completion_locks:
                    key, _ = identity(lock)
                    if kind != "bulk":
                        assert not pending.get(key), ("overwrote an unobserved event", key)
                    pending[key] = True
                    generations[key] = generations.get(key, 0) + 1
                    if kind == "tcgen05":
                        completion_ordinals[key] = counts["mma"] - 1
            elif isinstance(op, AwaitOp):
                if op.lock.type.kind != "bulk":
                    observe(op.lock)
            elif isinstance(op, BarrierOp) and op.sync is not None:
                key, _ = identity(op.sync)
                if op.attrs.get("phase") == "arrive":
                    assert not pending.get(key), "overwrote a slot release"
                    pending[key] = True
                    slot_released[int(op.sync.definition.attrs["label"].rsplit("_", 1)[1])] = True
                elif op.attrs.get("phase") == "wait":
                    observe(op.sync)
            elif isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType):
                if queue_samples is not None:
                    queue_samples.append(counts["mma"] - completed_through - 1)
                _, src = coordinates(op.src)
                assert slot_owners[src[1] // 128] == counts["read"] // 2, "read the wrong accumulator generation"
                counts["read"] += 1

    walk(mod.body.ops[0].body)
    assert all(not outstanding or key[0].type.attrs.get("arrivals")
               or key[0].type.attrs.get("release_warps") or key[0].type.kind == "bulk"
               # An ordered final completion also covers older MMA accesses.
               or (key[0].type.kind == "tcgen05"
                   and completion_ordinals[key] <= completed_through)
               for key, outstanding in pending.items()), "undrained operand/frame event"
    return counts


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("k,depth", [(192, 2), (320, 4), (384, 4), (768, 7), (1024, 5),
                                     # Rings deeper than one job's K batches.
                                     (128, 2), (128, 4), (128, 7), (192, 5), (256, 6)])
def test_global_epochs_and_bank_contents_with_uneven_jobs(group, overlap, k, depth):
    mod = build(k=k, operand_stages=depth, cta_group=group,
                persistent_output_overlap=overlap)
    assert_dominance(mod.body)
    VerifyGpuDialect().run(mod)
    expanded = LowerRolledSharedRing().run(clone_op(mod, {}))
    assert_dominance(expanded.body)
    tiles = math.ceil(2049 / (128 * group)) * group * math.ceil(520 / 128)
    for block in range(4):
        jobs = len(range(block, tiles, 4))
        assert check_protocol(expanded, block, k//64) == dict(
            load=jobs*(k//64), mma=jobs*(k//64), read=jobs*2, store=jobs*2)


def test_dynamic_k_body_does_not_grow_with_depth_or_bank_supercycle():
    counts = []
    for depth in (4, 5, 7, 8):
        mod = build(k=1024, operand_stages=depth)
        assert len(find_ops_of_type(mod, PersistentPipelineOp)) == 1
        expanded = LowerRolledSharedRing().run(mod)
        loops = find_ops_of_type(expanded, ForOp)
        jobs = next(x for x in loops if x.attrs.get("label") == "shared_operand_output_ring")
        assert jobs.step == 2
        assert len([x for x in loops if x.attrs.get("label") == "shared_ring_rolled_k"]) == 2
        counts.append(len(find_ops_of_type(expanded, MatmulOp)))
    assert len(set(counts)) == 1


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("k,depth", [(128, 5), (384, 4)])
def test_mixed_one_and_two_job_tails_preserve_bank_and_frame_lifetimes(group, overlap, k, depth):
    mod = build(m=640, n=256, k=k, operand_stages=depth, cta_group=group,
                output_grid_size=8, persistent_output_overlap=overlap)
    VerifyGpuDialect().run(mod)
    expanded = LowerRolledSharedRing().run(clone_op(mod, {}))
    assert_dominance(expanded.body)
    tiles = math.ceil(640 / (128 * group)) * group * 2
    for block in range(8):
        jobs = len(range(block, tiles, 8))
        assert check_protocol(expanded, block, k // 64) == dict(
            load=jobs * (k // 64), mma=jobs * (k // 64), read=jobs * 2, store=jobs * 2)


def test_pingpong_groups_each_issue_loads_and_mmas_for_their_whole_job():
    from ayla.gpu_dialect import TmaMoveOp, execution_issuer_warp
    from ayla.passes.lower_to_tma_move import LowerToTmaMove

    mod = build(m=640, n=256, k=128, operand_stages=5, output_grid_size=8,
                persistent_output_overlap=True,
                persistent_output_job_regions=True, tmem_writeback_warps=4)
    expanded = LowerRolledSharedRing().run(LowerToTmaMove().run(mod))
    groups = {}
    for typ in (TmaMoveOp, MatmulOp):
        groups[typ] = {execution_issuer_warp(op) for op in find_ops_of_type(expanded, typ)
                       if typ is MatmulOp or isinstance(op.dst.type, SmemTensorType)
                       and op.attrs.get("tma")}
    assert groups[TmaMoveOp] == groups[MatmulOp] == {0, 4}


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("pingpong", [False, True])
def test_gpu_mixed_one_and_two_job_graph_tails(dtype, pingpong):
    import torch
    from ayla.ptx_runtime import assemble_ptx, make_ptx_launcher
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    mod = build(m=640, n=256, k=128, operand_stages=5, output_grid_size=8,
                dtype=dtype, persistent_output_overlap=True,
                persistent_output_job_regions=pingpong,
                tmem_writeback_warps=4)
    kernel = make_ptx_kernel(mod)
    launcher = make_ptx_launcher(kernel, image=assemble_ptx(kernel, "/usr/local/cuda-13.0/bin/ptxas"))
    check_gemm(launcher, 640, 256, 128, dtype, graph=True)


def test_predicate_composition_requires_predicate_operands():
    index = IndexConstantOp.build(0).result()
    pred = IndexLtOp.build(index, IndexConstantOp.build(1).result()).result()
    for kind, operands in (("and", (pred, index)), ("or", (index, pred)),
                           ("not", (index,))):
        with pytest.raises(AssertionError):
            PredicateOp.build(kind, *operands)


def test_completed_boundary_covers_both_slots_and_keeps_events_private():
    mod = build()
    assert repr(gpu_dialect.parse_module(repr(mod))) == repr(mod)
    [region] = find_ops_of_type(mod, PersistentPipelineOp)
    effects = memory_effects(region)
    assert not any(isinstance(e.value.type, LockType) for e in effects)
    assert any(e.kind == "W" and e.value is region.operands[0] for e in effects)


def test_unrelated_effects_and_inconsistent_depth_are_rejected():
    mod = build()
    [region] = find_ops_of_type(mod, PersistentPipelineOp)
    region.attrs["depth"] += 1
    with pytest.raises(ValueError, match="invalid compact persistent"):
        VerifyGpuDialect().run(mod)
    mod = build()
    [region] = find_ops_of_type(mod, PersistentPipelineOp)
    loop = region.body.ops[0]
    loop.body.ops.append(BarrierOp.build("block"))
    with pytest.raises(ValueError, match="unrelated effects"):
        VerifyGpuDialect().run(mod)


@pytest.mark.parametrize("group,k,depth,warps,nt,descriptors,buffers", [
    (1, 320, 4, 4, 128, False, 1), (2, 384, 4, 4, 128, False, 1),
    (2, 768, 7, 8, 128, False, 1), (2, 784, 5, 4, 192, True, 2),
    (2, 128, 4, 4, 128, True, 1), (1, 128, 3, 4, 128, False, 1), (2, 192, 5, 4, 192, True, 1)])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gpu_changed_input_graph_replays(group, k, depth, warps, nt, descriptors, buffers, dtype, queue=1):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    for overlap in (False, True):
        mod = build(k=k, operand_stages=depth, cta_group=group, dtype=dtype,
                    cyclic_operand_pipeline=queue,
                    tmem_writeback_warps=warps, persistent_output_overlap=overlap,
                    output_n_tile=nt, hoist_mma_descriptors=descriptors,
                    output_store_buffers=buffers, output_store_pending_groups=buffers - 1)
        check_gemm(make_ptx_launcher(make_ptx_kernel(mod)), 2049, 520, k, dtype, graph=True)
