"""Independent event generations and rolled-loop tails in the shared pipeline."""

import pytest

from ayla.base.ir import clone_op, find_ops_of_type, iter_ops
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    BlockIdOp,
    ForOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    LockType,
    MatmulOp,
    MoveOp,
    RegMemTensorType,
    TmemTensorType,
    gpu_dialect,
    root_buffers,
)
from ayla.analysis.sync import SyncAnalysis, BarrierScope
from ayla.passes.fuse_async import FuseAsync
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.ptx_codegen import make_ptx_kernel
from tests.testing_utils import assert_dominance, check_gemm


def build(**kwargs):
    config = dict(
        m=1152,
        n=520,
        k=384,
        dtype="f16",
        output_n_tile=128,
        output_grid_size=3,
        output_store="stmatrix",
        output_slices=2,
        output_layout="tcgen05_16x256b",
        operand_stages=2,
        cyclic_operand_pipeline=True,
        allow_partial_tiles=True,
        persistent_output_pipeline=True,

    )
    config.update(kwargs)
    return blackwell_tcgen05_gemm_module(**config)


def event_walk(module, block_index):
    """Execute CTA-uniform control flow, asserting one observation per epoch.

    This catches reused/missing awaits and tail issues without assuming the
    compiler's intended bank schedule. Hardware completion is immediate here;
    GPU checks separately exercise real asynchronous timing and data hazards.
    """
    values, pending, counts = {}, {}, {"load": 0, "mma": 0, "read": 0, "store": 0}
    bank_contents = {}

    def walk(block):
        for op in block.ops:
            if isinstance(op, IndexConstantOp):
                values[op.result()] = op.value
            elif isinstance(op, BlockIdOp):
                values[op.result()] = block_index
            elif isinstance(
                op, (IndexAddOp, IndexDivOp, IndexLtOp, IndexModOp, IndexMulOp)
            ):
                a, b = values[op.lhs], values[op.rhs]
                fn = {
                    IndexAddOp: lambda: a + b,
                    IndexDivOp: lambda: a // b,
                    IndexLtOp: lambda: a < b,
                    IndexModOp: lambda: a % b,
                    IndexMulOp: lambda: a * b,
                }[type(op)]
                values[op.result()] = fn()
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
                        for root in root_buffers(move.dst):
                            bank_contents[root] = op.attrs.get("label")
                elif kind == "tcgen05" and op.attrs.get("label", "").startswith("mma_issue_"):
                    expected = op.attrs["label"].replace("mma_issue_", "operand_issue_")
                    for mma in op.body.ops:
                        if isinstance(mma, MatmulOp):
                            for root in root_buffers(mma.lhs) | root_buffers(mma.rhs):
                                assert bank_contents.get(root) == expected, (op.attrs, bank_contents.get(root))
                counts[{"mbar": "load", "tcgen05": "mma", "bulk": "store"}[kind]] += 1
                for event in op.completion_locks:
                    if kind != "bulk":
                        assert not pending.get(event), (
                            "reissued unobserved event",
                            event,
                        )
                    pending[event] = True
            elif isinstance(op, AwaitOp):
                if op.lock.type.kind != "bulk":
                    assert pending.get(op.lock), ("await without new phase", op.lock)
                pending[op.lock] = False
            elif isinstance(op, BarrierOp) and op.sync is not None:
                if op.attrs.get("phase") == "arrive":
                    assert not pending.get(op.sync), (
                        "unconsumed slot release",
                        op.sync,
                    )
                    pending[op.sync] = True
                elif op.attrs.get("phase") == "wait":
                    assert pending.get(op.sync), (
                        "slot acquired before release",
                        op.sync,
                    )
                    pending[op.sync] = False
            elif isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType):
                counts["read"] += 1

    walk(module.body.ops[0].body)
    assert all(
        not outstanding or event.type.attrs.get("arrivals") or event.type.attrs.get("release_warps")
        for event, outstanding in pending.items()
    ), "undrained async event at exit"
    return counts


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "m,n,grid", [(256, 256, 2), (384, 384, 2), (1153, 520, 3), (1792, 512, 5)]
)
def test_event_epochs_and_complete_jobs_across_peeling_and_partial_cta_tails(
    overlap, m, n, grid
):
    mod = build(m=m, n=n, output_grid_size=grid, persistent_output_overlap=overlap)
    assert_dominance(mod.body)
    VerifyGpuDialect().run(mod)
    tiles = ((m + 127) // 128) * ((n + 127) // 128)
    for block in range(grid):
        counts = event_walk(mod, block)
        jobs = len(range(block, tiles, grid))
        assert counts == dict(
            load=jobs * 6, mma=jobs * 6, read=jobs * 4, store=jobs * 4
        )


def test_one_shared_loop_and_four_separate_resource_lifetimes():
    mod = build()
    [loop] = find_ops_of_type(mod, ForOp)
    assert loop.attrs["label"] == "shared_operand_output_ring"
    assert not any(op.attrs.get("participant_warps") for op in iter_ops(mod))
    for wait in find_ops_of_type(mod, AwaitOp):
        if wait.attrs.get("label") in (
            "operand_ready",
            "bank_release",
            "bank_terminal_drain",
        ):
            assert wait.waiter_warps == (0,)
        else:
            assert wait.waiter_warps is None
    events = [
        op
        for op in find_ops_of_type(mod, AllocOp)
        if isinstance(op.result().type, LockType)
    ]
    labels = [event.attrs["label"] for event in events]
    assert len([x for x in labels if x.startswith("bank_release_")]) == 2
    assert len([x for x in labels if x.startswith("accumulator_ready_")]) == 2
    assert len([x for x in labels if x.startswith("last_tmem_read_")]) == 2
    assert labels.count("output_source_release") == 1
    for issue in find_ops_of_type(mod, AsyncOp):
        if len(issue.completion_locks) > 1:
            assert issue.lock.definition.attrs["label"].startswith("bank_release_")
            assert (
                issue.completion_locks[1]
                .definition.attrs["label"]
                .startswith("accumulator_ready_")
            )
    for release in find_ops_of_type(mod, BarrierOp):
        if release.attrs.get("label") == "last_tmem_read":
            ops = release.parent_block.ops
            previous = ops[ops.index(release) - 1]
            assert isinstance(previous, MoveOp) and isinstance(
                previous.src.type, TmemTensorType
            )
            # Conversion/store of this fragment may cross the backedge;
            # slot release is placed immediately after the completed read.


@pytest.mark.parametrize("warps", [4, 8])
def test_serial_and_overlap_keep_the_same_storage_and_event_budget(warps):
    serial, overlap = (
        build(persistent_output_overlap=False, tmem_writeback_warps=warps),
        build(tmem_writeback_warps=warps),
    )

    def storage(mod):
        return sorted(
            (op.attrs.get("label", ""), str(op.result().type))
            for op in find_ops_of_type(mod, AllocOp)
        )

    assert storage(serial) == storage(overlap)
    assert serial.body.ops[0].block_dim == overlap.body.ops[0].block_dim
    assert serial.body.ops[0].grid_dim == overlap.body.ops[0].grid_dim
    for mod in (serial, overlap):
        for allocation in find_ops_of_type(mod, AllocOp):
            if isinstance(allocation.result().type, RegMemTensorType):
                assert allocation.parent_block is mod.body.ops[0].body


@pytest.mark.parametrize("overlap", [False, True])
def test_steady_engine_reuse_needs_no_collective_gather(overlap):
    mod = build(persistent_output_overlap=overlap)
    analysis = SyncAnalysis()
    analysis.analyze(mod)
    loop = find_ops_of_type(mod, ForOp)[0]
    issues = find_ops_of_type(loop, AsyncOp)
    assert any(op.lock.type.kind == "mbar" for op in issues)
    for issue in issues:
        if issue.lock.type.kind in ("mbar", "tcgen05"):
            assert analysis.targets[issue].scope == BarrierScope.THREAD
    # Readiness acquires all earlier ordered MMA writes, including writes
    # whose bank-release event was observed only by the engine issuer.
    for read in find_ops_of_type(loop, MoveOp):
        if isinstance(read.src.type, TmemTensorType):
            assert analysis.targets[read].scope == BarrierScope.THREAD


@pytest.mark.parametrize("invalid", ["duplicate", "mechanism", "relay_type", "unobserved"])
def test_invalid_additional_completions_are_rejected(invalid):
    mod = dual_completion()
    issue = find_ops_of_type(mod, AsyncOp)[0]
    if invalid == "duplicate":
        issue.operands[1] = issue.lock
    elif invalid == "mechanism":
        issue.completion_locks[1].type = LockType.build(kind="mbar")
    elif invalid == "relay_type":
        issue.attrs["release_prior"] = 1
    else:
        wait = find_ops_of_type(mod, AwaitOp)[0]
        wait.parent_block.ops.remove(wait)
    with pytest.raises(ValueError):
        VerifyGpuDialect().run(mod)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(operand_stages=1),
        dict(operand_stages=7),
        dict(cta_group=2),
        dict(output_n_tile=512),
        dict(warp_specific_operand_waits=True),
    ],
)
def test_unsupported_schedules_fail_explicitly(kwargs):
    with pytest.raises(ValueError, match="shared ring|stages"):
        build(**kwargs)


def dual_completion(group=1):
    text = """module attrs={target:"sm_100a"} {
      kernel @two_events(%a:smem<128x16xf16>, %b:smem<128x16xf16>,
        %r:wgrmem<128x128xf32 warps=4 layout="tcgen05_16x256b">) grid_dim=[1,1,1] block_dim=[128,1,1] {
        %c:tmem<128x128xf32>=alloc()
        %bank:lock<tcgen05>=alloc() attrs={label:"bank"}
        %ready:lock<tcgen05>=alloc() attrs={label:"ready"}
        async(%bank:lock<tcgen05>,%ready:lock<tcgen05>) collective=xyz.thread
          matmul(%a:smem<128x16xf16>,%b:smem<128x16xf16>,%c:tmem<128x128xf32>) collective=xyz.thread
        await(%ready:lock<tcgen05>)
        move(%r:wgrmem<128x128xf32 warps=4 layout="tcgen05_16x256b">,%c:tmem<128x128xf32>)
        await(%bank:lock<tcgen05>)
        dealloc(%c:tmem<128x128xf32>)
      }}"""
    if group == 2:
        text = text.replace("%b:smem<128x16xf16>", "%b:smem<64x16xf16>")
        text = text.replace("tmem<128x128xf32>", "tmem<128x128xf32 attrs={cta_group:2}>")
        text = text.replace("lock<tcgen05>", "lock<tcgen05 attrs={cta_group:2}>")
        text = text.replace("grid_dim=[1,1,1] block_dim=[128,1,1]",
                            "grid_dim=[2,1,1] block_dim=[128,1,1] cluster_dim=[2,1,1]")
    return gpu_dialect.parse_module(text)


def split_release(group, sync, phase):
    return BarrierOp.build(
        "cluster_to_leader" if group == 2 else "block_leader", sync,
        {"phase": phase, **({"participant_warps": [0, 1, 2, 3] if phase == "arrive" else [0]}
                            if group == 2 else {})},
    )


@pytest.mark.parametrize("group", [1, 2])
def test_additional_completion_round_trip_fusion_and_data_acquire(group):
    mod = dual_completion(group)
    assert repr(gpu_dialect.parse_module(repr(mod))) == repr(mod)
    expected = repr(clone_op(mod, {}))
    FuseAsync().run(mod)
    assert repr(mod) == expected
    analysis = SyncAnalysis()
    analysis.analyze(mod)
    read = find_ops_of_type(mod, MoveOp)[0]
    assert analysis.targets[read].scope == BarrierScope.THREAD
    ptx = make_ptx_kernel(mod).ptx
    assert ptx.count(f"tcgen05.commit.cta_group::{group}") == 2
    if group == 2:
        assert ptx.count(".shared::cluster.multicast::cluster.b64") == 2


def test_additional_completion_does_not_retire_primary_event_phase():
    mod = dual_completion()
    bank = find_ops_of_type(mod, AsyncOp)[0].lock
    # The bank event remains outstanding when only readiness is observed.
    analysis = SyncAnalysis()
    analysis.analyze(mod)
    ready_wait = find_ops_of_type(mod, AwaitOp)[0]
    state = analysis.live_out(ready_wait)
    assert any(
        effect.pending_await is bank for effect in state.concurrent_memory_effects[bank]
    )


@pytest.mark.parametrize("read_after_arrival", [False, True])
@pytest.mark.parametrize("group", [1, 2])
def test_split_release_publishes_only_tmem_reads_before_arrival(read_after_arrival, group):
    mod = dual_completion(group)
    kernel = mod.body.ops[0]
    read = find_ops_of_type(mod, MoveOp)[0]
    phase = AllocOp.build_with_fresh_id(
        LockType.build(kind="cluster_mbar" if group == 2 else "mbar",
                       attrs={"release_warps": [0, 1, 2, 3]} if group == 2 else {"arrivals": 4})
    )
    kernel.body.ops.insert(0, phase)
    insertion = kernel.body.ops.index(read) + 1
    ops = [split_release(group, phase.result(), "arrive")]
    if read_after_arrival:
        ops.append(clone_op(read, {}))
    ops.append(split_release(group, phase.result(), "wait"))
    # Reusing the accumulator is safe only for captured readers.
    original = find_ops_of_type(mod, AsyncOp)[0]
    overwrite = clone_op(original, {})
    overwrite.operands = [original.lock]
    ops += [overwrite, AwaitOp.build(original.lock)]
    # Observe the original primary phase before issuing its next generation.
    primary_wait = find_ops_of_type(mod, AwaitOp)[1]
    kernel.body.ops.remove(primary_wait)
    kernel.body.ops[insertion:insertion] = [primary_wait, *ops]
    analysis = SyncAnalysis()
    analysis.analyze(mod)
    assert analysis.targets[overwrite].scope == (
        (BarrierScope.CLUSTER if group == 2 else BarrierScope.BLOCK)
        if read_after_arrival else BarrierScope.THREAD
    )


@pytest.mark.parametrize("observation_after_arrival", [False, True])
@pytest.mark.parametrize("group", [1, 2])
def test_split_release_does_not_publish_later_event_observations(
    observation_after_arrival, group,
):
    mod = dual_completion(group)
    kernel = mod.body.ops[0]
    original = find_ops_of_type(mod, AsyncOp)[0]
    ready_wait, bank_wait = find_ops_of_type(mod, AwaitOp)
    read = find_ops_of_type(mod, MoveOp)[0]
    kernel.body.ops.remove(ready_wait)
    kernel.body.ops.remove(bank_wait)
    kernel.body.ops.insert(kernel.body.ops.index(read), bank_wait)
    phase = AllocOp.build_with_fresh_id(
        LockType.build(kind="cluster_mbar" if group == 2 else "mbar",
                       attrs={"release_warps": [0, 1, 2, 3]} if group == 2 else {"arrivals": 4})
    )
    kernel.body.ops.insert(0, phase)
    overwrite = clone_op(original, {})
    ops = [] if observation_after_arrival else [ready_wait]
    ops.append(split_release(group, phase.result(), "arrive"))
    if observation_after_arrival:
        ops.append(ready_wait)
    ops += [
        split_release(group, phase.result(), "wait"),
        overwrite,
        AwaitOp.build(original.lock),
        AwaitOp.build(original.completion_locks[1]),
    ]
    insertion = kernel.body.ops.index(read) + 1
    kernel.body.ops[insertion:insertion] = ops
    analysis = SyncAnalysis()
    analysis.analyze(mod)
    assert analysis.targets[overwrite].scope == (
        (BarrierScope.CLUSTER_TO_LEADER if group == 2 else BarrierScope.BLOCK_TO_LEADER)
        if observation_after_arrival
        else BarrierScope.THREAD
    )


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("buffers", [1, 2])
def test_gpu_shared_ring_changed_input_graph_replays(dtype, warps, buffers):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    m, n, grid = (769, 256, 1) if buffers == 1 else (1153, 520, 3)
    for overlap in (False, True):
        mod = build(
            m=m,
            n=n,
            output_grid_size=grid,
            dtype=dtype,
            tmem_writeback_warps=warps,
            output_slices=8 if buffers == 1 else 2,
            output_store_buffers=buffers,
            output_store_pending_groups=buffers - 1,
            persistent_output_overlap=overlap,
        )
        check_gemm(make_ptx_launcher(make_ptx_kernel(mod)), m, n, 384, dtype, graph=True)


def test_staging_banks_have_independent_completion_and_queued_source_waits():
    mod = build(
        m=1152,
        n=512,
        output_store_buffers=2,
        output_store_pending_groups=1,
    )
    source_events = [
        op.result()
        for op in find_ops_of_type(mod, AllocOp)
        if isinstance(op.result().type, LockType) and op.result().type.kind == "bulk"
    ]
    assert len(source_events) == 2
    by_bank = {}
    for store in find_ops_of_type(mod, AsyncOp):
        if store.lock.type.kind != "bulk":
            continue
        bank = frozenset(root_buffers(store.body.ops[0].src))
        by_bank.setdefault(bank, set()).add(store.lock)
    assert len(by_bank) == 2
    assert all(len(events) == 1 for events in by_bank.values())
    assert len(set().union(*by_bank.values())) == 2
    waits = find_ops_of_type(mod, AwaitOp)
    assert any(wait.lock in source_events and wait.pending_groups == 1 for wait in waits)
    drains = [wait for wait in waits if wait.attrs.get("label") == "output_terminal_drain"]
    assert {wait.lock for wait in drains} == set(source_events)
    # FIFO scheduling can observe the older bank with one newer group still
    # pending; the final observation must drain the whole issuer queue.
    assert drains[-1].is_full_wait
    for block in range(3):
        assert event_walk(mod, block)["store"] == 48


def test_five_bank_ring_event_generations_with_uneven_jobs():
    for overlap in (False, True):
        mod = build(
            m=1793,
            n=520,
            k=320,
            operand_stages=5,
            output_grid_size=7,
            tmem_writeback_register_budget=64,
            persistent_output_overlap=overlap,
        )
        assert_dominance(mod.body)
        for block in range(7):
            jobs = len(range(block, 75, 7))
            assert event_walk(mod, block) == dict(
                load=jobs * 5, mma=jobs * 5, read=jobs * 2, store=jobs * 2
            )


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gpu_shared_ring_five_bank_uneven_jobs(dtype):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    for overlap in (False, True):
        mod = build(
            m=1793,
            n=520,
            k=320,
            dtype=dtype,
            operand_stages=5,
            output_grid_size=7,
            tmem_writeback_register_budget=64,
            persistent_output_overlap=overlap,
        )
        check_gemm(make_ptx_launcher(make_ptx_kernel(mod)), 1793, 520, 320, dtype, graph=True)


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("k,banks", [(320, 4), (384, 4), (768, 7)])
def test_rotating_bank_generations_and_real_supercycle(group, overlap, k, banks):
    import math
    nt = 192 if banks == 7 and group == 2 else 96 if banks == 7 else 128
    m, n = (4097, 776) if banks == 7 else (2049, 520)
    mod = build(
        m=m, n=n, k=k, operand_stages=banks, cta_group=group,
        output_n_tile=nt, output_slices=3 if banks == 7 else 2,
        output_grid_size=4, output_grid_fast_axis="m",
        tmem_writeback_register_budget=64,
        persistent_output_overlap=overlap,
    )
    assert_dominance(mod.body)
    VerifyGpuDialect().run(mod)
    loop = next(loop for loop in find_ops_of_type(mod, ForOp)
                if loop.attrs.get("label") == "shared_operand_output_ring")
    assert loop.step == math.lcm(2, banks // math.gcd(k // 64, banks))
    assert len(range(loop.lb, loop.ub, loop.step)) >= 2
    tiles = math.ceil(m / (128 * group)) * group * math.ceil(n / nt)
    fragments = 3 if banks == 7 else 2
    for block in range(4):
        jobs = len(range(block, tiles, 4))
        assert event_walk(mod, block) == dict(
            load=jobs * (k // 64), mma=jobs * (k // 64),
            read=jobs * fragments, store=jobs * fragments,
        )
    # Every physical bank must have one stable set of operand roots even
    # when successive output jobs begin on different banks.
    roots_by_bank = {}
    for issue in find_ops_of_type(mod, AsyncOp):
        if issue.lock.type.kind != "mbar":
            continue
        roots = frozenset().union(*(root_buffers(move.dst) for move in issue.body.ops))
        roots_by_bank.setdefault(issue.attrs["ring_bank"], set()).add(roots)
    assert len(roots_by_bank) == banks
    assert all(len(roots) == 1 for roots in roots_by_bank.values())


@pytest.mark.parametrize("group,nt,banks,grid", [(1, 192, 5, 48), (1, 96, 7, 96),
                                                (2, 192, 7, 48), (2, 256, 6, 36)])
@pytest.mark.parametrize("overlap", [False, True])
def test_hoisted_operand_views_follow_rotated_banks(group, nt, banks, grid, overlap):
    mod = build(m=3072, n=768, k=1024, cta_group=group, output_n_tile=nt,
                operand_stages=banks, output_grid_size=grid,
                output_grid_fast_axis="m", output_slices=4 if nt == 256 else 3,
                tmem_writeback_register_budget=64, persistent_output_overlap=overlap)
    VerifyGpuDialect().run(mod)
    for block in (0, grid - 1):
        assert event_walk(mod, block) == dict(load=32, mma=32,
                                              read=8 if nt == 256 else 6,
                                              store=8 if nt == 256 else 6)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("group,k,banks,warps", [
    (1, 320, 4, 4), (1, 768, 7, 4),
    (2, 320, 4, 4), (2, 768, 7, 8), (2, 384, 6, 4),
    (1, 384, 4, 4), (2, 384, 4, 4),
    (1, 1024, 5, 4), (2, 1024, 7, 4),
])
def test_gpu_rotating_and_paired_ring_changed_input_replays(dtype, group, k, banks, warps):
    import torch
    from ayla.ptx_runtime import make_ptx_launcher
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    nt = 192 if k == 1024 or banks == 7 and group == 2 else 96 if banks == 7 else 128
    m, n = (3072, 768) if k == 1024 else (4097, 776) if banks == 7 else (2049, 520)
    for overlap in (False, True):
        mod = build(m=m, n=n, k=k, dtype=dtype, operand_stages=banks,
                    cta_group=group, output_n_tile=nt,
                    output_slices=3 if k == 1024 or banks == 7 else 2,
                    output_grid_size=48 if k == 1024 else 4, output_grid_fast_axis="m",
                    tmem_writeback_warps=warps, tmem_writeback_register_budget=64,
                    persistent_output_overlap=overlap)
        check_gemm(make_ptx_launcher(make_ptx_kernel(mod)), m, n, k, dtype, graph=True)
