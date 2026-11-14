import pytest
import torch

from ayla.profiling._sm_trace import (
    READINESS_SNAPSHOT,
    _decode,
    _readiness_snapshots,
    _tag_resident_tasks,
    _capture_batch,
    _bench_batch,
    _launch_activity_summary,
    _work_segments,
)


def test_launch_summary_includes_resets_but_not_cpu_launch_events(tmp_path):
    import json

    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel"}, {"cat": "gpu_memset"},
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch"},
    ]}))
    assert _launch_activity_summary(path) == dict(
        compute_launches=1, memset_operations=1, memcpy_operations=0,
    )


def test_preparation_and_metadata_regions_are_not_counted_twice():
    raw = torch.zeros((1, 7, 2), dtype=torch.int64)
    raw[0, 0] = torch.tensor([32, 6])
    raw[0, 1:] = torch.tensor([
        [100, 11], [200, 3019], [250, 21],
        [300, 22], [400, 3020], [500, 12],
    ])
    intervals, _ = _decode(raw)
    segments = _work_segments(intervals)
    setup = [r for r in segments if r["kind"] == "coefficient / row setup"]
    assert len(setup) == 2
    assert sum(r["end_ms"] - r["start_ms"] for r in setup) == pytest.approx(150 / 1e6)
    assert any(r["kind"] == "scheduler metadata / dispatch" for r in segments)


@pytest.mark.parametrize(
    "observed,kind", [(3, "dependency wait"), (4, "readiness acquire / fence")]
)
def test_counter_snapshot_distinguishes_blocking_from_ready_checks(observed, kind):
    raw = torch.zeros((1, 6, 2), dtype=torch.int64)
    raw[0, 0] = torch.tensor([7, 5])
    packet = READINESS_SNAPSHOT | (2 << 48) | (5 << 32) | (4 << 16) | observed
    raw[0, 1:] = torch.tensor(
        [[100, 11], [200, 3007], [300, packet], [400, 3008], [500, 12]]
    )
    [interval], [lifetime] = _decode(raw)
    assert interval["kind"] == kind
    assert interval["root"] == 2
    assert lifetime["sm"] == 7
    [snapshot] = _readiness_snapshots(raw)
    assert (snapshot["event"], snapshot["expected"], snapshot["observed"]) == (
        5,
        4,
        observed,
    )


def test_resident_full_and_partial_wave_regions_share_a_phase_root():
    from ayla.base.ir import Block
    from ayla.gpu_dialect import (
        BlockIdOp,
        CollectiveAxis,
        ForOp,
        IfOp,
        IndexConstantOp,
        IndexAddOp,
        IndexLtOp,
        KernelOp,
        ModuleOp,
    )

    full = ForOp.build(
        0, 132, 132, attrs={"tile_output_loop": True, "tile_task_count": 256}
    )
    partial = ForOp.build(
        132, 256, 132, attrs={"tile_output_loop": True, "tile_task_count": 256}
    )
    limit = IndexConstantOp.build(256)
    block = BlockIdOp.build(CollectiveAxis.X)
    tile = IndexAddOp.build(partial.iv, block.value)
    valid = IndexLtOp.build(tile.result(), limit.result())
    guard = IfOp.build(valid.result(), Block())
    partial.body.ops = [block, tile, limit, valid, guard]
    first_wave = IndexConstantOp.build(132)
    first = IndexLtOp.build(full.iv, first_wave.result())
    prefetch_guard = IfOp.build(first.result(), Block())
    full.body.ops = [first_wave, first, prefetch_guard]
    kernel = KernelOp.build(
        "split_waves",
        Block(ops=[full, partial]),
        grid_dim=(132, 1, 1),
        block_dim=(256, 1, 1),
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    roots = _tag_resident_tasks(module)
    assert len(roots) == 1
    assert full.attrs["sm_trace_root"] == guard.attrs["sm_trace_root"] == 0
    assert "sm_trace_root" not in prefetch_guard.attrs
    assert _tag_resident_tasks(module) == roots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_timing_batch_captures_every_launch_and_its_reset():
    launches = torch.zeros((), device="cuda")
    scratch = torch.zeros((), device="cuda")

    def run():
        scratch.zero_()
        scratch.add_(1)
        launches.add_(1)

    batch = _capture_batch(run, replays=7)
    batch.replay()
    batch.replay()
    assert launches.item() == 14
    assert scratch.item() == 1
    assert _bench_batch(batch, replays=7) > 0
    assert launches.item() == 28
    assert scratch.item() == 1
