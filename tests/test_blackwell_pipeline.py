"""Ordering and hardware regressions for queued MMA and coalesced writeback."""

import pytest
import torch

from ayla.analysis.sync import BarrierScope, SyncAnalysis
from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AsyncOp, AwaitOp, ForOp, KernelOp, MoveOp, gpu_dialect
from ayla.passes.pipeline_loop import PipelineLoop
from ayla.passes.schedule_async import ScheduleAsync
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher


def _module(body):
    return gpu_dialect.parse_module(f"""
      module attrs={{target: "sm_100a"}} {{
        kernel @k(%a: smem<128x16xbf16>, %b: smem<128x16xbf16>, %g: gmem<128x16xbf16>, %c: tmem<128x128xf32>, %r: wgrmem<128x128xf32 warps=4 layout="tcgen05_32x32b">) block_dim=[128, 1, 1] {{
          %l0: lock<tcgen05> = alloc()
          %l1: lock<tcgen05> = alloc()
          %load: lock<mbar> = alloc()
          {body}
        }}
      }}
    """)


MMA0 = """async(%l0: lock<tcgen05>) collective=xyz.block attrs={label: "compute0"} matmul(%a: smem<128x16xbf16>, %a: smem<128x16xbf16>, %c: tmem<128x128xf32>) collective=xyz.block"""
MMA1 = """async(%l1: lock<tcgen05>) collective=xyz.block attrs={label: "compute1"} matmul(%b: smem<128x16xbf16>, %b: smem<128x16xbf16>, %c: tmem<128x128xf32>) collective=xyz.block"""
LOAD = """async(%load: lock<mbar>) collective=xyz.block move(%a: smem<128x16xbf16>, %g: gmem<128x16xbf16>) collective=xyz.block attrs={tma: true}"""
READ = """move(%r: wgrmem<128x128xf32 warps=4 layout="tcgen05_32x32b">, %c: tmem<128x128xf32>) collective=xyz.block"""


@pytest.mark.parametrize("trips", [1, 4])
def test_event_pipeline_primes_and_drains_each_event(trips):
    module = _module(f"""for (%k: index = 0 to {trips}) attrs={{label: "k"}} {{
      {MMA0}
      await(%l0: lock<tcgen05>)
      {MMA1}
      await(%l1: lock<tcgen05>)
    }}""")
    PipelineLoop("k", "compute.*", 1).run(module)
    [kernel] = find_ops_of_type(module, KernelOp)
    pending = set()
    issues = []

    def execute(ops):
        for op in ops:
            if isinstance(op, ForOp):
                for _ in range(op.trip_count):
                    execute(op.body.ops)
            elif isinstance(op, AsyncOp):
                assert op.lock not in pending  # Never rearm an outstanding event.
                pending.add(op.lock)
                issues.append(op.lock.name)
            elif isinstance(op, AwaitOp):
                assert op.lock in pending  # No unprimed parity wait.
                assert "pending" not in op.attrs  # Named events have no wait_group N.
                pending.remove(op.lock)

    execute(kernel.body.ops)
    assert not pending
    assert issues == ["l0", "l1"] * trips


@pytest.mark.parametrize("tail", [READ, LOAD])
def test_event_pipeline_rejects_moving_completion_past_a_consumer(tail):
    module = _module(f"""for (%k: index = 0 to 4) attrs={{label: "k"}} {{
      {MMA0}
      await(%l0: lock<tcgen05>)
      {tail}
    }}""")
    with pytest.raises(ValueError, match="dependent operation"):
        PipelineLoop("k", "compute.*", 1).run(module)


def test_schedule_queues_mma_but_waits_before_smem_reuse_and_tmem_read():
    module = _module(f"""{MMA0}
      await(%l0: lock<tcgen05>)
      {MMA1}
      await(%l1: lock<tcgen05>)
      {LOAD}
      {READ}
    """)
    ScheduleAsync(pipeline_compute=True).run(module)
    [kernel] = find_ops_of_type(module, KernelOp)
    issues = find_ops_of_type(module, AsyncOp)
    waits = find_ops_of_type(module, AwaitOp)
    [read] = find_ops_of_type(module, MoveOp)[-1:]
    ops = kernel.body.ops
    assert ops.index(issues[0]) < ops.index(issues[1]) < ops.index(waits[0])
    assert ops.index(waits[0]) < ops.index(issues[2])
    assert ops.index(waits[1]) < ops.index(read)
    analysis = SyncAnalysis()
    analysis.analyze(module)
    assert analysis.targets[issues[1]].scope == BarrierScope.THREAD


def test_ordered_accumulator_does_not_waive_other_memory_hazards():
    module = _module(f"{MMA0}\n{MMA1}\n{LOAD}\n{READ}")
    analysis = SyncAnalysis()
    analysis.analyze(module)
    _, second, load = find_ops_of_type(module, AsyncOp)
    read = find_ops_of_type(module, MoveOp)[-1]
    assert analysis.targets[second].scope == BarrierScope.THREAD
    assert analysis.targets[load].scope == BarrierScope.BLOCK_TO_LEADER
    assert analysis.targets[read].scope == BarrierScope.BLOCK


def test_schedule_preserves_operand_readiness_and_event_reuse():
    module = _module(f"""{LOAD}
      await(%load: lock<mbar>)
      {MMA0}
      await(%l0: lock<tcgen05>)
      {MMA0}
      await(%l0: lock<tcgen05>)
      {READ}
    """)
    ScheduleAsync(pipeline_compute=True).run(module)
    [kernel] = find_ops_of_type(module, KernelOp)
    load, first, second = find_ops_of_type(module, AsyncOp)
    ready, completed_first, completed_second = find_ops_of_type(module, AwaitOp)
    ops = kernel.body.ops
    assert ops.index(load) < ops.index(ready) < ops.index(first)
    assert ops.index(first) < ops.index(completed_first) < ops.index(second)
    assert ops.index(second) < ops.index(completed_second)


def test_event_pipeline_requires_distinct_stage_events():
    module = _module(f"""for (%k: index = 0 to 4) attrs={{label: "k"}} {{
      {MMA0}
      await(%l0: lock<tcgen05>)
      {MMA0}
      await(%l0: lock<tcgen05>)
    }}""")
    with pytest.raises(ValueError, match="distinct completion events"):
        PipelineLoop("k", "compute.*", 1).run(module)


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)


@requires_blackwell
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize(
    "n_tile,k_tile,stages,k",
    [
        (8, 64, 2, 320),
        (16, 64, 8, 1024),
        (24, 64, 3, 448),
        (64, 64, 4, 640),
        (128, 128, 3, 896),
        (256, 64, 4, 1024),
        (240, 16, 4, 720),
        (144, 16, 4, 432),
    ],
)
def test_queued_writeback_exact_across_bank_wraps_and_reduction_tails(
    dtype, n_tile, k_tile, stages, k
):
    m, n = 256, n_tile * 3
    kernel = make_ptx_kernel(
        blackwell_tcgen05_gemm_module(
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            output_n_tile=n_tile,
            operand_k_tile=k_tile,
            operand_stages=stages,
            cyclic_operand_pipeline=True,
            output_grid_size=3,
        )
    )
    if n_tile % 16 == 0:
        assert "tcgen05.ld.sync.aligned.16x32bx2.x8.b32" in kernel.ptx
    launch = make_ptx_launcher(kernel)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    for seed in (0, 11, 73):
        torch.manual_seed(seed)
        a = torch.randn(m, k, device="cuda", dtype=torch_dtype)
        b = torch.randn(n, k, device="cuda", dtype=torch_dtype)
        c = torch.full((m, n), float("nan"), device="cuda", dtype=torch_dtype)
        for _ in range(10):
            launch(a, b, c)
        torch.testing.assert_close(c, a @ b.t(), rtol=0, atol=0)
