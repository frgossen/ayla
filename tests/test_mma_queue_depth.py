"""Cyclic operand pipelines that keep d tcgen05 MMA batches queued.

Also: shared symbols translated by cluster `mapa` live in one arena (ptx_codegen).
"""

import re

import pytest

from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.passes.schedule_async import ScheduleAsync
from ayla.ptx_codegen import make_ptx_kernel

PAIRED = dict(cta_group=2, tmem_writeback_register_budget=64)
CLUSTER = dict(cta_group=2, cluster_n_ctas=2, cluster_dim=(4, 1, 1), tmem_writeback_register_budget=64)


def _queued_batches(ptx):
    """For every completion wait in the K loop: MMA batches issued after the observed one."""
    body = ptx[ptx.index("// BEGIN: `for"):ptx.index("// END: `for")]
    events = []
    for m in re.finditer(
        r"// PTX template `(probe-ahead )?(await|async)\(%0: lock<tcgen05[^\n]*\n(.*?)(?=// PTX template|\Z)",
        body, re.S,
    ):
        if m.group(1):
            continue
        if m.group(2) == "async":
            events.append(("issue", set(re.findall(r"tcgen05\.commit[^\[]*\[(%r\d+)\]", m.group(3)))))
        else:
            events.append(("wait", re.search(r"mbarrier\.try_wait[^\[]*\[(%r\d+)\]", m.group(3)).group(1)))
    # Two copies: waits at the top of the body observe batches of the previous iteration.
    sequence, queued = events * 2, []
    for i in range(len(events), len(sequence)):
        kind, event = sequence[i]
        if kind == "wait":
            issues = [e for _, e in reversed(sequence[:i]) if _ == "issue"]
            queued.append(next(j for j, e in enumerate(issues) if event in e))
    return queued


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("layout", [{}, PAIRED, CLUSTER], ids=["single", "pair", "cluster"])
def test_each_operand_refill_observes_the_batch_d_batches_back(depth, layout):
    ptx = make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=1024, n=1024, k=4096, dtype="f16", output_n_tile=64, operand_stages=8,
        cyclic_operand_pipeline=depth, **layout,
    )).ptx
    queued = _queued_batches(ptx)
    assert len(queued) == 8 and set(queued) == {depth}


@pytest.mark.parametrize("config", [
    dict(operand_stages=2, cyclic_operand_pipeline=2),
    dict(operand_stages=8, cyclic_operand_pipeline=-1),
    dict(operand_stages=8, cyclic_operand_pipeline=8, rolled_operand_pipeline=True),
])
def test_invalid_mma_queue_depth(config):
    with pytest.raises(ValueError):
        blackwell_tcgen05_gemm_module(m=512, n=512, k=1024, output_n_tile=64, **config)


def test_mma_queue_depth_requires_compute_pipelining():
    with pytest.raises(ValueError, match="compute pipelining"):
        ScheduleAsync(mma_queue_depth=2)
    with pytest.raises(ValueError, match="positive integer"):
        ScheduleAsync(pipeline_compute=True, mma_queue_depth=0)


def _ptx(k, **layout):
    return make_ptx_kernel(blackwell_tcgen05_gemm_module(
        m=1024, n=1024, k=k, dtype="f16", output_n_tile=64, operand_stages=8,
        cyclic_operand_pipeline=2, **layout,
    )).ptx


@pytest.mark.parametrize("layout", [CLUSTER, PAIRED], ids=["cluster", "pair"])
def test_every_mapa_translates_the_arena_base_by_an_immediate(layout):
    ptx = _ptx(2048, **layout)
    [(arena, size)] = re.findall(r"\.shared \.align \d+ \.b8 (_smem_\d+)\[(\d+)\];", ptx)
    [base] = re.findall(rf"mov\.u32 (%r\d+), {arena};", ptx)
    mapas = re.findall(r"mapa\.shared::cluster\.u32 (%r\d+), (%r\d+), %r\d+;\n\s*(.*)", ptx)
    assert mapas
    for dst, source, following in mapas:
        assert source == base
        offset = re.fullmatch(rf"add\.u32 {dst}, {dst}, (\d+);", following)
        assert offset and int(offset[1]) < int(size)


def test_kernels_without_mapa_keep_separate_symbols():
    ptx = _ptx(2048)
    assert "mapa" not in ptx
    assert not re.search(r"_smem_\d+\+\d+", ptx)
