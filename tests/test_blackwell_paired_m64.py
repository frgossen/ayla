"""Blackwell's paired M128 instruction has a 2x2 TMEM result layout."""
import math

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AwaitOp, BarrierOp, MoveOp, TmemTensorType
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from ayla.tmem_layout import TmemWarpLayout
from tests.testing_utils import check_gemm


@pytest.mark.parametrize('warps', [4, 8])
@pytest.mark.parametrize('columns', [64, 96, 128, 192, 256])
def test_2x2_fragment_covers_every_value_once(warps, columns):
    ownership = TmemWarpLayout(64, warps, 2)
    logical = set()
    physical = set()
    for warp in range(warps):
        for row in range(ownership.rows_per_warp):
            for col in range(ownership.columns_per_warp(columns)):
                r = ownership.row_origin(warp) + row
                c = ownership.column_origin(warp, columns) + col
                p = ownership.row_origin(warp, physical=True) + row
                assert p // 32 == warp % 4
                # CUTLASS's SM100 2x2 accumulator atom: (M,N/2,2)
                # strides (DP, column, 64 DP).
                assert p == r + (c // (columns // 2)) * 64
                logical.add((r, c))
                physical.add((p, c % (columns // 2)))
    assert logical == {(r, c) for r in range(64) for c in range(columns)}
    assert len(physical) == 64 * columns


def test_budget_is_enforced_instead_of_silently_widening_the_fragment():
    with pytest.raises(ValueError, match='whole writeback fragment'):
        blackwell_tcgen05_gemm_module(m=256, n=384, k=512, cta_group=2,
            output_m_tile=64, output_n_tile=192, output_store='stmatrix')


CASES = [
    (256, 128, 512, 64, 64, 3, 4),
    (384, 384, 512, 192, 64, 6, 4),
    (384, 384, 400, 192, 64, 6, 8),
    (256, 256, 832, 128, 256, 2, 4),
    (320, 128, 448, 64, 128, 3, 4),
    (256, 512, 272, 192, 64, 3, 4),
    (256, 192, 400, 96, 64, 3, 4),
]


@pytest.mark.parametrize('dtype', ['f16', 'bf16'])
@pytest.mark.parametrize('m,n,k,nt,kt,stages,warps', CASES)
def test_common_loop_bank_reuse_and_output_edges(dtype, m, n, k, nt, kt, stages, warps):
    module = blackwell_tcgen05_gemm_module(m=m, n=n, k=k, dtype=dtype,
        cta_group=2, output_m_tile=64, output_n_tile=nt, operand_k_tile=kt,
        operand_stages=stages, cyclic_operand_pipeline=True,
        output_store='stmatrix', output_layout='tcgen05_16x256b',
        tmem_writeback_warps=warps, tmem_writeback_register_budget=64 * nt // (32 * warps),
        allow_partial_tiles=True)
    waits = find_ops_of_type(module, AwaitOp)
    assert any(w.lock.type.kind == 'tcgen05' and w.waiter_warps == (0,) for w in waits)
    assert any(w.lock.type.kind == 'tcgen05' and w.waiter_warps is None for w in waits)
    assert any(b.attrs.get('mma_wait_handoff') and b.scope == 'cluster'
               for b in find_ops_of_type(module, BarrierOp))
    reads = [op for op in find_ops_of_type(module, MoveOp) if isinstance(op.src.type, TmemTensorType)]
    assert len(reads) == 1 and reads[0].dst.type.attrs['tmem_cta_group'] == 2
    check_gemm(make_ptx_launcher(make_ptx_kernel(module)), m, n, k, dtype, graph=True, runs=3,
               seed=21374, atol=math.inf)


@pytest.mark.parametrize('relay_enabled', [True, False])
def test_transfer_completion_retires_phase_reads_only(relay_enabled):
    from ayla.analysis.sync import SyncAnalysis, SyncState, ConcurrentMemoryEffect
    from ayla.gpu_dialect import AsyncOp, CollectiveScope
    from ayla.passes.lower_to_tma_move import LowerToTmaMove
    from ayla.passes.lower_to_async import LowerToAsync
    from ayla.pipelines.canonicalize import Canonicalize

    module = blackwell_tcgen05_gemm_module(m=256, n=128, k=512,
        cta_group=2, output_m_tile=64, output_n_tile=64,
        operand_stages=3, output_store='stmatrix')
    module = Canonicalize().run(LowerToAsync().run(LowerToTmaMove().run(module)))
    module.body.ops[0].attrs['paired_phase_relay'] = relay_enabled
    waits = find_ops_of_type(module, AwaitOp)
    bank_wait = next(w for w in waits if w.lock.type.kind == 'tcgen05' and w.waiter_warps == (0,))
    refill = next(op for op in find_ops_of_type(module, AsyncOp) if op.lock.type.kind == 'mbar')
    ready = next(w for w in waits if w.lock is refill.lock)
    old_phase = ConcurrentMemoryEffect('R', CollectiveScope.THREAD, None, None,
        observed_lock=bank_wait.lock, completed_after=bank_wait, visible_to=frozenset({0}))
    # A data access is not a phase observation and must keep its own dependency.
    unrelated = ConcurrentMemoryEffect('W', CollectiveScope.CLUSTER, None, None,
        completed_after=bank_wait, visible_to=frozenset({0}))
    state = SyncState.empty()
    state.concurrent_memory_effects[bank_wait.lock] = {old_phase, unrelated}
    analysis = SyncAnalysis()
    state = analysis.transfer_async(refill, state)
    state = analysis.transfer_await(ready, state)
    effects = state.concurrent_memory_effects[bank_wait.lock]
    assert any(effect.kind == 'W' for effect in effects)
    assert any(effect.observed_lock is bank_wait.lock for effect in effects) != relay_enabled
