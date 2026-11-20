"""Issuer identity, asymmetric completion gathers, and shared GEMM schedules."""
import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AsyncOp,AwaitOp,BarrierOp,CollectiveScope,MatmulOp,MoveOp,
    execution_issuer_warp,gpu_dialect,
)
from ayla.analysis.sync import BarrierScope,SyncAnalysis,_barrier_scope
from ayla.passes.fuse_async import FuseAsync
from ayla.passes.lower_to_tma_move import LowerToTmaMove
from ayla.passes.schedule_async import _can_swap
from ayla.passes.insert_syncs import InsertSyncs
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import DTYPES,LOOSE,check_gemm


@pytest.mark.parametrize('mma_issuer', [0, 2])
def test_completion_initialization_uses_event_issuer(mma_issuer):
    from ayla.gpu_dialect import AllocOp, LockType
    module = events(mma_issuer, mma_issuer)
    InsertSyncs().run(module)
    locks = {op.result().name: op for op in find_ops_of_type(module, AllocOp)
             if isinstance(op.result().type, LockType)}
    assert locks['lt'].attrs['issuer_warp'] == 1
    assert locks['l0'].attrs.get('issuer_warp', 0) == mma_issuer
    assert 'issuer_warp' not in locks['l1'].attrs
    if mma_issuer == 0:
        assert 'issuer_warp' not in locks['l0'].attrs


def test_completion_event_cannot_change_issuer():
    with pytest.raises(ValueError, match='completion lock requires one issuing warp'):
        InsertSyncs().run(events(0, 1))


def test_repeated_completion_events_are_retired_after_observers():
    from ayla.gpu_dialect import AllocOp, DeallocOp, LockType
    module = blackwell_tcgen05_gemm_module(
        m=4096, n=128, k=512, dtype='bf16', output_n_tile=128,
        operand_k_tile=64, operand_stages=3, cyclic_operand_pipeline=True,
        output_grid_size=3, reuse_tmem_allocation=True,
        operand_move_attrs={'lhs': {'issuer_warp': 0}, 'rhs': {'issuer_warp': 0}},
    )
    InsertSyncs().run(module)
    releases = [op for op in find_ops_of_type(module, DeallocOp)
                if isinstance(op.value.type, LockType)]
    assert releases
    for release in releases:
        alloc = release.value.definition
        assert isinstance(alloc, AllocOp)
        assert execution_issuer_warp(release) == execution_issuer_warp(alloc)
        block = release.parent_block
        assert block is not None
        index = block.ops.index(release)
        barrier = max(i for i, op in enumerate(block.ops[:index])
                      if isinstance(op, BarrierOp) and op.scope == 'block')
        assert all(isinstance(op, DeallocOp) for op in block.ops[barrier+1:index])


def test_exhausted_named_counters_upgrade_to_cta_rendezvous():
    from ayla.analysis.sync import SyncState, ConcurrentMemoryEffect, _transfer_memory_effects
    module = events()
    load = find_ops_of_type(module, AsyncOp)[0]
    dst = find_ops_of_type(load, MoveOp)[0].dst
    state = SyncState.empty()
    state.actor_tracking = True
    state.named_barrier_ids = {(BarrierScope.BLOCK_TO_LEADER, i) for i in range(1, 16)}
    state.concurrent_memory_effects[dst] = {
        ConcurrentMemoryEffect('R', CollectiveScope.THREAD, None, None),
    }
    targets = {}
    after = _transfer_memory_effects(load, state, targets, async_lock=load.lock)
    assert targets[load].scope == BarrierScope.BLOCK
    assert targets[load].named_id is None
    assert not after.named_barrier_ids


def events(first_mma=0,second_mma=0):
    return gpu_dialect.parse_module(f'''
    module attrs={{target: "sm_100a"}} {{
      kernel @events(%g: gmem<128x64xf16>) block_dim=[128,1,1] {{
        %a: smem<128x64xf16> = alloc()
        %b: smem<128x64xf16> = alloc()
        %c: tmem<128x128xf32> = alloc()
        %lt: lock<mbar> = alloc()
        %l0: lock<tcgen05> = alloc()
        %l1: lock<tcgen05> = alloc()
        async(%lt: lock<mbar>) attrs={{label: "load0"}} move(%a: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={{tma: True, issuer_warp: 1}}
        await(%lt: lock<mbar>)
        async(%l0: lock<tcgen05>) attrs={{label: "mma0"}} matmul(%a: smem<128x64xf16>, %b: smem<128x64xf16>, %c: tmem<128x128xf32>) attrs={{issuer_warp: {first_mma}}}
        await(%l0: lock<tcgen05>)
        async(%lt: lock<mbar>) attrs={{label: "load1"}} move(%a: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={{tma: True, issuer_warp: 1}}
        await(%lt: lock<mbar>)
        async(%l0: lock<tcgen05>) attrs={{label: "mma1"}} matmul(%a: smem<128x64xf16>, %b: smem<128x64xf16>, %c: tmem<128x128xf32>) attrs={{issuer_warp: {second_mma}}}
        await(%l0: lock<tcgen05>)
      }}
    }}''')


def test_another_producer_gather_keeps_completion_observers():
    m=events();a=SyncAnalysis();a.analyze(m)
    loads=[o for o in find_ops_of_type(m,AsyncOp) if o.lock.type.kind=='mbar']
    mmas=[o for o in find_ops_of_type(m,AsyncOp) if o.lock.type.kind=='tcgen05']
    assert a.targets[loads[-1]].scope==BarrierScope.BLOCK_TO_LEADER
    assert a.targets[loads[-1]].issuer_warp==1
    assert a.targets[mmas[-1]].scope==BarrierScope.BLOCK_TO_LEADER
    assert a.targets[mmas[-1]].issuer_warp==0
    observations=a.live_in(mmas[-1]).concurrent_memory_effects[mmas[-1].lock]
    assert any(e.scope==CollectiveScope.THREAD and 1 in e.gathered_by for e in observations)


@pytest.mark.parametrize('prior,current,expected',[(0,0,BarrierScope.THREAD),(1,1,BarrierScope.THREAD),(0,1,BarrierScope.BLOCK),(2,1,BarrierScope.BLOCK)])
def test_distinct_leaders_require_publication(prior,current,expected):
    assert _barrier_scope(CollectiveScope.BLOCK_LEADER,CollectiveScope.BLOCK_LEADER,prior,current)==expected


@pytest.mark.parametrize('rhs,expected',[(1,1),(2,2)])
def test_fusion_keeps_distinct_issuers_separate(rhs,expected):
    m=gpu_dialect.parse_module(f'''
      module {{kernel @loads(%g: gmem<128x64xf16>) {{
        %a: smem<128x64xf16> = alloc()
        %b: smem<128x64xf16> = alloc()
        %la: lock<mbar> = alloc()
        %lb: lock<mbar> = alloc()
        async(%la: lock<mbar>) move(%a: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={{tma: True,issuer_warp:1}}
        await(%la: lock<mbar>)
        async(%lb: lock<mbar>) move(%b: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={{tma: True,issuer_warp:{rhs}}}
        await(%lb: lock<mbar>)
      }} }}''')
    FuseAsync().run(m)
    groups=find_ops_of_type(m,AsyncOp)
    assert len(groups)==expected
    assert {execution_issuer_warp(g) for g in groups}=={1,rhs}


@pytest.mark.parametrize('second,expected',[(0,True),(1,False)])
def test_mma_wait_reordering_requires_same_issuer(second,expected):
    m=events(second_mma=second)
    mmas=[o for o in find_ops_of_type(m,AsyncOp) if o.lock.type.kind=='tcgen05']
    wait=next(o for o in find_ops_of_type(m,AwaitOp) if o.lock is mmas[0].lock)
    # Give the later group a distinct event: engine ordering is the only
    # possible reason to move this completion wait through its issue.
    kernel=m.body.ops[0]
    from ayla.gpu_dialect import AllocOp,LockType
    lock=AllocOp.build_with_fresh_id(LockType.build(kind='tcgen05'))
    kernel.body.ops.insert(0,lock);mmas[1].operands[0]=lock.result()
    following=kernel.body.ops[kernel.body.ops.index(mmas[1])+1]
    assert isinstance(following,AwaitOp)
    following.operands[0]=lock.result()
    assert _can_swap(wait,mmas[1])==expected


def test_bulk_wait_does_not_publish_to_another_issuer():
    m=gpu_dialect.parse_module('''
      module attrs={target:"sm_100a"} {kernel @bulk(%src:gmem<128x64xf16>, %dst:gmem<128x64xf16>) block_dim=[128,1,1] {
        %s:smem<128x64xf16 swizzle="128B">=alloc()
        %ls:lock<bulk>=alloc()
        %ll:lock<mbar>=alloc()
        async(%ls:lock<bulk>) move(%dst:gmem<128x64xf16>,%s:smem<128x64xf16 swizzle="128B">) attrs={tma:True,issuer_warp:0}
        await(%ls:lock<bulk>)
        async(%ll:lock<mbar>) move(%s:smem<128x64xf16 swizzle="128B">,%src:gmem<128x64xf16>) attrs={tma:True,issuer_warp:1}
        await(%ll:lock<mbar>)
      }}''')
    LowerToTmaMove().run(m);a=SyncAnalysis();a.analyze(m)
    load=next(o for o in find_ops_of_type(m,AsyncOp) if o.lock.type.kind=='mbar')
    assert a.targets[load].scope==BarrierScope.BLOCK


@pytest.mark.parametrize('issuer',[-1,4,True,'1'])
def test_invalid_issuer_rejected(issuer):
    with pytest.raises(ValueError):
        make_ptx_kernel(blackwell_tcgen05_gemm_module(m=512,n=128,k=64,dtype='f16',output_n_tile=128,operand_stages=1,operand_move_attrs={'lhs':{'issuer_warp':issuer},'rhs':{'issuer_warp':issuer}}))


GPU_CASES=[
    dict(n=128,k=512,operand_stages=3,cyclic_operand_pipeline=True),
    dict(n=128,k=512,operand_stages=3,cyclic_operand_pipeline=False),
    dict(n=128,k=640,operand_stages=5,cyclic_operand_pipeline=True,load_warp=3),
    dict(n=64,k=1024,output_n_tile=64,operand_stages=8,cyclic_operand_pipeline=True),
    dict(n=64,k=1280,output_n_tile=64,output_m_tile=64,operand_stages=10,cyclic_operand_pipeline=True),
    dict(n=512,k=64,output_n_tile=512,output_store='stmatrix',output_slices=2,output_store_buffers=2,tmem_writeback_register_budget=128),
    dict(n=256,k=256,output_n_tile=256,operand_stages=4,cyclic_operand_pipeline=True,output_store='stmatrix',output_slices=2,output_store_buffers=2,tmem_writeback_warps=8,load_warp=5),
    dict(n=128,k=128,operand_stages=2,cyclic_operand_pipeline=True),
    dict(n=144,k=144,output_n_tile=144,operand_stages=3,cyclic_operand_pipeline=True,allow_partial_tiles=True),
    dict(n=16,k=16,output_n_tile=16,operand_k_tile=16,output_m_tile=64,tmem_writeback_warps=8,output_layout='tcgen05_16x256b'),
    dict(n=8,k=16,output_n_tile=16,operand_k_tile=16,output_m_tile=64,tmem_writeback_warps=8,output_layout='tcgen05_16x256b',allow_partial_tiles=True),
    dict(n=128,k=512,operand_stages=3,cyclic_operand_pipeline=True,rhs_warp=2),
    dict(n=128,k=512,operand_stages=3,cyclic_operand_pipeline=True,output_store='stmatrix',store_warp=2),
    dict(n=128,k=512,operand_stages=3,cyclic_operand_pipeline=True,load_warp=2,mma_warp=1),
    dict(m=1536,n=384,k=12608,output_n_tile=64,output_m_tile=64,operand_stages=8,cyclic_operand_pipeline=True,allow_partial_tiles=True,output_layout='tcgen05_16x256b',tmem_writeback_register_budget=128,rhs_warp=2),
    dict(m=1536,n=512,k=16448,output_n_tile=64,operand_stages=8,cyclic_operand_pipeline=True,output_store='stmatrix',rhs_warp=2),
]


def issuer_gemm(case,dtype,**options):
    """Run a GPU_CASES schedule (with extra builder `options`) on changing operands."""
    c=dict(case);load=c.pop('load_warp',1);rhs=c.pop('rhs_warp',load)
    store=c.pop('store_warp',0);mma=c.pop('mma_warp',0)
    config=dict(m=4096,n=128,k=64,output_n_tile=128,operand_k_tile=64,operand_stages=1,
        output_grid_size=3,output_grid_fast_axis='n',reuse_tmem_allocation=True,
        operand_move_attrs={'lhs':{'issuer_warp':load},'rhs':{'issuer_warp':rhs},**({'dst':{'issuer_warp':store}} if store else {})})
    config.update(c)
    if load or rhs or store or mma:
        with pytest.raises(ValueError, match='without warp specialization'):
            blackwell_tcgen05_gemm_module(dtype=dtype,**options,**config)
        return
    module=blackwell_tcgen05_gemm_module(dtype=dtype,**options,**config)
    if mma:
        for op in find_ops_of_type(module,MatmulOp):op.attrs['issuer_warp']=mma
    check_gemm(make_ptx_launcher(make_ptx_kernel(module)),*(config[x] for x in 'mnk'),dtype,
               mutate=True,runs=3,seed=98743,**LOOSE)


@pytest.mark.parametrize('case',GPU_CASES)
@pytest.mark.parametrize('dtype',DTYPES)
def test_historical_separate_warp_schedules_are_rejected(case,dtype):
    issuer_gemm(case,dtype)
