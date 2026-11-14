"""Warp-local completion visibility, release relay, and persistent event reuse."""
import pytest
from ayla.analysis.sync import SyncAnalysis, BarrierScope
from ayla.base.ir import Block, find_ops_of_type
from ayla.gpu_dialect import (AllocOp, AsyncOp, AwaitOp, KernelOp, LockType,
    MoveOp, gpu_dialect)
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.passes.schedule_warp_waits import ScheduleWarpWaits
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
import tests.test_warp_issuers as warp_cases


def module(*, release=True, tail='', target='sm_100a'):
    m=gpu_dialect.parse_module('''module attrs={target: "%s"} {
      kernel @visibility(%%g: gmem<128x64xf16>, %%h: gmem<64x64xf16>) block_dim=[128,1,1] {
        %%a: smem<128x64xf16> = alloc()
        %%b: smem<64x64xf16> = alloc()
        %%c: tmem<128x64xf32> = alloc()
        %%ready: lock<mbar> = alloc()
        %%done: lock<tcgen05> = alloc()
        async(%%ready: lock<mbar>) {
          move(%%a: smem<128x64xf16>, %%g: gmem<128x64xf16>) attrs={tma: True, issuer_warp: 1}
          move(%%b: smem<64x64xf16>, %%h: gmem<64x64xf16>) attrs={tma: True, issuer_warp: 1}
        }
        await(%%ready: lock<mbar>) attrs={waiter_warps: [0]}
        async(%%done: lock<tcgen05>) {
          matmul(%%a: smem<128x64xf16>, %%b: smem<64x64xf16>, %%c: tmem<128x64xf32>)
        }
        await(%%done: lock<tcgen05>)
        %s
      }
    }''' % (target,tail))
    asynchronous=find_ops_of_type(m,AsyncOp)
    asynchronous[0].attrs['issuer_warp']=1
    if release: asynchronous[1].attrs['release_prior']=True
    return m


def test_local_acquire_is_visible_only_to_the_waiting_warp():
    m=module();ready,done=find_ops_of_type(m,AwaitOp);a=SyncAnalysis();a.analyze(m)
    effects=[e for es in a.live_out(ready).concurrent_memory_effects.values() for e in es
             if e.lock is ready.lock and e.kind=="W"]
    assert effects and all(e.pending_await is None and e.visible_to==frozenset({0}) for e in effects)
    mma=find_ops_of_type(m,AsyncOp)[1]
    assert a.targets[mma].scope==BarrierScope.THREAD


@pytest.mark.parametrize('release',[False,True])
def test_completion_relay_requires_an_explicit_release_arrival(release):
    m=module(release=release);ready,done=find_ops_of_type(m,AwaitOp);a=SyncAnalysis();a.analyze(m)
    effects=[e for es in a.live_out(done).concurrent_memory_effects.values() for e in es
             if (e.lock is ready.lock and e.visible_to is not None) or e.observed_lock is ready.lock]
    assert bool(effects)==(not release)


@pytest.mark.parametrize('release', [False, True])
def test_narrowed_completion_relays_readiness_to_refill_only_with_release(release):
    from ayla.gpu_dialect import memory_effects, CollectiveScope
    m = module(release=release)
    copy, mma = find_ops_of_type(m, AsyncOp)
    ready, done = find_ops_of_type(m, AwaitOp)
    done.attrs['waiter_warps'] = [0, 1]
    refill = AsyncOp.build(copy.lock, collective=copy.collective,
        body=Block(ops=[MoveOp.build(move.dst, move.src,
            collective=move.collective, attrs=dict(move.attrs))
            for move in copy.body.ops]), attrs={'issuer_warp': 1})
    m.body.ops[0].body.ops.append(refill)
    analysis = SyncAnalysis()
    analysis.analyze(m)
    assert analysis.targets[refill].scope == (
        BarrierScope.THREAD if release else BarrierScope.BLOCK_TO_LEADER)
    observations = [effect for effects in
        analysis.live_out(done).concurrent_memory_effects.values()
        for effect in effects if effect.observed_lock is ready.lock]
    assert observations
    assert all(effect.visible_to == frozenset({0, 1} if release else {0})
        for effect in observations)
    # Event reads inherit the issuer instead of inventing a CTA reader.
    assert all(effect.scope == CollectiveScope.BLOCK_LEADER and effect.issuer_warp == 1
        for effect in memory_effects(refill) if effect.value is copy.lock)


def test_narrowed_release_does_not_publish_to_a_nonwaiting_refill_issuer():
    m = module()
    copy, _ = find_ops_of_type(m, AsyncOp)
    _, done = find_ops_of_type(m, AwaitOp)
    done.attrs['waiter_warps'] = [0, 1]
    refill = AsyncOp.build(copy.lock, collective=copy.collective,
        body=Block(ops=[MoveOp.build(move.dst, move.src,
            collective=move.collective, attrs={**move.attrs, 'issuer_warp': 2})
            for move in copy.body.ops]), attrs={'issuer_warp': 2})
    m.body.ops[0].body.ops.append(refill)
    analysis = SyncAnalysis()
    analysis.analyze(m)
    assert analysis.targets[refill].scope == BarrierScope.BLOCK_TO_LEADER


def test_acquire_in_one_warp_does_not_order_another_issuer():
    m=module(release=False);body=m.body.ops[0].body
    done_wait=find_ops_of_type(m,AwaitOp)[1];body.ops.remove(done_wait)
    mm=find_ops_of_type(m,AsyncOp)[1];mm.attrs['issuer_warp']=2
    for op in mm.body.ops:op.attrs['issuer_warp']=2
    a=SyncAnalysis();a.analyze(m)
    assert a.targets[mm].scope==BarrierScope.BLOCK_TO_LEADER


def test_release_cannot_publish_a_later_prefetch():
    m=module();body=m.body.ops[0].body;ready,done=find_ops_of_type(m,AwaitOp)
    original=find_ops_of_type(m,AsyncOp)[0]
    # Distinct event and buffer: its prefetch starts after the release issue.
    alloc=AllocOp.build_with_fresh_id(LockType.build(kind='mbar'))
    buffer=AllocOp.build_with_fresh_id(original.body.ops[0].dst.type)
    move=MoveOp.build(buffer.result(),original.body.ops[0].src,collective=original.body.ops[0].collective,attrs={'tma':True,'issuer_warp':1})
    copy=AsyncOp.build(alloc.result(),collective=original.collective,body=Block(ops=[]),attrs={'issuer_warp':1})
    copy.body.ops=[move]
    index=body.ops.index(done);body.ops[index:index]=[alloc,buffer,copy]
    a=SyncAnalysis();a.analyze(m)
    effects=a.live_out(done).concurrent_memory_effects[buffer.result()]
    assert any(e.lock is alloc.result() and e.pending_await is alloc.result() for e in effects)


@pytest.mark.parametrize('value',[[],[True],[-1],[4],[0,0],'0'])
def test_invalid_waiter_lists_are_rejected(value):
    m=blackwell_tcgen05_gemm_module(m=128,n=64,k=64,output_n_tile=64,operand_stages=1)
    find_ops_of_type(m,AwaitOp)[0].attrs['waiter_warps']=value
    with pytest.raises(ValueError,match='waiter_warps'):VerifyGpuDialect().run(m)


def test_hopper_rejects_warp_operand_waits():
    with pytest.raises(ValueError,match='SM100'):ScheduleWarpWaits().run(module(target='sm_90a'))


def test_scheduler_preserves_a_wait_used_by_non_mma_consumers():
    m=module();body=m.body.ops[0].body;copy=find_ops_of_type(m,AsyncOp)[0]
    source=copy.body.ops[0].dst
    body.ops.append(MoveOp.build(copy.body.ops[0].src,source,collective=copy.body.ops[0].collective))
    ready=find_ops_of_type(m,AwaitOp)[0];ready.attrs.clear()
    ScheduleWarpWaits().run(m)
    assert ready.waiter_warps is None


@pytest.mark.parametrize('release',[False,True])
def test_partial_wait_keeps_gather_live_until_a_publishing_completion(release):
    m=module(release=release)
    # Start at a new phase with the preceding generation's per-thread
    # observation. Its next producer must gather that observation.
    from ayla.analysis.sync import SyncState, ConcurrentMemoryEffect
    from ayla.gpu_dialect import CollectiveScope
    copy,mma=find_ops_of_type(m,AsyncOp);ready,done=find_ops_of_type(m,AwaitOp)
    state=SyncState(actor_tracking=True,concurrent_memory_effects={copy.lock:{
        ConcurrentMemoryEffect('R',CollectiveScope.THREAD,None,None)}})
    analysis=SyncAnalysis()
    state=analysis.transfer_async(copy,state)
    assert analysis.targets[copy].scope==BarrierScope.BLOCK_TO_LEADER
    state=analysis.transfer_await(ready,state)
    assert state.named_barrier_ids_pending_await.get(ready.lock)
    state=analysis.transfer_async(mma,state)
    state=analysis.transfer_await(done,state)
    assert bool(state.named_barrier_ids_pending_await.get(ready.lock))==(not release)


def test_fusion_preserves_scoped_wait_participants():
    from ayla.passes.fuse_async import FuseAsync
    m=module();body=m.body.ops[0].body
    ready=find_ops_of_type(m,AwaitOp)[0]
    FuseAsync().run(m)
    selected=[w for w in find_ops_of_type(m,AwaitOp) if w.lock.type.kind=='mbar']
    assert selected and selected[0].waiter_warps==(0,)


def test_opaque_consumers_prevent_operand_wait_specialization():
    from ayla.gpu_dialect import PTXTemplateOp
    m=module();ready=find_ops_of_type(m,AwaitOp)[0];ready.attrs.clear()
    m.body.ops[0].body.ops.append(PTXTemplateOp.build(operands=[],result_types=[],template='nop;'))
    ScheduleWarpWaits().run(m)
    assert ready.waiter_warps is None


@pytest.mark.parametrize('ending',['rendezvous','publishing_completion'])
def test_old_publication_cannot_retire_a_later_gather_reusing_its_id(ending):
    from ayla.analysis.sync import SyncState, _after_barrier
    m=module();ready,done=find_ops_of_type(m,AwaitOp)
    extra=AllocOp.build_with_fresh_id(LockType.build(kind='tcgen05'))
    other=AwaitOp.build(extra.result())
    future=AllocOp.build_with_fresh_id(LockType.build(kind='mbar'))
    key=(BarrierScope.BLOCK_TO_LEADER,1)
    state=SyncState(actor_tracking=True,
        named_barrier_ids_pending_await={ready.lock:{key}},
        gather_publications={done.lock:{key},other.lock:{key}})
    a=SyncAnalysis()
    if ending=='rendezvous':state=_after_barrier(state,BarrierScope.BLOCK,done)
    else:state=a.transfer_await(done,state)
    state.named_barrier_ids_pending_await[future.result()]={key}
    state=a.transfer_await(other,state)
    assert state.named_barrier_ids_pending_await[future.result()]=={key}


CASES=[*warp_cases.GPU_CASES,*[dict(warp_cases.GPU_CASES[i],load_warp=0,rhs_warp=0)
          for i in (0,4,5,7)],*[dict(warp_cases.GPU_CASES[i],reuse_tmem_allocation=False)
          for i in (0,5,11)]]
@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_warp_operand_waits(case,dtype):
    warp_cases.issuer_gemm(case,dtype,warp_specific_operand_waits=True,reuse_completion_locks=True)
