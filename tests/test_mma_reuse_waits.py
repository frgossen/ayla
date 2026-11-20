"""Consumer waits, event generation handoffs and unchanged shared schedules."""
import pytest
from ayla.base.ir import Block,find_ops_of_type
from ayla.gpu_dialect import AllocOp,AsyncOp,AwaitOp,BarrierOp,ForOp,LockType,MoveOp,PTXTemplateOp
from ayla.passes.schedule_warp_waits import ScheduleWarpWaits
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.analysis.sync import SyncAnalysis,BarrierScope
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from tests.test_blackwell_pipeline import _module,MMA0,MMA1,LOAD,READ

def sample():
    return _module(f'{MMA0}\nawait(%l0: lock<tcgen05>)\n{LOAD}\n'
        f'await(%load: lock<mbar>)\n{MMA1}\nawait(%l1: lock<tcgen05>)\n{READ}')

@pytest.mark.parametrize('issuer',[0,1,2])
def test_wait_follows_actual_refill_consumer_without_changing_issuers(issuer):
    m=sample();mma,refill,next_mma=find_ops_of_type(m,AsyncOp)
    for member in refill.body.ops:member.attrs['issuer_warp']=issuer
    initial=[dict(op.attrs) for op in find_ops_of_type(m,AsyncOp)]
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    first,ready,final=find_ops_of_type(m,AwaitOp)
    assert first.waiter_warps==tuple(sorted({0,issuer}))
    assert ready.waiter_warps is None and final.waiter_warps is None
    assert [dict(op.attrs) for op in find_ops_of_type(m,AsyncOp)]==initial
    body=m.body.ops[0].body;index=body.ops.index(final)
    assert isinstance(body.ops[index-1],BarrierOp)
    assert body.ops[index-1].attrs['mma_wait_handoff']

def test_refill_can_acquire_completion_without_collective_gather():
    m=sample();ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    mma,refill,next_mma=find_ops_of_type(m,AsyncOp)
    analysis=SyncAnalysis();analysis.analyze(m)
    assert analysis.targets[refill].scope==BarrierScope.THREAD

def test_all_bank_refill_consumers_observe_the_event():
    m=sample();first,refill,second=find_ops_of_type(m,AsyncOp)
    first.body.ops[0].operands[1]=second.body.ops[0].lhs
    refill.body.ops[0].attrs['issuer_warp']=1
    lock=AllocOp.build_with_fresh_id(LockType.build(kind='mbar'))
    move=MoveOp.build(second.body.ops[0].lhs,refill.body.ops[0].src,
        collective=refill.collective,attrs={'tma':True,'issuer_warp':2})
    copy=AsyncOp.build(lock.result(),collective=refill.collective,
        body=Block(ops=[move]))
    body=m.body.ops[0].body;i=body.ops.index(refill)+1
    body.ops[i:i]=[lock,copy,AwaitOp.build(lock.result())]
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert find_ops_of_type(m,AwaitOp)[0].waiter_warps==(0,1,2)

def test_collective_accumulator_access_preserves_its_wait():
    m=_module(f'{MMA0}\nawait(%l0: lock<tcgen05>)\n{READ}\n{LOAD}')
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert find_ops_of_type(m,AwaitOp)[0].waiter_warps is None

def test_other_mma_bank_reader_also_observes_event_generations():
    m=sample();first,refill,second=find_ops_of_type(m,AsyncOp)
    first.body.ops[0].operands[1]=second.body.ops[0].lhs
    second.body.ops[0].attrs['issuer_warp']=1
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert find_ops_of_type(m,AwaitOp)[0].waiter_warps==(0,1)

def test_synchronous_smem_observer_prevents_narrowing_even_after_refill():
    m=sample();load=find_ops_of_type(m,AsyncOp)[1].body.ops[0]
    m.body.ops[0].body.ops.append(MoveOp.build(load.src,load.dst,collective=load.collective))
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert find_ops_of_type(m,AwaitOp)[0].waiter_warps is None

def test_opaque_effects_prevent_narrowing():
    m=sample();m.body.ops[0].body.ops.append(PTXTemplateOp.build(operands=[],result_types=[],template='nop;'))
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert find_ops_of_type(m,AwaitOp)[0].waiter_warps is None

def test_different_mma_issuer_preserves_wait_before_ordered_updates():
    m=sample();body=m.body.ops[0].body
    first,load,second=find_ops_of_type(m,AsyncOp)
    body.ops.remove(second);body.ops.insert(body.ops.index(load),second)
    for member in second.body.ops:member.attrs['issuer_warp']=1
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert find_ops_of_type(m,AwaitOp)[0].waiter_warps is None

def test_existing_cta_barrier_supplies_generation_handoff():
    m=sample();body=m.body.ops[0].body;final=find_ops_of_type(m,AwaitOp)[2]
    body.ops.insert(body.ops.index(final),BarrierOp.build(attrs={'scope':'block'}))
    ScheduleWarpWaits(operand_ready=False,mma_reuse=True).run(m)
    assert not any(b.attrs.get('mma_wait_handoff') for b in find_ops_of_type(m,BarrierOp))

def test_schedule_is_idempotent():
    m=sample();p=ScheduleWarpWaits(operand_ready=False,mma_reuse=True)
    p.run(m);first=str(m);p.run(m);assert str(m)==first

@pytest.mark.parametrize('mma_reuse',[False,True])
def test_mma_reuse_requires_resolved_cta_without_changing_operand_ready_api(mma_reuse):
    m=sample();m.body.ops[0].attrs['block_dim']=[None,1,1]
    p=ScheduleWarpWaits(mma_reuse=mma_reuse)
    if mma_reuse:
        with pytest.raises(ValueError,match='resolved whole-warp CTA'):p.run(m)
    else:
        p.run(m)

@pytest.mark.parametrize('stages',[2,3,8])
def test_shared_pipeline_drains_collectively_after_scoped_loop_waits(stages):
    args=dict(m=1536,n=512,k=16448,dtype='f16',output_n_tile=64,
        operand_k_tile=64,operand_stages=stages,cyclic_operand_pipeline=True,
        output_store='stmatrix',allow_partial_tiles=True)
    plain=blackwell_tcgen05_gemm_module(**args)
    m=ScheduleWarpWaits(mma_reuse=True).run(blackwell_tcgen05_gemm_module(**args))
    VerifyGpuDialect().run(m)
    assert [l.trip_count for l in find_ops_of_type(m,ForOp)]==[l.trip_count for l in find_ops_of_type(plain,ForOp)]
    assert m.body.ops[0].block_dim==plain.body.ops[0].block_dim
    assert any(w.waiter_warps==(0,) and w.lock.type.kind=='tcgen05' for w in find_ops_of_type(m,AwaitOp))
    final=[w for w in find_ops_of_type(m,AwaitOp) if w.lock.type.kind=='tcgen05'][-1]
    assert final.waiter_warps is None
    assert any(b.attrs.get('mma_wait_handoff') for b in find_ops_of_type(m,BarrierOp))
