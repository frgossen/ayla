"""Balanced event lifetimes, event reuse and phase carry across persistent GEMM tiles."""
import pytest
from ayla.analysis.sync import BarrierScope, SyncAnalysis
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import AllocOp, AsyncOp, AwaitOp, ForOp, KernelOp, LockType, gpu_dialect
from ayla.passes.licm import LICM
import tests.test_warp_issuers as warp_cases


def module(body, target='sm_100a'):
    return gpu_dialect.parse_module('''module attrs={target: "%s"} {
      kernel @events(%%g: gmem<128x64xf16>) block_dim=[128,1,1] {
        %%a: smem<128x64xf16> = alloc()
        for (%%i: index = 0 to 3) {
          %%three: index = index_constant() attrs={value: 3}
          %%valid: pred = index_lt(%%i: index, %%three: index)
          if(%%valid: pred) {
            %%l: lock<mbar> = alloc()
            %s
          }
        }
      }
    }''' % (target, body))

ISSUE='async(%l: lock<mbar>) move(%a: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={tma: True}'
WAIT='await(%l: lock<mbar>)'


def test_balanced_events_cross_loop_and_default_is_unchanged():
    m=module(ISSUE+'\n'+WAIT)
    before=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType)).parent_block
    assert next(a for a in find_ops_of_type(LICM().run(m),AllocOp) if isinstance(a.result().type,LockType)).parent_block is before
    m=LICM(reuse_completion_locks=True).run(m)
    lock=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType))
    assert not isinstance(lock.parent_block.parent_op,KernelOp)
    m=LICM(reuse_completion_locks=True).run(m)
    assert isinstance(lock.parent_block.parent_op,KernelOp)
    assert len(find_ops_of_type(m,AsyncOp))==len(find_ops_of_type(m,AwaitOp))==1


@pytest.mark.parametrize('body',[ISSUE,WAIT+'\n'+ISSUE])
def test_unbalanced_or_unissued_phases_stay_inside(body):
    m=module(body);m=LICM(reuse_completion_locks=True).run(m)
    lock=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType))
    assert not isinstance(lock.parent_block.parent_op,KernelOp)


def test_balanced_pipeline_can_enter_inner_loop_with_pending_phase():
    m=module(ISSUE+'''\nfor (%k: index = 0 to 4) {'''+WAIT+'\n'+ISSUE+'}\n'+WAIT)
    m=LICM(reuse_completion_locks=True).run(m)
    lock=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType))
    assert not isinstance(lock.parent_block.parent_op,KernelOp)
    m=LICM(reuse_completion_locks=True).run(m)
    assert isinstance(lock.parent_block.parent_op,KernelOp)


@pytest.mark.parametrize('target',['sm_90a','sm_100a'])
def test_target_and_nonuniform_guard_proofs(target):
    m=module('''%tid: index = thread_id.x
        %one: index = index_constant() attrs={value: 1}
        %p: pred = index_lt(%tid: index, %one: index)
        if(%p: pred) {'''+ISSUE+'\n'+WAIT+'}',target)
    m=LICM(reuse_completion_locks=True).run(m)
    lock=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType))
    assert not isinstance(lock.parent_block.parent_op,KernelOp)


def test_uniform_guard_moves_event_without_moving_observers():
    m=module('''%four: index = index_constant() attrs={value: 4}
        %p: pred = index_lt(%i: index, %four: index)
        if(%p: pred) {'''+ISSUE+'\n'+WAIT+'}')
    m=LICM(reuse_completion_locks=True).run(m)
    lock=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType))
    assert not isinstance(lock.parent_block.parent_op,KernelOp)
    m=LICM(reuse_completion_locks=True).run(m)
    assert isinstance(lock.parent_block.parent_op,KernelOp)
    assert len(find_ops_of_type(m,AwaitOp))==1


CASES=[*warp_cases.GPU_CASES,*[dict(warp_cases.GPU_CASES[i],load_warp=0,rhs_warp=0)
                                for i in (0,4,5,7)],*[dict(warp_cases.GPU_CASES[i],reuse_tmem_allocation=False) for i in (0,5,11)]]
@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_phase_carry(case,dtype):
    warp_cases.issuer_gemm(case,dtype,reuse_completion_locks=True,block_gather_backend="mbar")


@pytest.mark.parametrize('case',CASES[20:])
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_gather_with_per_tile_completion_events(case,dtype):
    warp_cases.issuer_gemm(case,dtype,block_gather_backend="mbar")


@pytest.mark.parametrize('case',[dict(c,load_warp=0,rhs_warp=0) for c in CASES[20:]])
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_single_issuer_gather_with_per_tile_events(case,dtype):
    warp_cases.issuer_gemm(case,dtype,block_gather_backend="mbar")


@pytest.mark.parametrize('case',CASES[20:])
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_completion_reuse_default_backend(case,dtype):
    warp_cases.issuer_gemm(case,dtype,reuse_completion_locks=True)


@pytest.mark.parametrize('case',[dict(c,load_warp=0,rhs_warp=0) for c in CASES[20:]])
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_default_repeated_tmem_and_event_lifetimes(case,dtype):
    warp_cases.issuer_gemm(case,dtype)


# Named completion gathers with a verified outer TMEM lifetime.
NAMED_CASES=[*([warp_cases.GPU_CASES[i] for i in (0,5,11)]),
             *([dict(warp_cases.GPU_CASES[i],load_warp=0,rhs_warp=0) for i in (0,5,11)])]

@pytest.mark.parametrize('case',NAMED_CASES)
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_named_phase_with_reusable_tmem(case,dtype):
    warp_cases.issuer_gemm(case,dtype,reuse_completion_locks=True,block_gather_backend='named')


# Event reuse must gather all waiters despite intervening async batches.
@pytest.mark.parametrize('group,expected',[(1,BarrierScope.BLOCK_TO_LEADER)])
def test_other_batch_does_not_erase_previous_event_observations(group,expected):
 module=gpu_dialect.parse_module(f'''
 module {{
  kernel @events() block_dim=[128, 1, 1] cluster_dim=[{group}, 1, 1] {{
   %a: smem<128x64xf16> = alloc()
   %b: smem<64x64xf16> = alloc()
   %c: tmem<128x64xf32 attrs={{cta_group: {group}}}> = alloc()
   %l0: lock<tcgen05 attrs={{cta_group: {group}}}> = alloc()
   %l1: lock<tcgen05 attrs={{cta_group: {group}}}> = alloc()
   async(%l0: lock<tcgen05 attrs={{cta_group: {group}}}>) matmul(%a: smem<128x64xf16>, %b: smem<64x64xf16>, %c: tmem<128x64xf32 attrs={{cta_group: {group}}}>)
   await(%l0: lock<tcgen05 attrs={{cta_group: {group}}}>)
   async(%l1: lock<tcgen05 attrs={{cta_group: {group}}}>) matmul(%a: smem<128x64xf16>, %b: smem<64x64xf16>, %c: tmem<128x64xf32 attrs={{cta_group: {group}}}>)
   await(%l1: lock<tcgen05 attrs={{cta_group: {group}}}>)
   async(%l0: lock<tcgen05 attrs={{cta_group: {group}}}>) matmul(%a: smem<128x64xf16>, %b: smem<64x64xf16>, %c: tmem<128x64xf32 attrs={{cta_group: {group}}}>)
  }}
 }}''')
 analysis=SyncAnalysis();analysis.analyze(module)
 batches=find_ops_of_type(module,AsyncOp)
 assert analysis.targets[batches[-1]].scope==expected
