"""Reject malformed gather resources before PTX lowering."""
import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import AllocOp, BarrierOp, LockType
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from tests.test_completion_lifetimes import module, ISSUE, WAIT
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module


def resource(body, count):
    m=module(body)
    alloc=next(a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType))
    alloc.result().type.attrs['arrivals']=count
    return m,alloc


@pytest.mark.parametrize('count',[0,-1,True,'4',3])
def test_invalid_arrival_counts(count):
    m,alloc=resource('',count)
    alloc.parent_block.ops.append(BarrierOp.build('block_leader',alloc.result(),{'num_threads':128}))
    with pytest.raises(ValueError): VerifyGpuDialect().run(m)


def test_gather_is_not_an_async_completion_event():
    m,_=resource(ISSUE,4)
    with pytest.raises(ValueError,match='gather mbarrier'): VerifyGpuDialect().run(m)


def test_gather_is_not_an_ordinary_await_event():
    m,_=resource(WAIT,4)
    with pytest.raises(ValueError,match='gather mbarrier'): VerifyGpuDialect().run(m)


def test_invalid_backend():
    with pytest.raises(ValueError,match='block_gather_backend'):
        blackwell_tcgen05_gemm_module(block_gather_backend='unknown')



def test_repeated_single_issuer_events_have_explicit_retirement():
    from ayla.gpu_dialect import AsyncOp,DeallocOp,ForOp
    from ayla.base.ir import find_ancestor_of_type
    from ayla.passes.insert_syncs import InsertSyncs
    m=blackwell_tcgen05_gemm_module(m=4096,n=128,k=512,dtype='f16',
        output_n_tile=128,operand_k_tile=64,operand_stages=3,
        cyclic_operand_pipeline=True,output_grid_size=3)
    InsertSyncs().run(m)
    repeated=[a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,LockType)
              and find_ancestor_of_type(a,ForOp) is not None
              and any(isinstance(u,AsyncOp) for u in a.result().users())]
    assert repeated
    for a in repeated:
        releases=[u for u in a.result().users() if isinstance(u,DeallocOp)]
        assert len(releases)==1 and releases[0].parent_block is a.parent_block
        assert a.parent_block.ops.index(releases[0])>a.parent_block.ops.index(a)



def test_completion_reuse_chooses_a_safe_default_backend():
    m=blackwell_tcgen05_gemm_module(m=4096,n=128,k=512,dtype='f16',
        output_n_tile=128,operand_k_tile=64,operand_stages=3,
        cyclic_operand_pipeline=True,output_grid_size=3,reuse_completion_locks=True)
    assert m.body.ops[0].attrs['block_gather_backend']=='mbar'


def test_named_completion_reuse_rejects_repeated_tmem_allocation():
    with pytest.raises(ValueError,match='reusable TMEM'):
        blackwell_tcgen05_gemm_module(m=4096,n=128,k=512,dtype='f16',
            output_n_tile=128,operand_k_tile=64,operand_stages=3,
            cyclic_operand_pipeline=True,output_grid_size=3,
            reuse_completion_locks=True,block_gather_backend='named')



def test_tmem_repetition_includes_outer_loops():
    from ayla.gpu_dialect import gpu_dialect
    m=gpu_dialect.parse_module('''module attrs={target:"sm_100a"} {
      kernel @nested() block_dim=[128,1,1] {
        for (%i:index=0 to 3) {for (%j:index=0 to 1) {
          %a:tmem<128x64xf32>=alloc()
          dealloc(%a:tmem<128x64xf32>)
        }}
      }
    }''')
    assert m.body.ops[0].has_repeated_tmem_allocation
