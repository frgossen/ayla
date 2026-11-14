import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (gpu_dialect, AsyncOp, MoveOp, CollectiveScope,
                              GmemTensorType, SmemTensorType, root_buffer)
from ayla.passes.eliminate_redundant_copies import EliminateRedundantCopies
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
@pytest.mark.parametrize('scoped,forced_alias',[(False,False),(True,False),(False,True)])
def test_copy_reuse_requires_shared_visibility_and_unaliased_storage(scoped,forced_alias):
    module=gpu_dialect.parse_module('''module {kernel @copy(%g: gmem<128x64xf16>, %h: gmem<128x64xf16>) {
      %s: smem<128x64xf16> = alloc()
      %t: smem<128x64xf16> = alloc()
      %l0: lock<mbar> = alloc()
      %l1: lock<mbar> = alloc()
      async(%l0: lock<mbar>) {
        move(%s: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={tma: True}
        move(%t: smem<128x64xf16>, %h: gmem<128x64xf16>) attrs={tma: True}
      }
      await(%l0: lock<mbar>)
      async(%l1: lock<mbar>) {
        move(%s: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={tma: True}
        move(%t: smem<128x64xf16>, %h: gmem<128x64xf16>) attrs={tma: True}
      }
      await(%l1: lock<mbar>)
    }}''')
    from ayla.gpu_dialect import AllocOp,AwaitOp
    if scoped:find_ops_of_type(module,AwaitOp)[0].attrs['waiter_warps']=[0]
    if forced_alias:find_ops_of_type(module,AllocOp)[0].attrs['reuse_group']=0
    issue=find_ops_of_type(module,AsyncOp)[1];original=issue.body.ops[0]
    EliminateRedundantCopies().run(module)
    assert (original in issue.body.ops)==(scoped or forced_alias)


@pytest.mark.parametrize('partial,write',[(True,False),(False,True)])
def test_copy_reuse_discards_partial_completions_and_destination_writes(partial,write):
    module=gpu_dialect.parse_module('''module {kernel @copy(%g: gmem<128x64xf16>, %h: gmem<128x64xf16>) {
      %s: smem<128x64xf16> = alloc()
      %t: smem<128x64xf16> = alloc()
      %l0: lock<mbar> = alloc()
      %l1: lock<mbar> = alloc()
      async(%l0: lock<mbar>) {
        move(%s: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={tma: True}
        move(%t: smem<128x64xf16>, %h: gmem<128x64xf16>) attrs={tma: True}
      }
      await(%l0: lock<mbar>)
      async(%l1: lock<mbar>) {
        move(%s: smem<128x64xf16>, %g: gmem<128x64xf16>) attrs={tma: True}
        move(%t: smem<128x64xf16>, %h: gmem<128x64xf16>) attrs={tma: True}
      }
      await(%l1: lock<mbar>)
    }}''')
    from ayla.gpu_dialect import AwaitOp
    issues=find_ops_of_type(module,AsyncOp)
    original=issues[1].body.ops[0]
    if partial:find_ops_of_type(module,AwaitOp)[0].attrs['pending']=1
    if write:
        block=issues[1].parent_block
        block.ops.insert(block.ops.index(issues[1]),MoveOp.build(dst=original.dst,src=module.body.ops[0].body.arguments[1],collective=original.attrs["collective"]))
    EliminateRedundantCopies().run(module)
    assert original in issues[1].body.ops

@pytest.mark.parametrize('reuse,barriers',[(False,False),(True,False),(True,True)])
def test_completed_b_panel_copy_count(reuse,barriers):
    module=blackwell_tcgen05_gemm_module(m=1024,n=128,k=64,dtype='f16',
        output_n_tile=128,operand_stages=1,output_m_tiles_per_cta=4,
        output_grid_fast_axis='n',output_store='stmatrix',
        reuse_tmem_allocation=True,reuse_completion_locks=True,
        reuse_completed_operand_copies=reuse,reuse_operand_copies_across_barriers=barriers,warp_specific_operand_waits=True)
    kernel=module.body.ops[0]
    copies=[op for op in find_ops_of_type(module,MoveOp)
            if isinstance(op.src.type,GmemTensorType) and isinstance(op.dst.type,SmemTensorType)]
    assert sum(root_buffer(op.src) is kernel.body.arguments[1] for op in copies)==(1 if barriers else 3 if reuse else 4)


def test_barrier_reuse_requires_completed_copy_reuse():
    with pytest.raises(ValueError,match="requires completed"):
        blackwell_tcgen05_gemm_module(reuse_operand_copies_across_barriers=True)
