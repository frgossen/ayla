"""Loop pipelining uses the same proven TMEM regions as async scheduling."""
import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import AsyncOp, BarrierOp, ForOp, IfOp, MoveOp, gpu_dialect
from ayla.passes.pipeline_loop import PipelineLoop
from ayla.passes.schedule_async import _can_swap
from tests.testing_utils import assert_dominance


def module(column=1, tracked=True, dynamic=False):
    text='''module attrs={target:"sm_100a"} {
      kernel @regions(%a:smem<128x16xf16>, %r:rmem<128x128xf32>, %j:index)
        block_dim=[128,1,1] {
        %arena:tmem<128x256xf32> = alloc()
        %c0:tmem<128x128xf32 columns=256> = tile(%arena:tmem<128x256xf32>,0,0)
        %c1:tmem<128x128xf32 columns=256> = tile(%arena:tmem<128x256xf32>,0,COLUMN)
        %l:lock<tcgen05> = alloc()
        %zero:index = index_constant() attrs={value:0}
        %one:index = index_constant() attrs={value:1}
        %p:pred = index_lt(%zero:index,%one:index)
        for (%i:index = 0 to 4) {
          move(%r:rmem<128x128xf32>,%c0:tmem<128x128xf32 columns=256>)
          if(%p:pred) {
            async(%l:lock<tcgen05>) matmul(%a:smem<128x16xf16>,%a:smem<128x16xf16>,%c1:tmem<128x128xf32 columns=256>)
          }
          await(%l:lock<tcgen05>)
        }
      }}'''.replace('COLUMN','%j:index' if dynamic else str(column))
    mod=gpu_dialect.parse_module(text)
    mod.body.ops[0].body.ops[0].attrs['track_column_regions']=tracked
    find_ops_of_type(mod,ForOp)[0].attrs['label']='jobs'
    find_ops_of_type(mod,IfOp)[0].attrs['label']='next_compute'
    return mod


@pytest.mark.parametrize('column,tracked,dynamic,legal',[
    (1,True,False,True),(0,True,False,False),
    (1,False,False,False),(1,True,True,False)])
def test_pipeline_and_async_scheduler_agree_on_tmem_aliases(column,tracked,dynamic,legal):
    mod=module(column,tracked,dynamic)
    loop=find_ops_of_type(mod,ForOp)[0]
    read,compute=loop.body.ops[:2]
    pipeline=PipelineLoop('jobs','next_compute',1)
    assert pipeline._can_reorder(read,compute)==legal
    assert _can_swap(read,compute)==legal
    if legal:
        pipeline.run(mod)
        assert_dominance(mod.body)
        assert len(find_ops_of_type(mod,AsyncOp))==2  # Prologue and loop; the drain retains its observer.
    else:
        with pytest.raises(ValueError,match='cannot be proven independent'):
            pipeline.run(mod)


def test_region_proof_does_not_remove_barrier_ordering():
    mod=module();loop=find_ops_of_type(mod,ForOp)[0]
    loop.body.ops.insert(0,BarrierOp.build())
    with pytest.raises(ValueError,match='cannot be proven independent'):
        PipelineLoop('jobs','next_compute',1).run(mod)
