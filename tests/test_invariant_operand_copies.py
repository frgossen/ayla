import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AllocOp, AsyncOp, AwaitOp, ForOp, IndexModOp, MoveOp, SmemTensorType,TileOp,GmemTensorType,gpu_dialect
from ayla.passes.hoist_invariant_copies import HoistInvariantCopies, SimplifyIndexRemainders
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import LOOSE,check_gemm


def module(**kwargs):
    config=dict(m=4096,n=128,k=64,dtype='f16',output_n_tile=128,
        operand_k_tile=64,operand_stages=1,output_grid_size=3,
        output_grid_fast_axis='n',reuse_tmem_allocation=True)
    config.update(kwargs)
    return blackwell_tcgen05_gemm_module(**config)


@pytest.mark.parametrize('n,k,stages,tile,grid,expected',[
    (128,64,1,128,3,1),(512,64,1,512,3,2),
    (128,128,2,128,3,2),(128,256,4,128,3,4),
    (128,256,2,128,3,0),(768,128,2,256,3,2),
    (768,128,2,256,4,0),(256,64,1,128,4,1),
])
def test_motion_proof(n,k,stages,tile,grid,expected):
    m=module(n=n,k=k,operand_stages=stages,output_n_tile=tile,output_grid_size=grid)
    p=HoistInvariantCopies();p.run(m)
    assert p.hoisted_copies==expected
    make_ptx_kernel(m)  # SSA/physical resource/synchronization verification
    again=HoistInvariantCopies();again.run(m)
    assert again.hoisted_copies==0


@pytest.mark.parametrize('hazard',['source_write','overlap','escaped_read','reuse_group','out_of_bounds'])
def test_motion_rejects_hazards(hazard):
    m=module();groups=find_ops_of_type(m,AsyncOp)
    group=next(g for g in groups if g.lock.type.kind=='mbar')
    rhs=group.body.ops[-1];root=rhs.dst
    if hazard=='reuse_group':
        root.definition.attrs['reuse_group']='unsafe'
    elif hazard=='overlap':
        group.body.ops.append(MoveOp.build(dst=rhs.dst,src=rhs.src,collective=rhs.collective,attrs={'tma':True}))
    elif hazard=='source_write':
        group.parent_block.ops.append(MoveOp.build(dst=rhs.src,src=rhs.dst,collective=rhs.collective))
    elif hazard=='out_of_bounds':
        dst_type=SmemTensorType.build([64,64],root.type.dtype,swizzle=root.type.swizzle)
        view=TileOp.build(root,[2,0],dst_type)
        src=TileOp.build(rhs.src,[0,0],GmemTensorType.build([64,64],rhs.src.type.dtype))
        position=group.parent_block.ops.index(group)
        group.parent_block.ops[position:position]=[view,src]
        rhs.operands[0]=view.result()
        rhs.operands[1]=src.result()
    else:
        kernel=m.body.ops[0]
        sink=AllocOp.build_with_fresh_id(root.type)
        kernel.body.ops.extend([sink,MoveOp.build(dst=sink.result(),src=root,collective=rhs.collective)])
    p=HoistInvariantCopies();p.run(m)
    assert p.hoisted_copies==0


GPU_CASES=[
    dict(n=128,k=64),
    dict(n=512,k=64,output_n_tile=512,output_store='stmatrix',output_slices=2,output_store_buffers=2,tmem_writeback_register_budget=128),
    dict(n=128,k=128,operand_stages=2,cyclic_operand_pipeline=True),
    dict(n=256,k=256,output_n_tile=256,operand_stages=4,cyclic_operand_pipeline=True,tmem_writeback_warps=8,output_store='stmatrix',output_slices=2,output_store_buffers=2),
    dict(n=768,k=128,output_n_tile=256,operand_stages=2,cyclic_operand_pipeline=True,output_store='stmatrix'),
    dict(n=768,k=128,output_n_tile=256,operand_stages=2,cyclic_operand_pipeline=True,output_grid_size=4),
    dict(n=128,k=256,operand_stages=2,cyclic_operand_pipeline=True),
    dict(n=16,k=16,output_n_tile=16,operand_k_tile=16,output_m_tile=64,tmem_writeback_warps=8,output_layout='tcgen05_16x256b'),
    dict(n=8,k=16,output_n_tile=16,operand_k_tile=16,allow_partial_tiles=True,output_m_tile=64,tmem_writeback_warps=8,output_layout='tcgen05_16x256b'),
    dict(n=144,k=144,output_n_tile=144,operand_stages=3,cyclic_operand_pipeline=True,allow_partial_tiles=True),
    dict(n=128,k=64,output_m_tile=64,tmem_writeback_warps=8,output_store='stmatrix'),
    dict(n=128,k=128,operand_stages=2,cyclic_operand_pipeline=True,reuse_tmem_allocation=False),
    dict(m=128,n=4096,k=64,output_n_tile=128,output_store='stmatrix'),
    dict(m=256,n=128,k=64,output_grid_size=7),
]


@pytest.mark.parametrize('config',GPU_CASES)
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_gpu_preload(config,dtype):
    m=module(**config,dtype=dtype,hoist_invariant_operand_copies=True)
    # A preheader cache must refresh for each launch with new operands.
    check_gemm(make_ptx_launcher(make_ptx_kernel(m)),config.get('m',4096),config['n'],config['k'],
               dtype,mutate=True,runs=3,seed=4341,**LOOSE)


@pytest.mark.parametrize('stride,upper,expected',[(3,5,False),(6,5,False),(4,5,True),(1073741823,6,True),(-3,5,True)])
def test_remainder_respects_multiple_and_unsigned_overflow(stride,upper,expected):
    m=gpu_dialect.parse_module(f'''
    module {{ kernel @remainder(%a: gmem<64x1xf32>, %b: smem<1x1xf32>) grid_dim=[3,1,1] {{
      %bx: index = block_id.x
      %s: index = index_constant() attrs={{value: {stride}}}
      %n: index = index_constant() attrs={{value: 3}}
      for (%i: index = 0 to {upper}) {{
        %t: index = index_mul(%i: index, %s: index)
        %v: index = index_add(%t: index, %bx: index)
        %r: index = index_mod(%v: index, %n: index)
        %src: gmem<1x1xf32> = tile(%a: gmem<64x1xf32>, %r: index, 0)
        move(%b: smem<1x1xf32>, %src: gmem<1x1xf32>)
      }}
    }} }}''')
    SimplifyIndexRemainders().run(m)
    loop=find_ops_of_type(m,ForOp)[0]
    remainder=find_ops_of_type(m,IndexModOp)[0]
    def depends(value):
        if value is loop.iv:return True
        return hasattr(value.definition,'operands') and any(depends(v) for v in value.definition.operands)
    assert bool(depends(remainder.lhs))==expected
