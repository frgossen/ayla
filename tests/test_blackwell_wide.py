"""CTA tiles assembled from MMA subtiles through existing promotion passes."""
import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AllocOp,ElementwiseOp,ForOp,MatmulOp,MoveOp,TmemTensorType,root_buffers,gpu_dialect
from ayla.passes.folding import Folding
from ayla.passes.unroll_loop import UnrollLoop
from ayla.pipelines.canonicalize import Canonicalize
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import check_gemm


@pytest.mark.parametrize('n',[384,512])
def test_mma_subtiles_share_cta_operands_and_accumulator(n):
    mod=blackwell_tcgen05_gemm_module(m=128,n=n,k=128,output_n_tile=n,operand_stages=2,dtype='f16')
    [acc]=[op.result() for op in find_ops_of_type(mod,AllocOp) if isinstance(op.result().type,TmemTensorType)]
    assert list(acc.type.sizes)==[128,n] and acc.type.columns==512
    mmas=find_ops_of_type(mod,MatmulOp)
    assert len(mmas)==16 and all(op.rhs.type.sizes[0] <= 256 for op in mmas)
    assert {op.accumulate for op in mmas[:2]}=={False}
    assert all(op.accumulate for op in mmas[2:])
    assert len(set().union(*(root_buffers(op.lhs) for op in mmas)))==2
    assert len(set().union(*(root_buffers(op.rhs) for op in mmas)))==2
    assert all(root_buffers(op.dst)=={acc} for op in mmas)
    assert not any(isinstance(op.dst.type,TmemTensorType) for op in find_ops_of_type(mod,ElementwiseOp))
    moves=[op for op in find_ops_of_type(mod,MoveOp) if op.attrs.get('tma')]
    lhs=[op for op in moves if any(root.name=='a' for root in root_buffers(op.src))]
    rhs=[op for op in moves if any(root.name=='b' for root in root_buffers(op.src))]
    assert len(lhs)==2 and len(rhs)==4


def _fill_views(body, fill=0.):
    return gpu_dialect.parse_module(f'''module {{
      kernel @views(%a: gmem<4x8xf32>, %b: gmem<16x8xf32>, %c: gmem<4x32xf32>) {{
        %zero: rmem<f32> = constant {fill}
        elementwise fill(%c: gmem<4x32xf32>, %zero: rmem<f32>)
        %lo: gmem<4x16xf32 strides=[32,1]> = tile(%c: gmem<4x32xf32>,0,0)
        %hi: gmem<4x16xf32 strides=[32,1]> = tile(%c: gmem<4x32xf32>,0,1)
        {body}
      }}
    }}''')


def _mma(dst):
    return f'matmul(%a: gmem<4x8xf32>, %b: gmem<16x8xf32>, %{dst}: gmem<4x16xf32 strides=[32,1]>)'


@pytest.mark.parametrize('repeat',[False,True])
def test_fill_folding_preserves_partial_accumulations(repeat):
    mod=_fill_views(_mma('lo')+('\n'+_mma('lo') if repeat else '')+'\n'+_mma('hi'))
    Folding().run(mod)
    assert not find_ops_of_type(mod,ElementwiseOp)
    assert [op.accumulate for op in find_ops_of_type(mod,MatmulOp)]==([False,True,False] if repeat else [False,False])


@pytest.mark.parametrize('case',['uncovered','overlap','read','dynamic','alias','nonzero'])
def test_fill_folding_requires_complete_safe_coverage(case):
    body=_mma('lo')
    if case=='overlap':
        body+='''
        %all: gmem<4x32xf32> = tile(%c: gmem<4x32xf32>,0,0)
        %b32: gmem<32x8xf32> = alloc()
        matmul(%a: gmem<4x8xf32>,%b32: gmem<32x8xf32>,%all: gmem<4x32xf32>)'''
    elif case=='read':
        body+='''
        %out: gmem<4x16xf32> = alloc()
        move(%out: gmem<4x16xf32>,%lo: gmem<4x16xf32 strides=[32,1]>)\n'''+_mma('hi')
    elif case=='dynamic':
        body='''%i: index = block_id.x
        %v: gmem<4x16xf32 strides=[32,1]> = tile(%c: gmem<4x32xf32>,0,%i: index)
        matmul(%a: gmem<4x8xf32>,%b: gmem<16x8xf32>,%v: gmem<4x16xf32 strides=[32,1]>)'''
    elif case=='alias':
        body='''%alias: gmem<4x8xf32 strides=[32,1]> = tile(%c: gmem<4x32xf32>,0,0)
        matmul(%alias: gmem<4x8xf32 strides=[32,1]>,%b: gmem<16x8xf32>,%lo: gmem<4x16xf32 strides=[32,1]>)\n'''+_mma('hi')
    elif case=='nonzero':body+='\n'+_mma('hi')
    mod=_fill_views(body,fill=7. if case=='nonzero' else 0.)
    Folding().run(mod)
    assert len(find_ops_of_type(mod,ElementwiseOp))==1
    assert all(op.accumulate for op in find_ops_of_type(mod,MatmulOp))


def test_unroll_uses_loop_extent_when_no_factor_is_supplied():
    mod=gpu_dialect.parse_module('''module {
      kernel @unroll(%dst: gmem<4x3xf32>,%src: rmem<4x1xf32>) {
        for (%i: index = 0 to 3) attrs={label: "n"} {
          %view: gmem<4x1xf32 strides=[3,1]> = tile(%dst: gmem<4x3xf32>,0,%i: index)
          move(%view: gmem<4x1xf32 strides=[3,1]>,%src: rmem<4x1xf32>)
        }
      }
    }''')
    mod=Canonicalize().run(UnrollLoop(None,label_filter='n').run(mod))
    assert not find_ops_of_type(mod,ForOp)
    assert len(find_ops_of_type(mod,MoveOp))==3


@pytest.mark.parametrize('warps',[4,8])
def test_wide_nonzero_initializer_is_retained(warps):
    from ayla.pipelines.tile_matmul_pipeline import _schedule_matmul
    mod=gpu_dialect.parse_module('''module attrs={target: "sm_100a"} {
      kernel @nonzero(%a: gmem<128x64xbf16>,%b: gmem<512x64xbf16>,%c: gmem<128x512xf32>) {
        %init: rmem<f32> = constant 7.0
        matmul(%a: gmem<128x64xbf16>,%b: gmem<512x64xbf16>,%c: gmem<128x512xf32>,%init: rmem<f32>) collective=xyz.grid attrs={label: "mm"}
      }
    }''')
    mod=_schedule_matmul(mod,output_m_tile=128,output_n_tile=512,operand_stages=1,
        output_store='global',tmem_writeback_warps=warps,output_tma_inner_size=None)
    assert any(isinstance(op.dst.type,TmemTensorType) for op in find_ops_of_type(mod,ElementwiseOp))
    check_gemm(make_ptx_launcher(make_ptx_kernel(mod)),128,512,64,out_dtype='f32',bias=7,
               atol=2e-5,rtol=1e-6,runs=1)
