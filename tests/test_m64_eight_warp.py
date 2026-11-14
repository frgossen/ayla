"""Column ownership for eight warps sharing a single M64 accumulator."""
import pytest
from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import MoveOp,TmemTensorType,WgrmemTensorType,gpu_dialect
from ayla.pipelines.tile_matmul_pipeline import _schedule_matmul
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from ayla.tmem_layout import TmemWarpLayout
from tests.testing_utils import LOOSE,check_gemm

@pytest.mark.parametrize('rows,warps',[(64,4),(64,8),(128,4),(128,8)])
@pytest.mark.parametrize('columns',[16,48,128,384,512])
def test_logical_and_physical_ownership(rows,warps,columns):
    layout=TmemWarpLayout(rows,warps)
    logical=[];physical=[]
    for w in range(warps):
        for r in range(layout.rows_per_warp):
            for c in range(layout.columns_per_warp(columns)):
                lr=layout.row_origin(w)+r;lc=layout.column_origin(w,columns)+c
                pr=layout.row_origin(w,physical=True)+r
                assert pr//32==w%4
                assert pr==(lr//16)*32+lr%16 if rows==64 else pr==lr
                logical.append((lr,lc));physical.append((pr,lc))
    assert len(logical)==rows*columns
    assert len(set(logical))==len(logical)==len(set(physical))
    assert set(logical)=={(r,c) for r in range(rows) for c in range(columns)}

@pytest.mark.parametrize('store',['global','stmatrix'])
def test_promotion_uses_single_budgeted_n512_fragment(store):
    mod=blackwell_tcgen05_gemm_module(m=128,n=512,k=64,output_m_tile=64,
        output_n_tile=512,operand_stages=1,tmem_writeback_warps=8,
        tmem_writeback_register_budget=128,output_store=store)
    loads=[op for op in find_ops_of_type(mod,MoveOp) if isinstance(op.src.type,TmemTensorType)]
    assert len(loads)==1 and loads[0].dst.type.sizes==[64,512]
    assert isinstance(loads[0].dst.type,WgrmemTensorType)
    assert loads[0].dst.type.num_elements_per_thread==128
    assert mod.body.ops[0].num_threads==256

@pytest.mark.parametrize('n,layout',[(8,None),(24,None),(40,None),(128,'tcgen05_32x32b')])
def test_unsupported_column_atoms_rejected(n,layout):
    with pytest.raises(ValueError):
        blackwell_tcgen05_gemm_module(m=128,n=n,k=64,output_m_tile=64,
            output_n_tile=n,operand_stages=1,tmem_writeback_warps=8,output_layout=layout)

@pytest.mark.parametrize('dtype',['f16','bf16'])
@pytest.mark.parametrize('n',[48,512])
def test_nonzero_fill_and_f32_writeback(dtype,n):
    m,k=128,1024
    mod=gpu_dialect.parse_module(f'''module attrs={{target: "sm_100a"}} {{
      kernel @fill(%a: gmem<{m}x{k}x{dtype}>,%b: gmem<{n}x{k}x{dtype}>,%c: gmem<{m}x{n}xf32>) {{
        %init: rmem<f32> = constant 7.0
        matmul(%a: gmem<{m}x{k}x{dtype}>,%b: gmem<{n}x{k}x{dtype}>,%c: gmem<{m}x{n}xf32>,%init: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
      }}
    }}''')
    mod=_schedule_matmul(mod,output_m_tile=64,output_n_tile=n,operand_stages=2,
        operand_k_tile=64,cyclic_operand_pipeline=True,output_store='global',
        tmem_writeback_warps=8,tmem_writeback_register_budget=128)
    check_gemm(make_ptx_launcher(make_ptx_kernel(mod)),m,n,k,dtype,out_dtype='f32',bias=7,
               atol=1e-3,rms=1e-5,runs=1,seed=6281)


@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_transposed_stmatrix_output(dtype):
    m,n,k=128,128,256
    mod=gpu_dialect.parse_module(f'''module attrs={{target: "sm_100a"}} {{
      kernel @transposed(%a: gmem<{m}x{k}x{dtype}>,%b: gmem<{n}x{k}x{dtype}>,%c: gmem<{m}x{n}x{dtype} strides=[1,{m}]>) {{
        %init: rmem<f32> = constant 0.0
        matmul(%a: gmem<{m}x{k}x{dtype}>,%b: gmem<{n}x{k}x{dtype}>,%c: gmem<{m}x{n}x{dtype} strides=[1,{m}]>,%init: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
      }}
    }}''')
    mod=_schedule_matmul(mod,output_m_tile=64,output_n_tile=128,operand_stages=2,
        operand_k_tile=64,cyclic_operand_pipeline=True,output_store='stmatrix',
        output_tma_inner_size=None,tmem_writeback_warps=8)
    check_gemm(make_ptx_launcher(make_ptx_kernel(mod)),m,n,k,dtype,runs=1,transpose_out=True,**LOOSE)
