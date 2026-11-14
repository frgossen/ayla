"""Eight-warp TMEM ownership, writeback bounds and event lifetime checks."""
import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import MoveOp, TmemTensorType, WgrmemTensorType, gpu_dialect
from ayla.pipelines.tile_matmul_pipeline import _schedule_matmul
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from ayla.tmem_layout import TmemWarpLayout
from tests.testing_utils import check_gemm


@pytest.mark.parametrize('rows,warps', [(64,4),(128,4),(128,8)])
def test_tmem_row_ownership_is_disjoint_and_complete(rows, warps):
    ownership = TmemWarpLayout(rows, warps)
    logical = [ownership.row_origin(w) + r for w in range(warps)
               for r in range(ownership.rows_per_warp)]
    assert sorted(logical) == list(range(rows))
    physical = [ownership.row_origin(w,physical=True) + r for w in range(warps)
                for r in range(ownership.rows_per_warp)]
    assert len(set(physical)) == rows
    for w in range(warps):
        first = ownership.row_origin(w,physical=True)
        assert first // 32 == w % 4
        assert (first + ownership.rows_per_warp - 1) // 32 == w % 4


@pytest.mark.parametrize('warps', [4,8])
def test_promotion_accounts_for_all_writeback_threads(warps):
    mod = blackwell_tcgen05_gemm_module(m=128,n=256,k=64,operand_stages=1,
                                       tmem_writeback_warps=warps)
    loads = [op for op in find_ops_of_type(mod,MoveOp)
             if isinstance(op.src.type,TmemTensorType)]
    assert loads and all(op.dst.type.warps == warps for op in loads)
    assert all(op.dst.type.num_elements_per_thread <= 32 for op in loads)
    assert sum(op.dst.type.num_elements for op in loads) == 128 * 256
    assert mod.body.ops[0].num_threads == warps * 32


@pytest.mark.parametrize('config', [dict(tmem_writeback_warps=6),
    dict(tmem_writeback_warps=8,output_layout='tcgen05_32x32b')])
def test_unsupported_copy_ownership_is_rejected(config):
    with pytest.raises(ValueError):
        blackwell_tcgen05_gemm_module(m=128,n=64,k=64,operand_stages=1,**config)


@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_eight_warp_nonzero_tmem_fill(dtype):
    mod = gpu_dialect.parse_module(f'''module attrs={{target: "sm_100a"}} {{
      kernel @fill_{dtype}(%a: gmem<128x64x{dtype}>, %b: gmem<64x64x{dtype}>, %c: gmem<128x64xf32>) {{
        %init: rmem<f32> = constant 7.0
        matmul(%a: gmem<128x64x{dtype}>, %b: gmem<64x64x{dtype}>, %c: gmem<128x64xf32>, %init: rmem<f32>) collective=xyz.grid attrs={{label: "mm"}}
      }}
    }}''')
    mod = _schedule_matmul(mod,output_m_tile=128,output_n_tile=64,operand_stages=1,
        operand_k_tile=64,output_store='global',output_tma_inner_size=None,tmem_writeback_warps=8)
    check_gemm(make_ptx_launcher(make_ptx_kernel(mod)),128,64,64,dtype,out_dtype='f32',bias=7,
               atol=2e-5,rtol=1e-6,runs=1)


def test_completion_event_reuse_gathers_prior_waiters():
    from ayla.analysis.sync import SyncAnalysis,BarrierScope
    from ayla.gpu_dialect import AsyncOp
    mod=gpu_dialect.parse_module('''module attrs={target: "sm_100a"} {
      kernel @phase(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<64x16xbf16 swizzle="128B">) block_dim=[256,1,1] {
        %dst: tmem<128x64xf32> = alloc()
        %event: lock<tcgen05> = alloc()
        async(%event: lock<tcgen05>) matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<64x16xbf16 swizzle="128B">, %dst: tmem<128x64xf32>) collective=xyz.block
        await(%event: lock<tcgen05>)
        async(%event: lock<tcgen05>) matmul(%a: smem<128x16xbf16 swizzle="128B">, %b: smem<64x16xbf16 swizzle="128B">, %dst: tmem<128x64xf32>) collective=xyz.block
        await(%event: lock<tcgen05>)
        dealloc(%dst: tmem<128x64xf32>)
      }
    }''')
    analysis=SyncAnalysis();analysis.analyze(mod)
    second=find_ops_of_type(mod,AsyncOp)[1]
    assert analysis.targets[second].scope==BarrierScope.BLOCK_TO_LEADER
