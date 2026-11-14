"""TMEM arena packing, static region proofs, allocation permits and lifetimes."""
import pytest
from ayla.analysis.sync import BarrierScope, SyncAnalysis
from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AsyncOp, BarrierOp, CollectiveScope, DeallocOp, ElementwiseOp, MemoryEffect, MoveOp,
    PTXTemplateOp, TileOp, TmemTensorType, gpu_dialect, memory_effects_may_alias,
    static_tmem_columns,
)
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.passes.pack_tmem_allocs import PackTmemAllocs
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.ptx_codegen import make_ptx_kernel


def test_grouped_capacity_is_rejected_before_launch():
    with pytest.raises(ValueError,match='512 columns'):
        blackwell_tcgen05_gemm_module(m=1024,n=128,k=64,dtype='f16',
            output_n_tile=128,operand_stages=1,output_m_tiles_per_cta=8,
            output_grid_fast_axis='n')


def test_packing_requires_matching_releases():
    module=gpu_dialect.parse_module('''module { kernel @missing_free() {
      %a: tmem<128x64xf32> = alloc()
      %b: tmem<128x64xf32> = alloc()
      dealloc(%b: tmem<128x64xf32>)
    }}''')
    with pytest.raises(ValueError,match='nonescaping lifetime'):
        PackTmemAllocs().run(module)


def test_packing_rejects_mismatched_shapes():
    module=gpu_dialect.parse_module('''module { kernel @mismatch() {
      %a: tmem<128x64xf32> = alloc()
      %b: tmem<128x128xf32> = alloc()
      dealloc(%a: tmem<128x64xf32>)
      dealloc(%b: tmem<128x128xf32>)
    }}''')
    with pytest.raises(ValueError,match='equal-shaped'):
        PackTmemAllocs().run(module)


def test_padded_column_views_lower_to_one_allocation():
    module=blackwell_tcgen05_gemm_module(m=1024,n=192,k=128,dtype='f16',
        output_n_tile=192,operand_stages=1,output_m_tiles_per_cta=2,
        output_grid_fast_axis='n',output_grid_size=3,output_store='stmatrix',
        reuse_tmem_allocation=True,reuse_completion_locks=True)
    ptx=make_ptx_kernel(module).ptx
    assert ptx.count('tcgen05.alloc.cta_group::1')==1
    assert ptx.count('tcgen05.dealloc.cta_group::1')==1


# Static arena proofs must preserve overlapping and unknown hazards.
def test_padded_nested_columns_and_overlapping_views():
    mod=gpu_dialect.parse_module('''module {kernel @regions(%i: index) {
      %arena: tmem<128x384xf32 columns=512> = alloc() attrs={track_column_regions: True}
      %a: tmem<128x192xf32 columns=512> = tile(%arena: tmem<128x384xf32 columns=512>, 0, 0)
      %b: tmem<128x192xf32 columns=512> = tile(%arena: tmem<128x384xf32 columns=512>, 0, 1)
      %slice: tmem<128x32xf32 columns=512> = tile(%b: tmem<128x192xf32 columns=512>, 0, 2)
      %overlap: tmem<128x128xf32 columns=512> = tile(%arena: tmem<128x384xf32 columns=512>, 0, 1)
      %dynamic: tmem<128x192xf32 columns=512> = tile(%arena: tmem<128x384xf32 columns=512>, 0, %i: index)
      %rows: tmem<64x192xf32 columns=512> = tile(%arena: tmem<128x384xf32 columns=512>, 0, 0)
    }}''')
    views=[op.result() for op in find_ops_of_type(mod,TileOp)]
    assert [static_tmem_columns(v) for v in views]==[
        (0,192),(192,384),(256,288),(128,256),None,None]
    effects=[MemoryEffect('W',v,CollectiveScope.BLOCK) for v in views]
    assert not memory_effects_may_alias(effects[0],effects[1])
    assert memory_effects_may_alias(effects[1],effects[3])
    assert memory_effects_may_alias(effects[0],effects[4])
    assert memory_effects_may_alias(effects[0],effects[5])


@pytest.mark.parametrize('disjoint',[False,True])
def test_sync_retains_overlapping_view_hazard(disjoint):
    column=1 if disjoint else 0
    mod=gpu_dialect.parse_module('''module {kernel @hazard(
      %a: smem<128x16xbf16>, %b: smem<128x16xbf16>,
      %r: wgrmem<128x128xf32 warps=4 layout="tcgen05_16x32bx2">) {
      %arena: tmem<128x256xf32> = alloc() attrs={track_column_regions: True}
      %l: lock<tcgen05> = alloc()
      %v0: tmem<128x128xf32 columns=256> = tile(%arena: tmem<128x256xf32>, 0, 0)
      %v1: tmem<128x128xf32 columns=256> = tile(%arena: tmem<128x256xf32>, 0, '''+str(column)+''')
      move(%r: wgrmem<128x128xf32 warps=4 layout="tcgen05_16x32bx2">, %v0: tmem<128x128xf32 columns=256>)
      async(%l: lock<tcgen05>) matmul(%a: smem<128x16xbf16>, %b: smem<128x16xbf16>, %v1: tmem<128x128xf32 columns=256>)
      await(%l: lock<tcgen05>)
    }}''')
    analysis=SyncAnalysis();analysis.analyze(mod)
    issue=find_ops_of_type(mod,AsyncOp)[0]
    assert analysis.targets[issue].scope==(BarrierScope.THREAD if disjoint else BarrierScope.BLOCK)


# Release allocation permits only after a proven final TMEM allocation.
def lower(body):
    module = gpu_dialect.parse_module('''module attrs={target: "sm_100a"} {
      kernel @allocation() block_dim=[128,1,1] {
        %s
      }
    }''' % body)
    lower_to_ptx_template(module)
    return [op.template for op in find_ops_of_type(module, PTXTemplateOp)]


def test_single_unconditional_arena_releases_before_writeback():
    templates = lower('''%a:tmem<128x64xf32>=alloc()
        dealloc(%a:tmem<128x64xf32>)''')
    allocation = next(t for t in templates if 'tcgen05.alloc.' in t)
    assert allocation.index('tcgen05.alloc.') < allocation.index('tcgen05.relinquish_alloc_permit')
    assert allocation.index('tcgen05.relinquish_alloc_permit') < allocation.index('bar.sync 0')
    assert sum(t.count('tcgen05.relinquish_alloc_permit') for t in templates) == 1


@pytest.mark.parametrize('body', [
    '''%a:tmem<128x64xf32>=alloc()
       dealloc(%a:tmem<128x64xf32>)
       %b:tmem<128x64xf32>=alloc()
       dealloc(%b:tmem<128x64xf32>)''',
    '''for (%i:index=0 to 3) {
         %a:tmem<128x64xf32>=alloc()
         dealloc(%a:tmem<128x64xf32>)
       }''',
    '''%zero:index=index_constant() attrs={value:0}
       %one:index=index_constant() attrs={value:1}
       %condition:pred=index_lt(%zero:index,%one:index)
       if(%condition:pred) {
         %a:tmem<128x64xf32>=alloc()
         dealloc(%a:tmem<128x64xf32>)
       }''',
])
def test_multiple_repeated_and_conditional_allocations_keep_the_permit(body):
    templates = lower(body)
    assert all('tcgen05.relinquish_alloc_permit' not in t
        for t in templates if 'tcgen05.alloc.' in t)
    assert templates[-1].count('tcgen05.relinquish_alloc_permit') == 1
    assert sum(t.count('tcgen05.relinquish_alloc_permit') for t in templates) == 1


# TMEM and staging storage retire at their own last consumers.
def build(**kwargs):
    cfg = dict(
        m=1153,
        n=520,
        k=320,
        dtype="f16",
        output_m_tile=128,
        output_n_tile=128,
        operand_stages=3,
        output_grid_size=4,
        output_store="stmatrix",
        output_slices=2,
        output_store_buffers=2,
        cyclic_operand_pipeline=True,
        allow_partial_tiles=True,
    )
    cfg.update(kwargs)
    return blackwell_tcgen05_gemm_module(**cfg)


@pytest.mark.parametrize("store", ["global", "stmatrix"])
@pytest.mark.parametrize("reuse", [False, True])
def test_release_orders_last_read_but_does_not_hold_conversion(store, reuse):
    mod = build(m=1024, n=512, k=384, output_store=store, reuse_tmem_allocation=reuse)
    if reuse:
        releases = [
            op
            for op in find_ops_of_type(mod, BarrierOp)
            if op.attrs.get("resource_lifetime_reuse")
        ]
    else:
        releases = [
            op
            for op in find_ops_of_type(mod, DeallocOp)
            if isinstance(op.value.type, TmemTensorType)
        ]
    assert releases
    for release in releases:
        ops = release.parent_block.ops
        before, after = ops[: ops.index(release)], ops[ops.index(release) + 1 :]
        assert any(
            isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType)
            for op in before
        )
        assert not any(
            isinstance(op, MoveOp) and isinstance(op.src.type, TmemTensorType)
            for op in after
        )
        assert any(
            isinstance(op, ElementwiseOp) and op.attrs.get("fn") == "convert"
            for op in after
        )
    make_ptx_kernel(mod)
