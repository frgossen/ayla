import pytest
from ayla.base.ir import find_ops_of_type,find_ancestor_of_type
from ayla.gpu_dialect import AllocOp,ElementwiseOp,ForOp,MatmulOp,MoveOp,TmemTensorType,gpu_dialect
from ayla.passes.tile_matmul import TileMatmul
from ayla.passes.promote_buffers import PromoteBuffers
from ayla.passes.unroll_loop import UnrollLoop
from ayla.pipelines.canonicalize import Canonicalize
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module


def reduction():
    m=gpu_dialect.parse_module("""module attrs={target: "sm_100a"} {
      kernel @scope(%a:gmem<256x256xf16>,%b:gmem<64x256xf16>,%c:gmem<256x64xf16>) block_dim=[128,1,1] {
        %zero:rmem<f32> = constant 0.0
        matmul(%a:gmem<256x256xf16>,%b:gmem<64x256xf16>,%c:gmem<256x64xf16>,%zero:rmem<f32>) collective=xyz.block attrs={label:"mm"}
      }
    }""")
    m=TileMatmul(op_dims=["K"],tile_sizes=[64],iteration_bindings=[None],for_loop_labels=["k"],op_label_filter="mm",retain_dst_tiling_scheme=True).run(m)
    m=TileMatmul(op_dims=["M"],tile_sizes=[128],iteration_bindings=[None],for_loop_labels=["m"],op_label_filter="mm").run(m)
    m=UnrollLoop(unroll_factor=None,label_filter="m").run(m)
    return Canonicalize().run(m)


def promote(m):
    return PromoteBuffers(target_type="tmem",operands=["dst"],target_dtype="f32",scope_loop_label="k",label_filter="mm(?:_iter\\d+)*",warps=4).run(m)


def test_initialization_and_writeback_span_the_reduction():
    m=promote(reduction())
    accumulators=[a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,TmemTensorType)]
    assert len(accumulators)==2
    assert all(find_ancestor_of_type(a,ForOp) is None for a in accumulators)
    fills=[e for e in find_ops_of_type(m,ElementwiseOp) if e.fn=="fill"]
    assert len(fills)==2 and all(find_ancestor_of_type(e,ForOp) is None for e in fills)
    mm=find_ops_of_type(m,MatmulOp)
    assert len(mm)==2 and all(a.init is None and a.accumulate for a in mm)
    assert all(find_ancestor_of_type(a,ForOp).attrs["label"]=="k" for a in mm)
    assert all(find_ancestor_of_type(a,ForOp) is None for a in find_ops_of_type(m,MoveOp))


def test_scoped_promotion_rejects_aliasing_sibling_outputs():
    m=reduction();a,b=find_ops_of_type(m,MatmulOp)
    b.operands[2]=a.dst
    b.operands[-1]=a.init
    # Identical destination is one accumulator; a distinct aliasing view must
    # instead be rejected by the static sibling partition check.
    from ayla.gpu_dialect import TileOp
    original=a.dst.definition
    clone=TileOp.build(original.arg,list(original.indices),original.result().type)
    b.parent_block.ops.insert(b.parent_block.ops.index(b),clone)
    b.operands[2]=clone.result()
    with pytest.raises(ValueError,match="overlap"):promote(m)


def test_scoped_promotion_rejects_another_output_reader():
    m=reduction();mm=find_ops_of_type(m,MatmulOp)[0]
    mm.parent_block.ops.append(MoveOp.build(mm.dst,mm.dst,collective=mm.collective))
    with pytest.raises(ValueError,match="observer"):promote(m)


@pytest.mark.parametrize("group,n",[(2,128),(4,64),(8,64)])
def test_operand_group_has_one_shared_rhs_and_multiple_accumulators(group,n):
    m=blackwell_tcgen05_gemm_module(m=2048,n=n,k=256,dtype="f16",
        output_m_tiles_per_cta=group,batch_output_m_operands=True,
        output_n_tile=n,output_store="stmatrix",operand_stages=1)
    from ayla.gpu_dialect import SmemTensorType
    allocations=[a for a in find_ops_of_type(m,AllocOp) if isinstance(a.result().type,SmemTensorType)]
    assert any(a.result().type.sizes[-2:]==[128*group,64] for a in allocations)
    assert any(a.result().type.sizes[-2:]==[n,64] for a in allocations)
    mm=find_ops_of_type(m,MatmulOp)
    assert mm and all(a.dst.type.sizes==[128,n] for a in mm)


def test_loop_scoped_promotion_rejects_smem_without_scalar_fill_lowering():
    with pytest.raises(ValueError,match="tensor or register"):
        PromoteBuffers(target_type="smem",operands=["dst"],scope_loop_label="k").run(reduction())


@pytest.mark.parametrize("retain",[False,True])
def test_sequential_k_preserves_output_partition_only_when_requested(retain):
    m=reduction()
    for op in find_ops_of_type(m,MatmulOp):op.attrs["dst_tiling_scheme"]=17
    m=TileMatmul(op_dims=["K"],tile_sizes=[16],iteration_bindings=[None],
        for_loop_labels=["kk"],retain_dst_tiling_scheme=retain).run(m)
    assert all(op.attrs.get("dst_tiling_scheme")== (17 if retain else None)
        for op in find_ops_of_type(m,MatmulOp))


def test_partition_retention_rejects_non_reduction_tiling():
    with pytest.raises(ValueError,match="sequential K"):
        TileMatmul(op_dims=["M"],tile_sizes=[128],iteration_bindings=[None],
            for_loop_labels=["m"],retain_dst_tiling_scheme=True)
