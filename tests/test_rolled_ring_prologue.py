"""The rolled shared ring's first loads precede TMEM allocation and stale publications."""

import re

import pytest

from ayla import Block
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AllocOp, BarrierOp, LockType
from ayla.passes.insert_syncs import drop_vacuous_publications
from ayla.ptx_codegen import make_ptx_kernel


def _ring(**overrides):
    config = dict(
        m=12672, n=3072, k=768, dtype="f16", cta_group=2, output_m_tile=128,
        output_n_tile=256, operand_k_tile=128, operand_stages=3, output_grid_size=148,
        output_grid_fast_axis="m", persistent_output_pipeline=True,
        persistent_output_schedule="shared_ring", rolled_operand_pipeline=True,
        hoist_mma_descriptors=True, output_store="stmatrix", output_slices=4,
        tmem_writeback_register_budget=64, tmem_writeback_warps=8, allow_partial_tiles=True,
    )
    config.update(overrides)
    return blackwell_tcgen05_gemm_module(**config)


@pytest.mark.parametrize("k_tile, stages", [(64, 4), (128, 3)])
def test_fill_precedes_tmem_allocation_behind_one_relaxed_rendezvous(k_tile, stages):
    ptx = make_ptx_kernel(_ring(operand_k_tile=k_tile, operand_stages=stages)).ptx
    first_load = ptx.index("cp.async.bulk.tensor")
    assert first_load < ptx.index("tcgen05.alloc.cta_group::2")
    # Only the ring's own event publication remains before the fill; it
    # orders no data.
    assert re.findall(r"barrier\.cluster\.arrive\.(\w+);", ptx[:first_load]) == ["relaxed"]


def test_k128_stages_are_only_accepted_by_the_rolled_shared_ring():
    with pytest.raises(ValueError, match="must be shared_ring"):
        _ring(persistent_output_schedule="framed")
    with pytest.raises(ValueError, match="three K stages"):
        _ring(k=256)


def _lock(**attrs):
    return AllocOp.build_with_fresh_id(LockType.build(kind="mbar", attrs=attrs or None))


def _publication(scope):
    return BarrierOp.build(scope, attrs={"initialization_publication": True})


@pytest.mark.parametrize(("locks", "kept"), [
    ([], []),
    ([_lock], ["block"]),
    ([lambda: _lock(cta_group=2)], ["block", "cluster"]),
])
def test_publications_without_initialized_locks_are_dropped(locks, kept):
    block = Block(ops=[make() for make in locks] + [_publication("block"), _publication("cluster")])
    drop_vacuous_publications(block)
    assert [op.scope for op in block.ops if isinstance(op, BarrierOp)] == kept


def test_lone_cluster_publication_of_cta_locks_is_kept():
    block = Block(ops=[_lock(), _publication("cluster")])
    drop_vacuous_publications(block)
    assert [op.scope for op in block.ops if isinstance(op, BarrierOp)] == ["cluster"]
