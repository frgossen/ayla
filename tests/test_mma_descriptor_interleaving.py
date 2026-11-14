"""Descriptor scheduling must preserve defaults, issue ownership and GEMM data."""

from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.ptx_codegen import make_ptx_kernel


def module(**extra):
    cfg = dict(m=1000, n=960, k=1088, dtype='f16', output_n_tile=64,
               operand_stages=4, cyclic_operand_pipeline=True,
               output_store='stmatrix', allow_partial_tiles=True,
               cta_group=2, cluster_n_ctas=2, cluster_dim=(4,1,1))
    cfg.update(extra)
    return blackwell_tcgen05_gemm_module(**cfg)


def test_disabled_option_preserves_default_codegen():
    default = make_ptx_kernel(module()).ptx
    disabled = make_ptx_kernel(module(interleave_mma_descriptors=False)).ptx
    assert default == disabled


def test_interleaving_keeps_issue_and_completion_contract():
    default = make_ptx_kernel(module()).ptx
    enabled = make_ptx_kernel(module(interleave_mma_descriptors=True)).ptx
    assert default != enabled
    for instruction in ('tcgen05.mma.', 'tcgen05.commit.', 'cp.async.bulk.tensor.',
                        'tcgen05.alloc.', 'tcgen05.dealloc.', 'barrier.cluster.'):
        assert default.count(instruction) == enabled.count(instruction)
    # The old compact control change elects the lane before preparation.
    assert 'tcgen05_issue_done' not in default and 'tcgen05_issue_done' not in enabled
