"""Distributed operands and completion lifetimes for paired Blackwell MMAs."""
import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.blackwell_tcgen05 import blackwell_tcgen05_gemm_module
from ayla.gpu_dialect import AsyncOp, MatmulOp, gpu_dialect
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.ptx_codegen import make_ptx_kernel


CONFIG = dict(m=512, n=512, k=384, dtype='f16', cta_group=2,
              output_n_tile=256, operand_stages=6, output_store='stmatrix',
              output_slices=2, output_store_buffers=2, output_layout='tcgen05_16x256b',
              tmem_writeback_register_budget=64, reuse_tmem_allocation=True,
              cyclic_operand_pipeline=True, allow_partial_tiles=True)


def _module(**kwargs):
    return blackwell_tcgen05_gemm_module(**{**CONFIG, **kwargs})


def test_paired_operand_placement_and_completion():
    module = _module()
    VerifyGpuDialect().run(module)
    assert module.body.ops[0].cluster_dim == (2, 1, 1)
    mmas = find_ops_of_type(module, MatmulOp)
    assert mmas
    assert all(mm.lhs.type.sizes == [128, 16] and mm.rhs.type.sizes == [128, 16]
               and mm.dst.type.sizes == [128, 256] and mm.dst.type.cta_group == 2 for mm in mmas)
    assert all(op.lock.type.attrs.get('cta_group') == 2 for op in find_ops_of_type(module, AsyncOp)
               if op.lock.type.kind == 'tcgen05')
    assert repr(gpu_dialect.parse_module(repr(module))) == repr(module)
    kernel = make_ptx_kernel(module)
    assert 'tcgen05.mma.cta_group::2.kind::f16' in kernel.ptx
    assert 'tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster' in kernel.ptx
    assert 'tcgen05.alloc.cta_group::2' in kernel.ptx
    assert 'tcgen05.dealloc.cta_group::2' in kernel.ptx
    assert 'tcgen05.relinquish_alloc_permit.cta_group::2' in kernel.ptx
    assert 'tcgen05.mma.cta_group::1' not in kernel.ptx


def test_paired_tail_pads_the_whole_cluster():
    module = _module(m=384, n=264, k=80, operand_stages=2)
    kernel = make_ptx_kernel(module)
    assert kernel.grid_dim == (4, 2, 1)
    assert kernel.cluster_dim == (2, 1, 1)


def test_paired_persistent_grid_keeps_pairs_uniform():
    kernel = make_ptx_kernel(_module(output_grid_size=2, reuse_completion_locks=True))
    assert kernel.grid_dim == (2, 1, 1)
    assert kernel.cluster_dim == (2, 1, 1)


def test_paired_persistent_rejects_an_odd_m_tile_count():
    with pytest.raises(ValueError, match='paired persistent'):
        _module(m=384, output_grid_size=2)


def test_paired_global_store_requires_complete_pairs():
    with pytest.raises(ValueError, match='paired global'):
        _module(m=384, output_store='global')


@pytest.mark.parametrize('cluster', [(1, 1, 1), (1, 2, 1)])
def test_paired_allocation_requires_matching_cluster(cluster):
    module = _module()
    module.body.ops[0].attrs['cluster_dim'] = list(cluster)
    with pytest.raises(ValueError, match='paired TMEM|paired completion'):
        VerifyGpuDialect().run(module)


@pytest.mark.parametrize('config', [dict(output_m_tile=64), dict(output_n_tile=512),
                                    dict(output_grid_size=147), dict(output_grid_fast_axis='n')])
def test_paired_rejects_unsupported_schedule(config):
    with pytest.raises(ValueError, match='paired MMA|shared-memory capacity|paired persistent'):
        _module(**config)
