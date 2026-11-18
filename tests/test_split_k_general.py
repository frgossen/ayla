import pytest

from ayla.gemm_jit import _seed, build_gemm_module, select_gemm_schedule
from ayla.gpu_dialect import TmaDescType
from ayla.passes.verify_gpu_dialect import VerifyGpuDialect
from ayla.ptx_codegen import make_ptx_kernel


@pytest.mark.parametrize("op", ["mm", "addmm", "bmm"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("m,n,k,parts", [(192, 264, 8200, 4), (23, 24, 264, 16)])
def test_generic_promoted_split_k_handles_tails_batches_and_bias(
    op, dtype, m, n, k, parts
):
    base = select_gemm_schedule("mm", "float16", 4096, 4096, 4096).captured_config
    cfg = dict(
        base,
        output_cta_count=None,
        transpose_output=False,
        output_tiles_per_cta=1,
        output_n_tiles_per_cta=1,
        output_tma_inner_size=None,
        output_slices=1,
        output_store_buffers=1,
        warpgroup_m_tile=None,
        output_m_tile=128,
        output_n_tile=128,
        warpgroup_n_tile=128,
        output_store="scalar",
        split_k_parts=parts,
        operand_stages=4,
        cyclic_operand_pipeline=False,
        lhs_l2_promotion="256B",
        rhs_l2_promotion="256B",
        multicast=False,
        cluster_m_ctas=1,
        cluster_n_ctas=1,
    )
    batch = 2 if op == "bmm" else 1
    seed = _seed(cfg, op, m, n, k, (batch, m, n, k), True)
    assert seed is not None
    module = build_gemm_module(op, dtype, batch, m, n, k, seed)
    for phase in (module.partials, module.reduction):
        VerifyGpuDialect().run(phase)
        assert make_ptx_kernel(phase).ptx
    partials = make_ptx_kernel(module.partials)
    # Both 64-row warpgroups compute output: there is no extra producer group.
    assert partials.block_dim == (256, 1, 1)
    assert "setmaxnreg" not in partials.ptx
    maps = [
        arg.type
        for arg in module.partials.body.ops[0].body.arguments
        if isinstance(arg.type, TmaDescType)
    ]
    assert len(maps) >= 2 and all(desc.l2_promotion == "256B" for desc in maps[:2])
    assert module.workspace_shape[-2:] == (m, n)


def test_retired_specialized_split_k_config_is_not_silently_reinterpreted():
    base = select_gemm_schedule("mm", "float16", 4096, 4096, 4096).captured_config
    cfg = dict(base, split_k_warp_specialized=True)
    assert _seed(cfg, "mm", 4096, 4096, 4096, (1, 4096, 4096, 4096), True) is None
