"""In-kernel split-K reductions: last-contributor (fixup) and resident backends."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)
@pytest.mark.parametrize(
    "shape,configuration",
    [
        # Full tiles: batched loads-first reduction.
        ((512, 512, 8192), dict(split_k=8, cta_group=1, operand_stages=7,
                                tmem_writeback_warps=8, tmem_writeback_register_budget=32)),
        ((768, 768, 8192), dict(split_k=4, cta_group=2, operand_stages=9,
                                tmem_writeback_warps=8, tmem_writeback_register_budget=32)),
        # Partial M/N tiles: guarded chunks, paired and single-CTA.
        ((144, 144, 65536), dict(split_k=32, cta_group=2, operand_stages=4,
                                 tmem_writeback_warps=8, tmem_writeback_register_budget=32)),
        ((2, 256, 32768), dict(split_k=32, cta_group=1, output_m_tile=64, output_n_tile=64,
                               operand_stages=4, tmem_writeback_warps=8,
                               tmem_writeback_register_budget=32)),
    ],
)
@pytest.mark.parametrize("backend", ["fixup", "resident"])
def test_last_contributor_split_k_reduces_in_one_kernel(shape, configuration, backend):
    from ayla.gemm_collection.blackwell_router import make_blackwell_router_launcher

    m, n, k = shape
    launcher = make_blackwell_router_launcher(
        m=m, n=n, k=k, dtype="f16",
        configuration={**configuration, "split_k_backend": backend},
    )
    assert launcher.partial_kernel.cooperative == (backend == "resident")
    assert launcher.in_kernel_reduction and launcher.reduction_launch is None
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    # Garbage before first use: the launcher clears counters once.
    workspace = torch.full(launcher.workspace_shape, float("nan"), device="cuda")
    expected = (a.float() @ b.float().T).half()
    for _ in range(3):
        out.fill_(float("nan"))
        launcher(a, b, out, workspace)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.25, rtol=0.02)
    # Every reducer restored its tile counters for the next launch.
    group = configuration.get("cta_group", 2)
    tile_m = configuration.get("output_m_tile", 128)
    padded_m = -(-m // (group * tile_m)) * group * tile_m
    counters = workspace[configuration["split_k"] * padded_m:].reshape(-1).view(torch.int32)
    assert int(counters.count_nonzero()) == 0


def test_last_contributor_split_k_rejects_unsupported_layouts():
    from ayla.gemm_collection.split_k import blackwell_split_k_workspace_modules

    with pytest.raises(ValueError, match="in-kernel reduction"):
        blackwell_split_k_workspace_modules(m=256, n=256, k=4096, split_k=4,
                                            transpose_output=True, in_kernel_reduction=True)
    with pytest.raises(ValueError, match="in-kernel reduction"):
        blackwell_split_k_workspace_modules(m=256, n=256, k=4096, split_k=4,
                                            bias=True, in_kernel_reduction=True)
    with pytest.raises(ValueError, match="resident reduction"):
        blackwell_split_k_workspace_modules(m=256, n=256, k=4096, split_k=4,
                                            resident_reduction=True)
