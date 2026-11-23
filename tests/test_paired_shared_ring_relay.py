"""Paired persistent shared rings: pair-major N-fast jobs and the paired relay."""

import pytest
import torch

from ayla.gemm_collection.blackwell_router import make_blackwell_router_launcher


def _config(grid, fast, stages, queue, n_tile=128):
    return dict(cta_group=2, output_grid_size=grid, output_grid_fast_axis=fast,
                output_n_tile=n_tile, operand_k_tile=64, operand_stages=stages,
                output_store="stmatrix", output_slices=2, output_store_buffers=2,
                output_store_pending_groups=1, allow_partial_tiles=True,
                pipeline={"operand_scope": "output_jobs", "mma_pending": queue})


def test_paired_shared_ring_uses_relay_instead_of_per_refill_gathers():
    launcher = make_blackwell_router_launcher(
        m=1280, n=640, k=512, dtype="f16", configuration=_config(6, "n", 3, 1))
    ptx = launcher.partial_kernel.ptx
    assert "2cta" in launcher.partial_kernel.name and "rolled" not in launcher.partial_kernel.name
    assert "cluster_to_leader" not in ptx


def test_paired_persistent_grid_requires_an_even_cta_count():
    with pytest.raises(ValueError, match="even"):
        make_blackwell_router_launcher(m=1280, n=640, k=512, dtype="f16",
                                       configuration=_config(7, "n", 3, 1))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)
@pytest.mark.parametrize("shape,grid,fast,stages,queue", [
    # Odd M-tile counts leave the last pair half outside M.
    ((1280, 640, 512), 2, "n", 2, 0),
    ((1280, 640, 512), 6, "n", 3, 1),
    ((1408, 320, 320), 6, "n", 3, 2),
    ((3200, 768, 256), 10, "m", 3, 2),
    ((3200, 768, 256), 10, "n", 3, 1),
    # A full persistent grid with several jobs per pair.
    ((12672, 768, 512), 148, "n", 5, 1),
])
def test_paired_shared_ring_matches_reference(shape, grid, fast, stages, queue):
    m, n, k = shape
    n_tile = 160 if n % 128 else 128
    launcher = make_blackwell_router_launcher(
        m=m, n=n, k=k, dtype="f16", configuration=_config(grid, fast, stages, queue, n_tile))
    torch.manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    expected = (a.float() @ b.float().T).half()
    for _ in range(3):
        out.fill_(float("nan"))
        launcher(a, b, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.25, rtol=0.02)
