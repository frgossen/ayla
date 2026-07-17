"""Tests for the cross-thread shared-buffer sync verifier."""
import contextlib
import io

import pytest
import torch

# Importing gemm_collection registers (and compiles) all kernels, hence CUDA.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@requires_cuda
def test_wgmma256_wg_sync_is_covered():
    """The scheduled wgmma256_wg IR has every cross-thread smem hazard covered,
    and removing the wgmma-await WAR barrier makes the verifier flag the exact
    read→reload hazard (the silent-deadlock class)."""
    import ayla.gemm_collection as gc
    from ayla.sync_analysis import verify_sync, SyncError

    with contextlib.redirect_stdout(io.StringIO()):
        wg = gc.auto_matmul_4096_wgmma256_wg_module()

    # As generated (the WAR bar.sync at the wgmma await is present): covered.
    verify_sync(wg, wgmma_await_barrier=True)

    # Without that barrier: the read→reload WAR on a shared buffer is uncovered.
    with pytest.raises(SyncError):
        verify_sync(wg, wgmma_await_barrier=False)
