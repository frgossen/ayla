"""Tests for the TorchInductor backend."""
import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code

from ayla import torch_inductor


# TODO: Duplicated in every test module -- this belongs in `ayla.testing`.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"
)


@pytest.fixture
def ayla_installed():
    """ayla claiming `aten.mm` for the duration of a test."""
    torch_inductor.install()
    yield
    torch_inductor.uninstall()


@requires_cuda
@pytest.mark.usefixtures("ayla_installed")
@pytest.mark.parametrize("shape,claimed", [
    pytest.param((4096, 4096, 4096), True, id="covered"),
    # One of the Glimmer decoder GEMMs; the collection has no kernel for it.
    pytest.param((2, 6656, 4096), False, id="uncovered"),
])
def test_a_matmul_goes_through_ayla_iff_covered(shape, claimed):
    """Claimed exactly when the collection holds a kernel for the shape."""
    torch.manual_seed(0)
    m, k, n = shape
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    def fn(x, w):
        return x @ w.t()

    expected = fn(x, w)
    # The backend is not part of the Inductor cache key, so a warm cache would
    # serve a stock-compiled graph and the lowering would never run.
    with inductor_config.patch(force_disable_caches=True), torch.no_grad():
        compiled = torch.compile(fn, dynamic=False)
        actual, (code,) = run_and_get_code(compiled, x, w)

    assert ("extern_kernels.ayla_" in code) is claimed
    if claimed:
        # ayla displaced the built-in path rather than sitting beside it.
        assert "extern_kernels.mm(" not in code
        # `w.t()` already is the contiguous [N, K] weight ayla wants, so no
        # transpose is materialized.
        assert "reinterpret_tensor" in code

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
