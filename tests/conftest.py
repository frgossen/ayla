"""Shared pytest fixtures."""
import itertools

import pytest

from ayla.gpu_dialect import AllocOp


@pytest.fixture(autouse=True)
def reset_alloc_ids():
    """Number allocations from zero in every test.

    The ids come from a counter shared by the whole process, so without this a
    test's expected IR would depend on how many allocations were built before
    it -- importing the kernel collection alone burns over a hundred.
    """
    AllocOp._id_counter = itertools.count()


@pytest.fixture
def ayla_installed():
    """ayla claiming `aten.mm` for the duration of a test."""
    # Imported here rather than at module scope: conftest is loaded for every
    # run, and pulling in `torch._inductor` costs a second and a half.
    from ayla import torch_compile

    torch_compile.install()
    yield
    torch_compile.uninstall()
