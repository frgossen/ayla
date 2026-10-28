"""Shared pytest fixtures."""
import itertools

import pytest

from ayla.gpu_dialect import AllocOp, MemoryEffect


@pytest.fixture(autouse=True)
def reset_fresh_ids():
    """Number allocations and tiling schemes from zero in every test.

    The ids come from a counter shared by the whole process, so without this a
    test's expected IR would depend on how many objects were built before it.
    """
    AllocOp._id_counter = itertools.count()
    MemoryEffect._tiling_scheme_id_counter = itertools.count()


@pytest.fixture
def ayla_installed():
    """Install Ayla's post-grad FX rewrites for the duration of a test."""
    from ayla import torch_compile

    torch_compile.install()
    try:
        yield
    finally:
        torch_compile.uninstall()
