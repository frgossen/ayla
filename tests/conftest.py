"""Shared pytest fixtures."""
import itertools

import pytest

from ayla.gpu_dialect import AllocOp


@pytest.fixture(autouse=True)
def reset_alloc_uniq():
    """Number allocations from zero in every test.

    The ids come from a counter shared by the whole process, so without this a
    test's expected IR would depend on how many allocations were built before
    it -- importing the kernel collection alone burns over a hundred.
    """
    AllocOp._uniq_counter = itertools.count()
