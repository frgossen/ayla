"""K unrolling preserves bank epochs and carried output ownership."""

import math

import pytest

from ayla.passes.pipeline_rolled_shared_ring import PrepareRolledSharedRing
from ayla.ptx_codegen import make_ptx_kernel
from tests.test_rolled_shared_ring import build, check_protocol
from tests.testing_utils import assert_dominance


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("k,factor", [(128, 2), (384, 2), (384, 6), (784, 2), (784, 13)])
def test_unrolling_preserves_uneven_jobs_and_k_tails(group, k, factor):
    mod = PrepareRolledSharedRing().run(build(
        k=k, cta_group=group, rolled_k_unroll=factor,
        persistent_output_overlap=True))
    assert_dominance(mod.body)
    tiles = math.ceil(2049 / (128 * group)) * group * math.ceil(520 / 128)
    steps = math.ceil(k / 64)
    for block in range(4):
        jobs = len(range(block, tiles, 4))
        assert check_protocol(mod, block, steps) == dict(
            load=jobs * steps, mma=jobs * steps, read=jobs * 2, store=jobs * 2)


def test_default_unroll_preserves_ptx():
    assert make_ptx_kernel(build()).ptx == make_ptx_kernel(build(rolled_k_unroll=1)).ptx


@pytest.mark.parametrize("factor", [0, -1, True, None, 1.5])
def test_invalid_unroll_rejected(factor):
    with pytest.raises(ValueError, match="positive integer"):
        build(rolled_k_unroll=factor)


def test_unroll_requires_rolled_operand_pipeline():
    with pytest.raises(ValueError, match="requires a rolled operand pipeline"):
        build(rolled_k_unroll=2, rolled_operand_pipeline=False)
