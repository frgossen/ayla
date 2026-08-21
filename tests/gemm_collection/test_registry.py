"""Tests for the GEMM kernel registry's shape derivation and lookup."""

import pytest
import torch

import ayla.gemm_collection  # noqa: F401  (registers every kernel)
from ayla.gemm_collection import query
from ayla.gemm_collection.registry import _gemm_dims, register
from ayla.gpu_dialect import BF16Type, F32Type, GmemTensorType


class FakeKernel:
    """Just enough of a built kernel for :func:`_gemm_dims`."""

    def __init__(self, name, signature):
        self.name = name
        self.signature = signature


def gmem(sizes, dtype=BF16Type()):
    return GmemTensorType.build(sizes, dtype)


def test_gemm_dims_reads_the_transposed_rhs():
    # The dialect holds the right operand as [N, K], so a kernel whose n and k
    # differ must still come back with n and k the right way round.
    kernel = FakeKernel(
        "nonsquare",
        [gmem([4096, 2048]), gmem([1024, 2048]), gmem([4096, 1024])],
    )
    assert _gemm_dims(kernel) == (4096, 2048, 1024, torch.bfloat16)


def test_gemm_dims_ignores_trailing_non_gmem_signature_entries():
    kernel = FakeKernel(
        "with_descriptors",
        [gmem([256, 64]), gmem([128, 64]), gmem([256, 128]), object(), object()],
    )
    assert _gemm_dims(kernel) == (256, 64, 128, torch.bfloat16)


@pytest.mark.parametrize(
    "name,signature",
    [
        ("too_few", [gmem([64, 64]), gmem([64, 64])]),
        ("not_2d", [gmem([2, 64, 64]), gmem([2, 64, 64]), gmem([2, 64, 64])]),
        # k disagrees between the operands.
        ("inconsistent_k", [gmem([64, 32]), gmem([64, 16]), gmem([64, 64])]),
        # dst does not match m x n.
        ("bad_dst", [gmem([64, 32]), gmem([16, 32]), gmem([64, 64])]),
        (
            "mixed_dtypes",
            [gmem([64, 32]), gmem([16, 32], F32Type()), gmem([64, 16])],
        ),
    ],
)
def test_gemm_dims_rejects_malformed_signatures(name, signature):
    with pytest.raises(ValueError, match=name):
        _gemm_dims(FakeKernel(name, signature))


def test_register_rejects_a_builder_that_takes_arguments():
    # Registering such a builder would silently cover only its defaults.
    with pytest.raises(TypeError, match="take no arguments"):

        @register
        def variant_module(output_tiles_per_cta: int = 1):
            raise AssertionError("builder should not run")


def test_query_filters_on_every_field():
    everything = query()
    assert everything, "no kernels registered"
    assert all(k.m == k.k == k.n == 4096 for k in everything)

    assert query(m=4096, k=4096, n=4096) == everything
    assert query(m=1024) == []
    assert query(dtype=torch.float16) == []

    by_dtype = query(dtype=torch.bfloat16)
    assert by_dtype and all(k.dtype is torch.bfloat16 for k in by_dtype)
    assert len(by_dtype) + len(query(dtype=torch.float32)) == len(everything)


def test_registered_kernels_match_their_op_names():
    for kernel in query():
        assert kernel.label == f"ayla.{kernel.name}"
        # make_torch_custom_call hands back the overload, not the packet.
        assert getattr(torch.ops.ayla, kernel.name).default is kernel.op
