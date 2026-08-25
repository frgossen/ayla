"""Tests for the GEMM kernel registry's shape derivation and lookup."""

import pytest
import torch

import ayla.gemm_collection  # noqa: F401  (registers every kernel)
from ayla.gemm_collection import query
from ayla.gemm_collection.registry import register


def _kernel_named(name):
    match = [k for k in query() if k.name == name]
    assert match, f"{name} is not registered"
    return match[0]


def test_kernel_shapes_come_from_the_built_kernel():
    bf16 = _kernel_named("auto_matmul_4096_wgmma256_wg_tma_store_k3")
    assert (bf16.m, bf16.k, bf16.n) == (4096, 4096, 4096)
    assert bf16.dtype is torch.bfloat16

    f32 = _kernel_named("loop_8x8_matmul_4096")
    assert (f32.m, f32.k, f32.n) == (4096, 4096, 4096)
    assert f32.dtype is torch.float32


def test_both_tma_store_variants_register_separately():
    one_tile = _kernel_named("auto_matmul_4096_wgmma256_wg_tma_store_k3")
    two_tiles = _kernel_named("auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles")
    assert one_tile.op is not two_tiles.op


def test_register_rejects_a_builder_that_takes_arguments():
    # Registering such a builder would silently cover only its defaults.
    with pytest.raises(TypeError, match="take no arguments"):

        @register
        def variant_module(output_tiles_per_cta: int = 1):
            raise AssertionError("builder should not run")


def test_query_filters_on_every_field():
    everything = query()
    assert everything, "no kernels registered"

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
