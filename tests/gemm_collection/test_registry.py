"""Tests for what registering a kernel defines."""

import pytest
import torch

import ayla.gemm_collection  # noqa: F401  (defines every op)
from ayla.gemm_collection.registry import register_gemm


def test_a_gemm_registers_an_op_that_allocates_its_result():
    op = torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles.default
    assert str(op._schema).endswith("(Tensor arg0, Tensor arg1) -> Tensor")


def test_both_tma_store_variants_register_separately():
    assert (
        torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3.default
        is not torch.ops.ayla.auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles.default
    )


def test_register_rejects_a_builder_that_takes_arguments():
    # Registering such a builder would silently cover only its defaults.
    with pytest.raises(TypeError, match="take no arguments"):

        @register_gemm
        def variant_module(output_m_tiles_per_cta: int = 1):
            raise AssertionError("builder should not run")


def test_register_defers_building_the_kernel():
    @register_gemm
    def lazy_registry_test_module():
        raise AssertionError("builder should not run")
