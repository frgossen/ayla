"""Tests for reusable GEMM construction and scheduling."""

import textwrap

import pytest

from ayla.base.ir import find_ops
from ayla.gemm_collection.gemm import schedule_tiled_matmul
from ayla.gpu_dialect import AllocOp, MatmulOp, WrmemTensorType, gpu_dialect


def test_tiled_schedule_requires_prior_block_tiling():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<64x64xbf16>) attrs={label: "mm"}
          }
        }"""))

    with pytest.raises(ValueError, match="must already be block-tiled"):
        schedule_tiled_matmul(
            module,
            label="mm",
            output_m_tile=64,
            output_n_tile=64,
            operand_stages=2,
        )


def test_tiled_schedule_preserves_a_register_resident_output():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %acc: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0, init: 0.0}
            matmul(%a: gmem<64x64xbf16>, %b: gmem<64x64xbf16>, %acc: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm", tiled: ["block.x", "wg.x"]}
          }
        }"""))  # noqa: E501

    schedule_tiled_matmul(
        module,
        label="mm",
        output_m_tile=64,
        output_n_tile=64,
        operand_stages=2,
        output_tma_inner_size=None,
    )

    wrmem_allocs = find_ops(
        module,
        lambda op: isinstance(op, AllocOp)
        and isinstance(op.result().type, WrmemTensorType),
    )
    assert len(wrmem_allocs) == 1
    matmuls = find_ops(module, lambda op: isinstance(op, MatmulOp))
    assert matmuls
    assert all(op.dst is wrmem_allocs[0].result() for op in matmuls)
    assert all(op.attrs["tiled"].count("wg.x") == 1 for op in matmuls)
