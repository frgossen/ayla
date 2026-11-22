"""Byte-width selection, TMA bit mappings, and actual WGMMA operand reads."""

import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    AllocOp,
    BF16Type,
    E4M3Type,
    E5M2Type,
    F16Type,
    SmemTensorType,
    gpu_dialect,
)
from ayla.passes.promote_buffers import PromoteBuffers
from ayla.pipelines.tile_matmul_pipeline import schedule_register_matmul
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher

DTYPES = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
CASES = [
    (dtype, tile)
    for dtype in DTYPES
    for tile in ((32, 64, 128, 256) if dtype.startswith("e") else (16, 32, 64, 128))
]


def small_matmul(dtype, tile):
    module = gpu_dialect.parse_module(f"""module {{
      kernel @small_{dtype}_{tile}(%a: gmem<64x512x{dtype}>, %b: gmem<168x512x{dtype}>, %c: gmem<64x168xf32>) grid_dim=[1,1,1] block_dim=[128,1,1] {{
        %r: wgrmem<64x168xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={{id: 0}}
        %zero: rmem<f32> = constant 0.0
        elementwise fill(%r: wgrmem<64x168xf32 warps=4 layout="wgmma_layout_0">, %zero: rmem<f32>)
        matmul(%a: gmem<64x512x{dtype}>, %b: gmem<168x512x{dtype}>, %r: wgrmem<64x168xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg,yz.grid] attrs={{label: "mm"}}
        move(%c: gmem<64x168xf32>, %r: wgrmem<64x168xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg,yz.grid]
      }}
    }}""")
    schedule_register_matmul(
        label="mm",
        operand_stages=3,
        operand_k_tile=tile,
        mma_k_tile=32 if dtype.startswith("e") else 16,
    ).run(module)
    return module


@pytest.mark.parametrize("dtype,tile", CASES)
def test_operand_swizzle_uses_bytes(dtype, tile):
    module = small_matmul(dtype, tile)
    operands = [
        op.result().type
        for op in find_ops_of_type(module, AllocOp)
        if isinstance(op.result().type, SmemTensorType)
        and op.result().type.dtype.name == dtype
    ]
    assert operands
    row_bytes = min(tile * torch.empty([], dtype=DTYPES[dtype]).element_size(), 128)
    assert {operand.swizzle for operand in operands} == {f"{row_bytes}B"}


@pytest.mark.parametrize("dtype", DTYPES)
def test_auto_swizzle_respects_reduced_physical_block(dtype):
    typ = {"e4m3": E4M3Type, "e5m2": E5M2Type, "bf16": BF16Type, "f16": F16Type}[
        dtype
    ]()
    promotion = PromoteBuffers(
        target_type="smem",
        operands=["lhs", "rhs"],
        swizzle="auto",
        physical_layout={"block_width": 128, "order": "block_major"},
    )
    # gcd(96,128)=32 elements: 32 bytes for FP8, 64 bytes for FP16/BF16.
    result = promotion._build_smem_type([8, 96], typ)
    assert result.swizzle == ("32B" if dtype.startswith("e") else "64B")
    assert result.attrs["physical_layout"]["block_width"] == 32


@cuda
@pytest.mark.parametrize("dtype,tile", CASES)
def test_wgmma_reads_every_swizzle_width(dtype, tile, tmp_path):
    generator = torch.Generator(device="cuda").manual_seed(20261119)
    typ = DTYPES[dtype]
    # Exactly representable products isolate addressing from FP8's reduced
    # accumulation precision, which differs from the FP32 reference.
    a = (
        torch.randint(-4, 5, (64, 512), device="cuda", generator=generator).float() / 4
    ).to(typ)
    b = (
        torch.randint(-4, 5, (168, 512), device="cuda", generator=generator).float() / 4
    ).to(typ)
    out = torch.empty(64, 168, device="cuda")
    run = make_ptx_launcher(make_ptx_kernel(small_matmul(dtype, tile)))
    run(a, b, out)
    torch.cuda.synchronize()
    reference = a.double() @ b.double().T
    torch.testing.assert_close(out.double(), reference, atol=0, rtol=0)
