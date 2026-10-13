"""Elementwise fill: broadcasting a scalar value into a register tensor."""
import textwrap

import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import PTXTemplateOp, gpu_dialect
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


@pytest.mark.parametrize("dst_type, scalar_type, expected", [
    ("rmem<3xf32>", "rmem<f32>", [
        "mov.f32 ${f32:operand.dst.r%d}, ${f32:operand.src.r0};" % i
        for i in range(3)
    ]),
    ('wgrmem<64x8xbf16 warps=4 layout="wgmma_layout_0">', "rmem<bf16>", [
        "mov.b32 ${b32:operand.dst.r%d}, "
        "{${b16:operand.src.r0}, ${b16:operand.src.r0}};" % i
        for i in range(2)
    ]),
])
def test_fill_broadcasts_a_nonconstant_scalar(dst_type, scalar_type, expected):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @fill(%src: {scalar_type}, %dst: {dst_type}) {{
            elementwise fill(%dst: {dst_type}, %src: {scalar_type})
          }}
        }}"""))
    lower_to_ptx_template(module)
    [fill] = find_ops_of_type(module, PTXTemplateOp)
    assert fill.template == "".join(line + "\n" for line in expected)


@pytest.mark.parametrize("src_type, message", [
    ("rmem<f64>", "matching dtypes"),
    ("rmem<1xf32>", "rank-0 source"),
])
def test_fill_rejects_incompatible_sources(src_type, message):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @fill(%src: {src_type}, %dst: rmem<3xf32>) {{
            elementwise fill(%dst: rmem<3xf32>, %src: {src_type})
          }}
        }}"""))
    with pytest.raises(ValueError, match=message):
        verify_gpu_dialect(module)


def test_alloc_rejects_an_initializer():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @alloc() {
            %acc: rmem<3xf32> = alloc() attrs={id: 0, init: 0.0}
          }
        }"""))
    with pytest.raises(ValueError, match="use elementwise fill"):
        verify_gpu_dialect(module)


@requires_cuda
@pytest.mark.parametrize("dtype, torch_dtype", [
    ("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16),
])
def test_fill_writes_every_element_on_gpu(dtype, torch_dtype):
    fragment = f'wgrmem<64x64x{dtype} warps=4 layout="wgmma_layout_0">'
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @fill(%out: gmem<64x64x{dtype}>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %acc: {fragment} = alloc() attrs={{id: 0}}
            %value: rmem<{dtype}> = constant() attrs={{value: 2.5}}
            elementwise fill(%acc: {fragment}, %value: rmem<{dtype}>)
            move(%out: gmem<64x64x{dtype}>, %acc: {fragment})
          }}
        }}"""))
    out = torch.empty((64, 64), dtype=torch_dtype, device="cuda")
    make_ptx_launcher(make_ptx_kernel(module))(out)
    torch.testing.assert_close(out, torch.full_like(out, 2.5), rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype, torch_dtype", [
    ("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16),
])
def test_fill_respects_views_on_gpu(dtype, torch_dtype):
    root = f'wgrmem<64x128x{dtype} warps=4 layout="wgmma_layout_0">'
    tile = f'wgrmem<64x64x{dtype} warps=4 layout="wgmma_layout_0">'
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @fill_view(%out: gmem<64x128x{dtype}>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %sentinel: rmem<{dtype}> = constant() attrs={{value: -1.0}}
            %value: rmem<{dtype}> = constant() attrs={{value: 2.5}}
            %acc: {root} = alloc() attrs={{id: 0}}
            elementwise fill(%acc: {root}, %sentinel: rmem<{dtype}>)
            %acc_tile: {tile} = tile(%acc: {root}, 0, 1)
            elementwise fill(%acc_tile: {tile}, %value: rmem<{dtype}>)
            move(%out: gmem<64x128x{dtype}>, %acc: {root})
          }}
        }}"""))
    out = torch.empty((64, 128), dtype=torch_dtype, device="cuda")
    make_ptx_launcher(make_ptx_kernel(module))(out)
    expected = torch.full_like(out, -1.0)
    expected[:, 64:] = 2.5
    torch.testing.assert_close(out, expected)
