"""Register copies and scalar broadcasting, independent of allocation hoisting."""
import textwrap

import pytest
import torch

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import PTXTemplateOp, gpu_dialect
from ayla.passes.lower_to_ptx_template import lower_to_ptx_template
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from tests.testing_utils import requires_cuda


@pytest.mark.parametrize("src_type", ["rmem<f32>", "rmem<1xf32>", "rmem<3xf32>"])
def test_register_move_copies_or_broadcasts_nonconstant_values(src_type):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @copy(%src: {src_type}, %dst: rmem<3xf32>) {{
            move(%dst: rmem<3xf32>, %src: {src_type})
          }}
        }}"""))
    lower_to_ptx_template(module)
    [move] = find_ops_of_type(module, PTXTemplateOp)
    assert move.template == "".join(
        f"mov.f32 ${{f32:operand.dst.r{i}}}, "
        f"${{f32:operand.src.r{i if src_type == 'rmem<3xf32>' else 0}}};\n"
        for i in range(3)
    )


@pytest.mark.parametrize("src_type, message", [
    ("rmem<f64>", "matching dtypes"),
    ("rmem<2xf32>", "matching types or a scalar"),
])
def test_register_move_rejects_incompatible_tensors(src_type, message):
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @copy(%src: {src_type}, %dst: rmem<3xf32>) {{
            move(%dst: rmem<3xf32>, %src: {src_type})
          }}
        }}"""))
    with pytest.raises(NotImplementedError, match=message):
        lower_to_ptx_template(module)


@requires_cuda
@pytest.mark.parametrize("dtype, torch_dtype", [
    ("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16),
])
def test_register_copy_and_broadcast_respect_views_on_gpu(dtype, torch_dtype):
    root = f'wgrmem<64x128x{dtype} warps=4 layout="wgmma_layout_0">'
    tile = f'wgrmem<64x64x{dtype} warps=4 layout="wgmma_layout_0">'
    module = gpu_dialect.parse_module(textwrap.dedent(f"""\
        module {{
          kernel @copy_view(%out: gmem<64x128x{dtype}>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %sentinel: rmem<{dtype}> = constant() attrs={{value: -1.0}}
            %value: rmem<{dtype}> = constant() attrs={{value: 2.5}}
            %src: {root} = alloc() attrs={{id: 0}}
            %dst: {root} = alloc() attrs={{id: 1}}
            move(%src: {root}, %sentinel: rmem<{dtype}>)
            move(%dst: {root}, %sentinel: rmem<{dtype}>)
            %src_tile: {tile} = tile(%src: {root}, 0, 1)
            %dst_tile: {tile} = tile(%dst: {root}, 0, 1)
            move(%src_tile: {tile}, %value: rmem<{dtype}>)
            move(%dst_tile: {tile}, %src_tile: {tile})
            move(%out: gmem<64x128x{dtype}>, %dst: {root})
          }}
        }}"""))
    out = torch.empty((64, 128), dtype=torch_dtype, device="cuda")
    make_ptx_launcher(make_ptx_kernel(module))(out)
    expected = torch.full_like(out, -1.0)
    expected[:, 64:] = 2.5
    torch.testing.assert_close(out, expected)
