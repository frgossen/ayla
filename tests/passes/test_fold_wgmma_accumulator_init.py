"""Tests for folding WGMMA accumulator initialization."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import AllocOp, MatmulOp, gpu_dialect
from ayla.passes.fold_wgmma_accumulator_init import fold_wgmma_accumulator_init
from ayla.ptx_codegen import make_ptx_kernel


def _module():
    return gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @test() grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {{
            %a: smem<64x16xbf16> = alloc() attrs={{id: 0}}
            %b: smem<256x16xbf16> = alloc() attrs={{id: 1}}
            %acc: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={{id: 2, init: 0.0}}
            for (%k: index = 0 to 4) attrs={{label: "k"}} {{
              matmul(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %acc: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">)
            }}
          }}
        }}"""))


def test_folds_zero_init_onto_first_peeled_matmul():
    module = _module()
    module = fold_wgmma_accumulator_init(module)

    [acc] = [
        op for op in find_ops(module, lambda op: isinstance(op, AllocOp))
        if op.result().name == "acc"
    ]
    matmuls = find_ops(module, lambda op: isinstance(op, MatmulOp))
    assert acc.init is None
    assert [op.init for op in matmuls] == [0.0, None]


def test_codegen_uses_a_static_scale_d_for_the_peeled_matmul():
    module = fold_wgmma_accumulator_init(_module())
    ptx = make_ptx_kernel(module).ptx

    assert ptx.count("wgmma.mma_async") == 2
    assert ptx.count("0, 1, 1, 0, 0;") == 1
    assert ptx.count("1, 1, 1, 0, 0;") == 1
    assert "0f00000000" not in ptx
    assert "scale_d" not in ptx
