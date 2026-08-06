"""Tests for the `folding` pass."""
import textwrap

import pytest

from ayla.base.ir import find_ops
from ayla.gpu_dialect import AllocOp, MatmulOp, gpu_dialect
from ayla.passes.folding import Folding, folding


def test_fold_constants():
    """Constant folding eliminates trivial index arithmetic."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c0: index = index_constant() attrs={value: 0}
            %c4096: index = index_constant() attrs={value: 4096}
            %zero_offset: index = index_mul(%c0: index, %c4096: index)
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %sum: index = index_add(%cx_offset: index, %zero_offset: index)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    before = repr(module)
    assert "index_mul(%c0: index, %c4096: index)" in before
    assert "index_add" in before

    module = folding(module)
    after = repr(module)

    # 0 * 4096 folds to 0, cx_offset + 0 folds to cx_offset.
    assert after == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c0: index = index_constant() attrs={value: 0}
            %c4096: index = index_constant() attrs={value: 4096}
            %c0_1: index = index_constant() attrs={value: 0}
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
          }
        }""")


@pytest.mark.parametrize("nested,folds", [(False, True), (True, False)])
def test_fold_accumulator_init(nested, folds):
    matmul = (
        "matmul(%a: smem_tensor<64x16xbf16>, "
        "%b: smem_tensor<256x16xbf16>, "
        "%acc: wrmem_tensor<64x256xf32 warps=4 "
        "layout=\"wgmma_layout_0\">)"
    )
    body = f"for (%kk: index = 0 to 4) {{ {matmul} }}" if nested else matmul
    module = gpu_dialect.parse(f"""
        module {{
          kernel @k() {{
            %a: smem_tensor<64x16xbf16> = alloc()
            %b: smem_tensor<256x16xbf16> = alloc()
            %acc: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={{init: 0.0}}
            for (%k: index = 0 to 4) {{ {body} }}
          }}
        }}
    """)
    Folding(fold_accumulator_init=True).run(module)
    [alloc] = find_ops(module, lambda op: isinstance(op, AllocOp)
                       and op.result().type.name == "wrmem_tensor")
    [matmul] = find_ops(module, lambda op: isinstance(op, MatmulOp))
    assert (alloc.init is None) is folds
    assert (matmul.scale_d is not None) is folds
