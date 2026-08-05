"""Tests for the `folding` pass."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import IndexConstantOp, IndexMulOp, gpu_dialect
from ayla.passes.folding import folding


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


def test_distribute_constant_multiply_over_affine_offset():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test() {
            %bx: index, %by: index, %bz: index = block_ids()
            %c3: index = index_constant() attrs={value: 3}
            %offset: index = index_add(%bx: index, %c3: index)
            %c64: index = index_constant() attrs={value: 64}
            %coordinate: index = index_mul(%offset: index, %c64: index)
          }
        }"""))
    folding(module)
    assert len(find_ops(module, lambda op: isinstance(op, IndexMulOp))) == 1
    constants = find_ops(module, lambda op: isinstance(op, IndexConstantOp))
    assert 192 in [op.value for op in constants]
