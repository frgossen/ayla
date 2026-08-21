"""Tests for the CSE pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect, IndexConstantOp
from ayla.base.ir import find_ops
from ayla.passes.cse import cse


def test_cse_deduplicates_index_constants():
    """Two index_constant(64) ops in the same block should be merged."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c64a: index = index_constant() attrs={value: 64}
            %offset_a: index = index_mul(%cx: index, %c64a: index)
            %c64b: index = index_constant() attrs={value: 64}
            %offset_b: index = index_mul(%cx: index, %c64b: index)
            %val: rmem_tensor<1xf32> = alloc()
            move(%val: rmem_tensor<1xf32>, %a: gmem_tensor<4096x4096xf32>, %offset_b: index)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    before_constants = find_ops(module, lambda op: isinstance(op, IndexConstantOp))
    assert sum(1 for op in before_constants if op.value == 64) == 2

    module = cse(module)
    after_constants = find_ops(module, lambda op: isinstance(op, IndexConstantOp))
    assert sum(1 for op in after_constants if op.value == 64) == 1

    # c64b is CSE'd to c64a, offset_b is CSE'd to offset_a.
    # The load now references offset_a. offset_a is kept (dead but no DCE yet).
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c64a: index = index_constant() attrs={value: 64}
            %offset_a: index = index_mul(%cx: index, %c64a: index)
            %val: rmem_tensor<1xf32> = alloc()
            move(%val: rmem_tensor<1xf32>, %a: gmem_tensor<4096x4096xf32>, %offset_a: index)
          }
        }""")


def test_cse_deduplicates_allocations_by_uniq():
    """Allocations merge iff they name the same buffer.

    An allocation is an ordinary value-identical op; `uniq` is what tells two
    of them apart. Clones of one allocation carry its id and fold back
    together, while separate allocations keep their own buffers.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            %same0: smem_tensor<4x4xf32> = alloc() attrs={uniq: 0}
            %same1: smem_tensor<4x4xf32> = alloc() attrs={uniq: 0}
            %other: smem_tensor<4x4xf32> = alloc() attrs={uniq: 1}
            %opaque: smem_tensor<4x4xf32> = alloc()
            %opaque2: smem_tensor<4x4xf32> = alloc()
          }
        }""")

    text = repr(cse(gpu_dialect.parse(module_str)))

    # The pair sharing uniq 0 collapses to one.
    assert text.count("alloc() attrs={uniq: 0}") == 1
    assert text.count("alloc() attrs={uniq: 1}") == 1
    # An allocation with no uniq names nothing, so it never merges.
    assert text.count("= alloc()\n") == 2
