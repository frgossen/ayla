"""Tests for the CSE pass."""
import textwrap

from ayla.gpu_dialect import IndexConstantOp, ModuleOp, gpu_dialect
from ayla.base.ir import find_ops_of_type
from ayla.passes.cse import cse


def test_cse_deduplicates_index_constants():
    """Two index_constant(64) ops in the same block should be merged."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c64a: index = index_constant() attrs={value: 64}
            %offset_a: index = index_mul(%cx: index, %c64a: index)
            %c64b: index = index_constant() attrs={value: 64}
            %offset_b: index = index_mul(%cx: index, %c64b: index)
            %val: rmem<1xf32> = alloc() attrs={id: 0}
            move(%val: rmem<1xf32>, %a: gmem<4096x4096xf32>, %offset_b: index)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    assert isinstance(module, ModuleOp)
    before_constants = find_ops_of_type(module, IndexConstantOp)
    assert sum(1 for op in before_constants if op.value == 64) == 2

    module = cse(module)
    after_constants = find_ops_of_type(module, IndexConstantOp)
    assert sum(1 for op in after_constants if op.value == 64) == 1

    # c64b is CSE'd to c64a, offset_b is CSE'd to offset_a.
    # The load now references offset_a. offset_a is kept (dead but no DCE yet).
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c64a: index = index_constant() attrs={value: 64}
            %offset_a: index = index_mul(%cx: index, %c64a: index)
            %val: rmem<1xf32> = alloc() attrs={id: 0}
            move(%val: rmem<1xf32>, %a: gmem<4096x4096xf32>, %offset_a: index)
          }
        }""")


def test_cse_deduplicates_allocations_by_id():
    """Allocations merge iff they name the same buffer.

    An allocation is an ordinary value-identical op; the id is what tells two
    of them apart. Copies of one allocation carry its id and fold back
    together, while separate allocations keep their own buffers.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            %same0: smem<4x4xf32> = alloc() attrs={id: 0}
            %same1: smem<4x4xf32> = alloc() attrs={id: 0}
            %other: smem<4x4xf32> = alloc() attrs={id: 1}
            %init0: rmem<2x2xf32> = alloc() attrs={id: 2, init: 0.0}
            %init1: rmem<2x2xf32> = alloc() attrs={id: 2, init: 0.0}
          }
        }""")

    module = gpu_dialect.parse(module_str)
    assert isinstance(module, ModuleOp)
    text = repr(cse(module))

    assert text.count("alloc() attrs={id: 0}") == 1
    assert text.count("alloc() attrs={id: 1}") == 1
    # An initialiser writes the buffer, so two of them are not interchangeable
    # even sharing an id -- unrolling clones them, and each copy must re-zero.
    assert text.count("alloc() attrs={id: 2, init: 0.0}") == 2
