"""Tests for the DCE pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.dce import dce


def test_dce_removes_unused_index_ops():
    """Dead index ops are removed, used ones are kept."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %dead: index = index_constant() attrs={value: 999}
            %also_dead: index = index_mul(%cy: index, %dead: index)
            %val: rmem_tensor<1xf32> = alloc()
            move(%val: rmem_tensor<1xf32>, %a: gmem_tensor<4096x4096xf32>, %cx_offset: index)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = dce(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %dead: index = index_constant() attrs={value: 999}
            %val: rmem_tensor<1xf32> = alloc()
            move(%val: rmem_tensor<1xf32>, %a: gmem_tensor<4096x4096xf32>, %cx_offset: index)
          }
        }""")
