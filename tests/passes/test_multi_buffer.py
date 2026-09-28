"""Tests for the multi_buffer pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.multi_buffer import multi_buffer


def test_clones_the_buffer_and_selects_per_iteration():
    """The allocation is duplicated in place; the loop is left alone."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              %s: smem<4x4xf32> = alloc() attrs={id: 0, label: "lhs"}
              move(%s: smem<4x4xf32>, %t: gmem<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = multi_buffer(
        module,
        depth=2,
        label_filter="lhs",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              %s: smem<4x4xf32> = alloc() attrs={id: 0, label: "lhs"}
              %s1: smem<4x4xf32> = alloc() attrs={id: 1, label: "lhs"}
              %c0: index = index_constant() attrs={value: 0}
              %0: index = index_add(%k: index, %c0: index)
              %c2: index = index_constant() attrs={value: 2}
              %1: index = index_mod(%0: index, %c2: index)
              %2: smem<4x4xf32> = select(%1: index, %s: smem<4x4xf32>, %s1: smem<4x4xf32>)
              move(%2: smem<4x4xf32>, %t: gmem<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")
