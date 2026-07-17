"""Tests for the unroll_loop pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.unroll_loop import unroll_loop


def test_unroll_by_2():
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %smem: smem_tensor<4x4xf32> = alloc()
            for (%k: index = 0 to 6 step 3) attrs={label: "k"} {
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%smem: smem_tensor<4x4xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = gpu_dialect.parse(module_str)
    unrolled = unroll_loop(module, unroll_factor=2, label_filter="k")
    text = repr(unrolled)
    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %smem: smem_tensor<4x4xf32> = alloc()
            for (%k: index = 0 to 6 step 6) attrs={label: "k"} {
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%smem: smem_tensor<4x4xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
              %c3: index = index_constant() attrs={value: 3}
              %0: index = index_add(%k: index, %c3: index)
              %t1: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %0: index)
              move(%smem: smem_tensor<4x4xf32>, %t1: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")
    assert text == expected
