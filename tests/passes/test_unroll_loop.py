"""Tests for the unroll_loop pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.unroll_loop import unroll_loop


def test_unroll_with_a_tail_and_a_shared_allocation():
    """Four iterations by three: three copies, one tail, one buffer.

    Every copy offsets the induction variable by i * step -- including the
    first, whose + 0 folding removes -- and the leftover iteration becomes
    straight-line code after the loop with a constant index.
    The allocation is lifted out and shared rather than copied per iteration --
    copying it would multi-buffer the loop, which is multi_buffer's job.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %s: smem_tensor<4x1xf32> = alloc()
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%s: smem_tensor<4x1xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = unroll_loop(gpu_dialect.parse(module_str), unroll_factor=3, label_filter="k")

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            %s: smem_tensor<4x1xf32> = alloc()
            for (%k: index = 0 to 3 step 3) attrs={label: "k"} {
              %c0: index = index_constant() attrs={value: 0}
              %0: index = index_add(%k: index, %c0: index)
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %0: index)
              move(%s: smem_tensor<4x1xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
              %c1: index = index_constant() attrs={value: 1}
              %1: index = index_add(%k: index, %c1: index)
              %t1: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %1: index)
              move(%s: smem_tensor<4x1xf32>, %t1: gmem_tensor<4x1xf32 strides=[4, 1]>)
              %c2: index = index_constant() attrs={value: 2}
              %2: index = index_add(%k: index, %c2: index)
              %t2: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %2: index)
              move(%s: smem_tensor<4x1xf32>, %t2: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
            %c3: index = index_constant() attrs={value: 3}
            %t3: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %c3: index)
            move(%s: smem_tensor<4x1xf32>, %t3: gmem_tensor<4x1xf32 strides=[4, 1]>)
          }
        }""")
