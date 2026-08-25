"""Tests for the unroll_loop pass."""
import textwrap

from ayla.gpu_dialect import AllocOp, gpu_dialect
from ayla.base.ir import iter_ops
from ayla.passes.canonicalize import canonicalize
from ayla.passes.licm import licm
from ayla.passes.unroll_loop import unroll_loop


def test_unroll_with_a_tail_and_a_shared_allocation():
    """Four iterations by three: three copies, one tail, one buffer.

    Every copy offsets the induction variable by i * step -- including the
    first, whose + 0 folding removes -- and the leftover iteration becomes
    straight-line code after the loop with a constant index.
    The allocation is cloned along with everything else. It keeps its id, so
    the copies stay interchangeable and fold back into the one buffer the loop
    started with -- kernels lift their buffers out with LICM beforehand rather
    than rely on that, since multi-buffering a loop is multi_buffer's job.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %s: smem_tensor<4x1xf32> = alloc() attrs={id: 0}
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%s: smem_tensor<4x1xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = unroll_loop(gpu_dialect.parse(module_str), unroll_factor=3, label_filter="k")

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            for (%k: index = 0 to 3 step 3) attrs={label: "k"} {
              %c0: index = index_constant() attrs={value: 0}
              %0: index = index_add(%k: index, %c0: index)
              %s: smem_tensor<4x1xf32> = alloc() attrs={id: 0}
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %0: index)
              move(%s: smem_tensor<4x1xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
              %c1: index = index_constant() attrs={value: 1}
              %1: index = index_add(%k: index, %c1: index)
              %s1: smem_tensor<4x1xf32> = alloc() attrs={id: 0}
              %t1: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %1: index)
              move(%s1: smem_tensor<4x1xf32>, %t1: gmem_tensor<4x1xf32 strides=[4, 1]>)
              %c2: index = index_constant() attrs={value: 2}
              %2: index = index_add(%k: index, %c2: index)
              %s2: smem_tensor<4x1xf32> = alloc() attrs={id: 0}
              %t2: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %2: index)
              move(%s2: smem_tensor<4x1xf32>, %t2: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
            %c3: index = index_constant() attrs={value: 3}
            %s3: smem_tensor<4x1xf32> = alloc() attrs={id: 0}
            %t3: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %c3: index)
            move(%s3: smem_tensor<4x1xf32>, %t3: gmem_tensor<4x1xf32 strides=[4, 1]>)
          }
        }""")

    # LICM lifts the loop's copy into the block the tail's copy lives in, where
    # CSE can see the two are the same buffer.
    folded = canonicalize(licm(module))
    assert sum(isinstance(o, AllocOp) for o in iter_ops(folded)) == 1
