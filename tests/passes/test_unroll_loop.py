"""Tests for the unroll_loop pass."""
import textwrap

from ayla.gpu_dialect import AllocOp, ForOp, MoveOp, gpu_dialect
from ayla.base.ir import iter_ops
from ayla.passes.canonicalize import canonicalize
from ayla.passes.folding import folding
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
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %s: smem<4x1xf32> = alloc() attrs={id: 0}
              %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              move(%s: smem<4x1xf32>, %t: gmem<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = unroll_loop(gpu_dialect.parse(module_str), unroll_factor=3, label_filter="k")

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 0 to 3 step 3) attrs={label: "k"} {
              %c0: index = index_constant() attrs={value: 0}
              %0: index = index_add(%k: index, %c0: index)
              %s: smem<4x1xf32> = alloc() attrs={id: 0}
              %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %0: index)
              move(%s: smem<4x1xf32>, %t: gmem<4x1xf32 strides=[4, 1]>)
              %c1: index = index_constant() attrs={value: 1}
              %1: index = index_add(%k: index, %c1: index)
              %s1: smem<4x1xf32> = alloc() attrs={id: 0}
              %t1: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %1: index)
              move(%s1: smem<4x1xf32>, %t1: gmem<4x1xf32 strides=[4, 1]>)
              %c2: index = index_constant() attrs={value: 2}
              %2: index = index_add(%k: index, %c2: index)
              %s2: smem<4x1xf32> = alloc() attrs={id: 0}
              %t2: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %2: index)
              move(%s2: smem<4x1xf32>, %t2: gmem<4x1xf32 strides=[4, 1]>)
            }
            %c3: index = index_constant() attrs={value: 3}
            %s3: smem<4x1xf32> = alloc() attrs={id: 0}
            %t3: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c3: index)
            move(%s3: smem<4x1xf32>, %t3: gmem<4x1xf32 strides=[4, 1]>)
          }
        }""")

    # LICM lifts the loop's copy into the block the tail's copy lives in, where
    # CSE can see the two are the same buffer.
    folded = canonicalize(licm(module))
    assert sum(isinstance(o, AllocOp) for o in iter_ops(folded)) == 1


def test_unroll_wider_than_the_trip_count_leaves_an_empty_main_loop():
    """Three iterations by four: no main loop survives, three tail copies.

    Nothing divides into the main loop, so it is emitted empty rather than
    special cased away, and folding erases it -- the same job it does for any
    other loop that never runs.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 0 to 3) attrs={label: "k"} {
              %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              move(%a: gmem<4x4xf32>, %t: gmem<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = unroll_loop(gpu_dialect.parse(module_str), unroll_factor=4, label_filter="k")

    loops = [o for o in iter_ops(module) if isinstance(o, ForOp)]
    assert len(loops) == 1 and loops[0].trip_count == 0

    folded = folding(module)
    assert not [o for o in iter_ops(folded) if isinstance(o, ForOp)]
    assert sum(isinstance(o, MoveOp) for o in iter_ops(folded)) == 3
