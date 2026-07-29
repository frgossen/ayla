"""Tests for the unroll_loop pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.unroll_loop import unroll_loop


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


def test_unroll_remainder_reuses_first_alloc():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %smem: smem_tensor<4x1xf32> = alloc()
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%smem: smem_tensor<4x1xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }"""))
    text = repr(unroll_loop(module, unroll_factor=3, label_filter="k"))
    assert text.count("smem_tensor<4x1xf32> = alloc()") == 3
    assert "for (%k: index = 0 to 3 step 3)" in text
    assert "index_constant() attrs={value: 3}" in text
    assert text.count("move(") == 4


def test_unroll_remainder_without_allocations():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<5x1xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            for (%k: index = 0 to 5) attrs={label: "k"} {
              %t: gmem_tensor<1x1xf32> = tile(%a: gmem_tensor<5x1xf32>, %k: index, 0)
            }
          }
        }"""))
    text = repr(unroll_loop(module, unroll_factor=3, label_filter="k"))
    assert "for (%k: index = 0 to 3 step 3)" in text
    assert "index_constant() attrs={value: 3}" in text
    assert "index_constant() attrs={value: 4}" in text
    assert text.count("gmem_tensor<1x1xf32> = tile(") == 5


def test_unroll_remainder_does_not_hoist_initialized_allocations():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test() grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            for (%k: index = 0 to 5) attrs={label: "k"} {
              %r: rmem_tensor<1xf32> = alloc() attrs={init: 0.0}
            }
          }
        }"""))
    text = repr(unroll_loop(module, unroll_factor=3, label_filter="k"))
    assert text.count("rmem_tensor<1xf32> = alloc() attrs={init: 0.0}") == 5
    loop_start = text.index("for (%k: index = 0 to 3 step 3)")
    first_alloc = text.index("rmem_tensor<1xf32> = alloc()", loop_start)
    assert first_alloc > loop_start
