"""Tests for the unroll_loop pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.unroll_loop import unroll_loop
from ayla.testing import assert_contains_in_order


def test_unroll_factor_equal_to_trip_count():
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


def test_unroll_factor_less_than_trip_count_indexes_tail():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<5x1xf32>) {
            for (%k: index = 0 to 5) attrs={label: "k"} {
              %t: gmem_tensor<1x1xf32> = tile(%a: gmem_tensor<5x1xf32>, %k: index, 0)
            }
          }
        }"""))
    text = repr(unroll_loop(module, unroll_factor=3, label_filter="k"))
    # Main loop covers %k = 0, 1, 2 (offset from the loop induction variable);
    # the tail covers the leftover iterations %k = 3, 4 as constants.
    assert_contains_in_order(
        text,
        'for (%k: index = 0 to 3 step 3) attrs={label: "k"} {',
        "%t: gmem_tensor<1x1xf32> = tile(%a: gmem_tensor<5x1xf32>, %k: index, 0)",
        "%c1: index = index_constant() attrs={value: 1}",
        "%0: index = index_add(%k: index, %c1: index)",
        "%c2: index = index_constant() attrs={value: 2}",
        "%1: index = index_add(%k: index, %c2: index)",
        "%c3: index = index_constant() attrs={value: 3}",
        "%c4: index = index_constant() attrs={value: 4}",
    )
    assert text.count("gmem_tensor<1x1xf32> = tile(") == 5


def test_unroll_factor_greater_than_trip_count_fully_unrolls():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<6x1xf32>) {
            for (%k: index = 2 to 6 step 2) attrs={label: "k"} {
              %t: gmem_tensor<1x1xf32> = tile(%a: gmem_tensor<6x1xf32>, %k: index, 0)
            }
          }
        }"""))
    text = repr(unroll_loop(module, unroll_factor=3, label_filter="k"))
    # No loop remains; every iteration %k = 2, 4 becomes a constant index.
    assert "for (" not in text
    assert_contains_in_order(
        text,
        "%c2: index = index_constant() attrs={value: 2}",
        "%t: gmem_tensor<1x1xf32> = tile(%a: gmem_tensor<6x1xf32>, %c2: index, 0)",
        "%c4: index = index_constant() attrs={value: 4}",
        "%t1: gmem_tensor<1x1xf32> = tile(%a: gmem_tensor<6x1xf32>, %c4: index, 0)",
    )


def test_unroll_remainder_reuses_first_alloc():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %smem: smem_tensor<4x1xf32> = alloc()
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%smem: smem_tensor<4x1xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }"""))
    text = repr(unroll_loop(module, unroll_factor=3, label_filter="k"))
    # Three main copies allocate; the single tail iteration reuses the first.
    assert text.count("smem_tensor<4x1xf32> = alloc()") == 3
    assert_contains_in_order(
        text,
        "%smem: smem_tensor<4x1xf32> = alloc()",
        'for (%k: index = 0 to 3 step 3) attrs={label: "k"} {',
        "%c3: index = index_constant() attrs={value: 3}",
    )
    assert text.count("move(") == 4


def test_unroll_remainder_does_not_hoist_initialized_allocations():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test() {
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


def test_plain_unrolling_shares_the_loop_local_buffer():
    """Without clone_allocs every copy uses the one allocation.

    Five iterations unrolled by two, so the tail runs too: it already reused
    the first copy's buffer, and now the main copies do as well. The default
    duplicates it instead, which multi-buffers the copies as a side effect.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            for (%k: index = 0 to 5 step 1) attrs={label: "k"} {
              %s: smem_tensor<4x4xf32> = alloc()
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%s: smem_tensor<4x4xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
          }
        }""")

    module = unroll_loop(
        gpu_dialect.parse(module_str),
        unroll_factor=2,
        label_filter="k",
        clone_allocs=False,
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %s: smem_tensor<4x4xf32> = alloc()
            for (%k: index = 0 to 4 step 2) attrs={label: "k"} {
              %t: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %k: index)
              move(%s: smem_tensor<4x4xf32>, %t: gmem_tensor<4x1xf32 strides=[4, 1]>)
              %c1: index = index_constant() attrs={value: 1}
              %0: index = index_add(%k: index, %c1: index)
              %t1: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %0: index)
              move(%s: smem_tensor<4x4xf32>, %t1: gmem_tensor<4x1xf32 strides=[4, 1]>)
            }
            %c4: index = index_constant() attrs={value: 4}
            %t2: gmem_tensor<4x1xf32 strides=[4, 1]> = tile(%a: gmem_tensor<4x4xf32>, 0, %c4: index)
            move(%s: smem_tensor<4x4xf32>, %t2: gmem_tensor<4x1xf32 strides=[4, 1]>)
          }
        }""")
