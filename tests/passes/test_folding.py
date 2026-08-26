"""Tests for the `folding` pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.canonicalize import canonicalize
from ayla.passes.folding import folding
from ayla.passes.unroll_loop import unroll_loop


def test_fold_constants():
    """Constant folding eliminates trivial index arithmetic."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c0: index = index_constant() attrs={value: 0}
            %c4096: index = index_constant() attrs={value: 4096}
            %zero_offset: index = index_mul(%c0: index, %c4096: index)
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %sum: index = index_add(%cx_offset: index, %zero_offset: index)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    before = repr(module)
    assert "index_mul(%c0: index, %c4096: index)" in before
    assert "index_add" in before

    module = folding(module)
    after = repr(module)

    # 0 * 4096 folds to 0, cx_offset + 0 folds to cx_offset.
    assert after == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4096x4096xf32>) {
            %cx: index, %cy: index, %cz: index = block_ids()
            %c0: index = index_constant() attrs={value: 0}
            %c4096: index = index_constant() attrs={value: 4096}
            %c0_1: index = index_constant() attrs={value: 0}
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
          }
        }""")


def test_per_iteration_buffer_choice_folds_after_unrolling():
    """The capability select and the iv congruence exist for.

    A loop that picks its buffer with ``(k % (depth*step)) / step`` is
    double-buffered explicitly in the IR. Unrolling multiplies the step, which
    makes each copy's index static, so folding resolves every select to a
    concrete buffer and the arithmetic disappears -- the same result unrolling
    currently gets by duplicating allocations behind CSE's back.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            %b0: smem_tensor<4x4xf32> = alloc() attrs={id: 0}
            %b1: smem_tensor<4x4xf32> = alloc() attrs={id: 1}
            for (%k: index = 0 to 24 step 3) attrs={label: "k"} {
              %c6: index = index_constant() attrs={value: 6}
              %r: index = index_mod(%k: index, %c6: index)
              %c3: index = index_constant() attrs={value: 3}
              %sel: index = index_div(%r: index, %c3: index)
              %buf: smem_tensor<4x4xf32> = select(%sel: index, %b0: smem_tensor<4x4xf32>, %b1: smem_tensor<4x4xf32>)
              move(%buf: smem_tensor<4x4xf32>, %a: gmem_tensor<4x4xf32>)
            }
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = canonicalize(folding(module))

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            %b0: smem_tensor<4x4xf32> = alloc() attrs={id: 0}
            %b1: smem_tensor<4x4xf32> = alloc() attrs={id: 1}
            for (%k: index = 0 to 24 step 6) attrs={label: "k"} {
              move(%b0: smem_tensor<4x4xf32>, %a: gmem_tensor<4x4xf32>)
              move(%b1: smem_tensor<4x4xf32>, %a: gmem_tensor<4x4xf32>)
            }
          }
        }""")


def test_buffer_choice_stays_dynamic_when_depth_does_not_divide_the_unroll():
    """Depth 3 with unroll 2: the index still varies, so nothing folds.

    The multi-buffering pass will have to reject this up front -- there is no
    static answer, and the backend cannot select between buffers at runtime.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            %b0: smem_tensor<4x4xf32> = alloc() attrs={id: 0}
            %b1: smem_tensor<4x4xf32> = alloc() attrs={id: 1}
            %b2: smem_tensor<4x4xf32> = alloc() attrs={id: 2}
            for (%k: index = 0 to 24 step 3) attrs={label: "k"} {
              %c9: index = index_constant() attrs={value: 9}
              %r: index = index_mod(%k: index, %c9: index)
              %c3: index = index_constant() attrs={value: 3}
              %sel: index = index_div(%r: index, %c3: index)
              %buf: smem_tensor<4x4xf32> = select(%sel: index, %b0: smem_tensor<4x4xf32>, %b1: smem_tensor<4x4xf32>, %b2: smem_tensor<4x4xf32>)
              move(%buf: smem_tensor<4x4xf32>, %a: gmem_tensor<4x4xf32>)
            }
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = canonicalize(folding(module))

    assert "select(" in repr(module)


def test_iv_congruence_uses_the_lower_bound_and_stays_in_its_loop():
    """The residue is ``lb % m``, and the fact does not outlive the loop."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<4x4xf32>) {
            for (%k: index = 5 to 29 step 6) attrs={label: "k"} {
              %c3: index = index_constant() attrs={value: 3}
              %inner: index = index_mod(%k: index, %c3: index)
            }
            %cx: index, %cy: index, %cz: index = block_ids()
            %c3b: index = index_constant() attrs={value: 3}
            %outer: index = index_mod(%cx: index, %c3b: index)
          }
        }""")

    module = folding(gpu_dialect.parse(module_str))
    after = repr(module)

    # 5 % 3 == 2, on every iteration.
    assert "index_constant() attrs={value: 2}" in after
    assert "index_mod(%k: index" not in after
    # The block id is not an induction variable, so its mod is untouched.
    assert "index_mod(%cx: index, %c3b: index)" in after


def test_distribute_constant_multiply_over_affine_offset():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test() {
            %bx: index, %by: index, %bz: index = block_ids()
            %c3: index = index_constant() attrs={value: 3}
            %offset: index = index_add(%bx: index, %c3: index)
            %c64: index = index_constant() attrs={value: 64}
            %coordinate: index = index_mul(%offset: index, %c64: index)
          }
        }"""))
    folding(module)
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test() {
            %bx: index, %by: index, %bz: index = block_ids()
            %c3: index = index_constant() attrs={value: 3}
            %offset: index = index_add(%bx: index, %c3: index)
            %c64: index = index_constant() attrs={value: 64}
            %c64_1: index = index_constant() attrs={value: 64}
            %0: index = index_mul(%bx: index, %c64_1: index)
            %c192: index = index_constant() attrs={value: 192}
            %1: index = index_add(%0: index, %c192: index)
          }
        }""")
