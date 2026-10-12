"""Tests for the `folding` pass."""
import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import TileOp, gpu_dialect
from ayla.passes.canonicalize import Canonicalize
from ayla.passes.folding import folding
from ayla.passes.unroll_loop import unroll_loop


def test_fold_constants():
    """Constant folding eliminates trivial index arithmetic."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) {
            %cx: index = block_id.x
            %c0: index = index_constant() attrs={value: 0}
            %c4096: index = index_constant() attrs={value: 4096}
            %zero_offset: index = index_mul(%c0: index, %c4096: index)
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
            %sum: index = index_add(%cx_offset: index, %zero_offset: index)
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    before = repr(module)
    assert "index_mul(%c0: index, %c4096: index)" in before
    assert "index_add" in before

    module = folding(module)
    after = repr(module)

    # 0 * 4096 folds to 0, cx_offset + 0 folds to cx_offset.
    assert after == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4096x4096xf32>) {
            %cx: index = block_id.x
            %c0: index = index_constant() attrs={value: 0}
            %c4096: index = index_constant() attrs={value: 4096}
            %c0_1: index = index_constant() attrs={value: 0}
            %c64: index = index_constant() attrs={value: 64}
            %cx_offset: index = index_mul(%cx: index, %c64: index)
          }
        }""")


def test_fold_constant_tile_indices_to_static_indices():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) {
            %c3: index = index_constant() attrs={value: 3}
            %tile: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = tile(%a: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, 0, %c3: index)
          }
        }"""))  # noqa: E501

    folding(module)

    [tile] = find_ops_of_type(module, TileOp)
    assert tile.static_indices == [0, 3]
    assert tile.indices == [0, 3]


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
          kernel @test(%a: gmem<4x4xf32>) {
            %b0: smem<4x4xf32> = alloc() attrs={id: 0}
            %b1: smem<4x4xf32> = alloc() attrs={id: 1}
            for (%k: index = 0 to 24 step 3) attrs={label: "k"} {
              %c6: index = index_constant() attrs={value: 6}
              %r: index = index_mod(%k: index, %c6: index)
              %c3: index = index_constant() attrs={value: 3}
              %sel: index = index_div(%r: index, %c3: index)
              %buf: smem<4x4xf32> = select(%sel: index, %b0: smem<4x4xf32>, %b1: smem<4x4xf32>)
              move(%buf: smem<4x4xf32>, %a: gmem<4x4xf32>)
            }
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = Canonicalize().run(folding(module))

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            %b0: smem<4x4xf32> = alloc() attrs={id: 0}
            %b1: smem<4x4xf32> = alloc() attrs={id: 1}
            for (%k: index = 0 to 24 step 6) attrs={label: "k"} {
              move(%b0: smem<4x4xf32>, %a: gmem<4x4xf32>)
              move(%b1: smem<4x4xf32>, %a: gmem<4x4xf32>)
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
          kernel @test(%a: gmem<4x4xf32>) {
            %b0: smem<4x4xf32> = alloc() attrs={id: 0}
            %b1: smem<4x4xf32> = alloc() attrs={id: 1}
            %b2: smem<4x4xf32> = alloc() attrs={id: 2}
            for (%k: index = 0 to 24 step 3) attrs={label: "k"} {
              %c9: index = index_constant() attrs={value: 9}
              %r: index = index_mod(%k: index, %c9: index)
              %c3: index = index_constant() attrs={value: 3}
              %sel: index = index_div(%r: index, %c3: index)
              %buf: smem<4x4xf32> = select(%sel: index, %b0: smem<4x4xf32>, %b1: smem<4x4xf32>, %b2: smem<4x4xf32>)
              move(%buf: smem<4x4xf32>, %a: gmem<4x4xf32>)
            }
          }
        }""")

    module = gpu_dialect.parse_module(module_str)
    module = unroll_loop(module, unroll_factor=2, label_filter="k")
    module = Canonicalize().run(folding(module))

    assert "select(" in repr(module)


def test_iv_congruence_uses_the_lower_bound_and_stays_in_its_loop():
    """The residue is ``lb % m``, and the fact does not outlive the loop."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 5 to 29 step 6) attrs={label: "k"} {
              %c3: index = index_constant() attrs={value: 3}
              %inner: index = index_mod(%k: index, %c3: index)
            }
            %cx: index = block_id.x
            %c3b: index = index_constant() attrs={value: 3}
            %outer: index = index_mod(%cx: index, %c3b: index)
          }
        }""")

    module = folding(gpu_dialect.parse_module(module_str))
    after = repr(module)

    # 5 % 3 == 2, on every iteration.
    assert "index_constant() attrs={value: 2}" in after
    assert "index_mod(%k: index" not in after
    # The block id is not an induction variable, so its mod is untouched.
    assert "index_mod(%cx: index, %c3b: index)" in after


def test_distribute_constant_multiply_over_affine_offset():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test() {
            %bx: index = block_id.x
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
            %bx: index = block_id.x
            %c3: index = index_constant() attrs={value: 3}
            %offset: index = index_add(%bx: index, %c3: index)
            %c64: index = index_constant() attrs={value: 64}
            %c64_1: index = index_constant() attrs={value: 64}
            %0: index = index_mul(%bx: index, %c64_1: index)
            %c192: index = index_constant() attrs={value: 192}
            %1: index = index_add(%0: index, %c192: index)
          }
        }""")


def test_for_that_never_runs_is_erased():
    """An empty range means the body never executes, so the loop goes away."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 8 to 8) attrs={label: "empty"} {
              %v: rmem<1xf32> = alloc()
            }
            for (%j: index = 5 to 3 step 2) attrs={label: "backwards"} {
              %w: rmem<1xf32> = alloc()
            }
            %c: index = index_constant() attrs={value: 7}
          }
        }""")

    after = repr(folding(gpu_dialect.parse_module(module_str)))

    assert "for (" not in after
    assert "alloc()" not in after
    # ...and the rest of the kernel is untouched.
    assert "index_constant() attrs={value: 7}" in after


def test_launch_bounded_ids_fold_mod_and_div():
    """An ID below the modulus is its own remainder and divides to zero."""
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<2x16xf32>) grid_dim=[2, 1, 1] block_dim=[64, 1, 1] {
            %bx: index = block_id.x
            %tx: index = thread_id.x
            %c2: index = index_constant() attrs={value: 2}
            %c32: index = index_constant() attrs={value: 32}
            %block: index = index_mod(%bx: index, %c2: index)
            %wave: index = index_div(%bx: index, %c2: index)
            %warp: index = index_div(%tx: index, %c32: index)
            %warp_in_pair: index = index_mod(%warp: index, %c2: index)
            %t0: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %block: index, 0)
            %t1: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %wave: index, 0)
            %t2: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %warp_in_pair: index, 0)
          }
        }"""))  # noqa: E501

    module = folding(module)

    # bx < 2: bx % 2 is bx and bx / 2 is 0 (which then becomes a static tile
    # index). tx < 64 gives tx / 32 < 2, so its % 2 goes too, but the / 32
    # itself stays because tx can reach 32.
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<2x16xf32>) grid_dim=[2, 1, 1] block_dim=[64, 1, 1] {
            %bx: index = block_id.x
            %tx: index = thread_id.x
            %c2: index = index_constant() attrs={value: 2}
            %c32: index = index_constant() attrs={value: 32}
            %c0: index = index_constant() attrs={value: 0}
            %warp: index = index_div(%tx: index, %c32: index)
            %t0: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %bx: index, 0)
            %0: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, 0, 0)
            %t2: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %warp: index, 0)
          }
        }""")  # noqa: E501


def test_launch_bound_keeps_mod_when_extent_exceeds_modulus():
    """bx reaches 3 and by is unbounded, so neither % 2 can fold."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<2x16xf32>) grid_dim=[4, None, 1] {
            %bx: index = block_id.x
            %by: index = block_id.y
            %c2: index = index_constant() attrs={value: 2}
            %x: index = index_mod(%bx: index, %c2: index)
            %y: index = index_mod(%by: index, %c2: index)
            %tx: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %x: index, 0)
            %ty: gmem<1x16xf32> = tile(%a: gmem<2x16xf32>, %y: index, 0)
          }
        }""")

    module = folding(gpu_dialect.parse_module(module_str))

    assert repr(module) == module_str


def test_loop_bound_folds_mod_of_induction_variable():
    """i < 4, so i % 4 is i and the tile indexes the loop variable directly."""
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x16xf32>) {
            %c4: index = index_constant() attrs={value: 4}
            for (%i: index = 0 to 4) {
              %r: index = index_mod(%i: index, %c4: index)
              %t: gmem<1x16xf32> = tile(%a: gmem<4x16xf32>, %r: index, 0)
            }
          }
        }"""))

    module = folding(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x16xf32>) {
            %c4: index = index_constant() attrs={value: 4}
            for (%i: index = 0 to 4) {
              %t: gmem<1x16xf32> = tile(%a: gmem<4x16xf32>, %i: index, 0)
            }
          }
        }""")
