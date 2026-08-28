"""Tests for the LICM (loop-invariant code motion) pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.licm import licm



def test_no_hoist_k_dependent_ops():
    """Ops that depend on the loop IV stay inside."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>, %s: smem<128x64xbf16 swizzle="128B">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %t: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %k: index)
              move(%s: smem<128x64xbf16 swizzle="128B">, %t: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
            }
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse(module_str)
    before = repr(module)
    module = licm(module)
    after = repr(module)
    assert before == after


def test_hoist_alloc_without_init():
    """Static allocs (no init) are hoisted; allocs with init stay."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
              %r: rmem<128x64xbf16> = alloc() attrs={id: 1, init: 0.0}
              move(%s: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = licm(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            for (%k: index = 0 to 64) {
              %r: rmem<128x64xbf16> = alloc() attrs={id: 1, init: 0.0}
              move(%s: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")
    assert text == expected


def test_hoist_out_of_a_ploop():
    """A ploop is a loop body like any other: what does not use the iv leaves."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>) grid_dim=[32, 1, 1] block_dim=[128, 1, 1] {
            ploop block.x (%bx: index = 0 to 32) {
              %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
              %t: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %bx: index)
              move(%s: smem<128x64xbf16 swizzle="128B">, %t: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
            }
          }
        }""")  # noqa: E501

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<128x4096xbf16>) grid_dim=[32, 1, 1] block_dim=[128, 1, 1] {
            %s: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            ploop block.x (%bx: index = 0 to 32) {
              %t: gmem<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem<128x4096xbf16>, 0, %bx: index)
              move(%s: smem<128x64xbf16 swizzle="128B">, %t: gmem<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
            }
          }
        }""")  # noqa: E501

    assert repr(licm(gpu_dialect.parse(module_str))) == expected
