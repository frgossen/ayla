"""Tests for the LICM (loop-invariant code motion) pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.licm import licm



def test_no_hoist_k_dependent_ops():
    """Ops that depend on the loop IV stay inside."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<128x4096xbf16>, %s: smem_tensor<128x64xbf16 swizzle="128B">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %t: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%a: gmem_tensor<128x4096xbf16>, 0, %k: index)
              move(%s: smem_tensor<128x64xbf16 swizzle="128B">, %t: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={tma: True}
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
          kernel @test(%a: gmem_tensor<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%k: index = 0 to 64) {
              %s: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
              %r: rmem_tensor<128x64xbf16> = alloc() attrs={init: 0.0}
              move(%s: smem_tensor<128x64xbf16 swizzle="128B">, %a: gmem_tensor<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = licm(module)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem_tensor<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %s: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
            for (%k: index = 0 to 64) {
              %r: rmem_tensor<128x64xbf16> = alloc() attrs={init: 0.0}
              move(%s: smem_tensor<128x64xbf16 swizzle="128B">, %a: gmem_tensor<128x64xbf16>) attrs={tma: True}
            }
          }
        }""")
    assert text == expected
