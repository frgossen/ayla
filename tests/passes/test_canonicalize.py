"""Tests for the canonicalize pass (folding + CSE + DCE)."""
import textwrap

from ayla.passes.canonicalize import Canonicalize
from ayla.gpu_dialect import gpu_dialect


def test_inline_single_trip_for():
    """A for loop with trip count 1 is inlined; a second pass folds constants."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x64xbf16 swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%i: index = 0 to 2 step 2) {
              %c1: index = index_constant() attrs={value: 1}
              %j: index = index_add(%i: index, %c1: index)
              %t: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%a: smem<128x64xbf16 swizzle="128B">, 0, %j: index)
              matmul(%t: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %t: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">)
            }
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse(module_str)

    module = Canonicalize().run(module)
    expected_after_first = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x64xbf16 swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %c0: index = index_constant() attrs={value: 0}
            %c1: index = index_constant() attrs={value: 1}
            %j: index = index_add(%c0: index, %c1: index)
            %t: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%a: smem<128x64xbf16 swizzle="128B">, 0, %j: index)
            matmul(%t: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %t: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">)
          }
        }""")  # noqa: E501
    assert repr(module) == expected_after_first

    module = Canonicalize().run(module)
    expected_after_second = textwrap.dedent("""\
        module {
          kernel @test(%a: smem<128x64xbf16 swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %0: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%a: smem<128x64xbf16 swizzle="128B">, 0, 1)
            matmul(%0: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %0: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %acc: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">)
          }
        }""")  # noqa: E501
    assert repr(module) == expected_after_second
