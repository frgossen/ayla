"""Tests for the tile_matmul pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.tile_matmul import tile_matmul


def test_tiles_gmem_matmul_along_m_and_k():
    # matmul shapes: lhs=[M, K], rhs=[N, K], dst=[M, N].
    # Here M=128, N=128, K=64.
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem_tensor<128x64xbf16>, %b: gmem_tensor<128x64xbf16>, %c: gmem_tensor<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem_tensor<128x64xbf16>, %b: gmem_tensor<128x64xbf16>, %c: gmem_tensor<128x128xf32>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = tile_matmul(module, ["M", "K"], [32, 16], ["for", "for"], ["m", "k"])
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem_tensor<128x64xbf16>, %b: gmem_tensor<128x64xbf16>, %c: gmem_tensor<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%m: index = 0 to 4) attrs={label: "m"} {
              for (%k: index = 0 to 4) attrs={label: "k"} {
                %0: gmem_tensor<32x64xbf16> = tile(%a: gmem_tensor<128x64xbf16>, %m: index, 0)
                %1: gmem_tensor<32x16xbf16 strides=[64, 1]> = tile(%0: gmem_tensor<32x64xbf16>, 0, %k: index)
                %2: gmem_tensor<128x16xbf16 strides=[64, 1]> = tile(%b: gmem_tensor<128x64xbf16>, 0, %k: index)
                %3: gmem_tensor<32x128xf32> = tile(%c: gmem_tensor<128x128xf32>, %m: index, 0)
                matmul(%1: gmem_tensor<32x16xbf16 strides=[64, 1]>, %2: gmem_tensor<128x16xbf16 strides=[64, 1]>, %3: gmem_tensor<32x128xf32>)
              }
            }
          }
        }""")
    assert text == expected
