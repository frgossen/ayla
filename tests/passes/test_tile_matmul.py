"""Tests for the tile_matmul pass."""
import textwrap

from ayla.base.ir import find_ops
from ayla.gpu_dialect import TileOp, gpu_dialect
from ayla.passes.lower_ploop import lower_ploop
from ayla.passes.tile_matmul import tile_matmul


def test_tiles_gmem_matmul_along_m_and_k():
    # matmul shapes: lhs=[M, K], rhs=[N, K], dst=[M, N].
    # Here M=128, N=128, K=64.
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = tile_matmul(module, ["M", "K"], [32, 16], ["for", "for"], ["m", "k"])
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%m: index = 0 to 4) attrs={label: "m"} {
              for (%k: index = 0 to 4) attrs={label: "k"} {
                %0: gmem<32x64xbf16> = tile(%a: gmem<128x64xbf16>, %m: index, 0)
                %1: gmem<32x16xbf16 strides=[64, 1]> = tile(%0: gmem<32x64xbf16>, 0, %k: index)
                %2: gmem<128x16xbf16 strides=[64, 1]> = tile(%b: gmem<128x64xbf16>, 0, %k: index)
                %3: gmem<32x128xf32> = tile(%c: gmem<128x128xf32>, %m: index, 0)
                matmul(%1: gmem<32x16xbf16 strides=[64, 1]>, %2: gmem<128x16xbf16 strides=[64, 1]>, %3: gmem<32x128xf32>)
              }
            }
          }
        }""")
    assert text == expected


def test_hierarchical_output_tiling_maps_two_waves_to_16_blocks():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<4096x16xbf16>, %b: gmem<256x16xbf16>, %c: gmem<4096x256xbf16>) grid_dim=[16, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<4096x16xbf16>, %b: gmem<256x16xbf16>, %c: gmem<4096x256xbf16>) attrs={label: "mm"}
          }
        }"""))
    module = tile_matmul(
        module,
        tile_dims=["M", "M"], tile_sizes=[2048, 128],
        tile_levels=["for", "block.x"], tile_labels=["outm", None],
        label_filter="mm",
    )
    text = repr(module)
    assert 'for (%outm: index = 0 to 2) attrs={label: "outm"}' in text
    assert "ploop block.x (%0: index = 0 to 16)" in text
    assert "tile(%a: gmem<4096x16xbf16>, %outm: index, 0)" in text
    assert "tile(%1: gmem<2048x16xbf16>, %0: index, 0)" in text
    # The pass leaves the kernel alone; the grid it was given is untouched.
    assert "grid_dim=[16, 1, 1]" in text


def test_warp_group_tiling_claims_the_block_shape_once_lowered():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>)
          }
        }"""))

    module = tile_matmul(
        module,
        tile_dims=["M"], tile_sizes=[64],
        tile_levels=["wg.x"], tile_labels=[None],
    )

    # Tiling only emits the ploop -- the launch geometry is the lowering's.
    assert "ploop wg.x (%0: index = 0 to 2)" in repr(module)
    [kernel] = module.body.ops
    assert kernel.block_dim == [None, None, None]
    assert kernel.num_warp_groups is None

    [kernel] = lower_ploop(module).body.ops
    assert kernel.block_dim == [256, 1, 1]
    assert kernel.num_warp_groups == 2


def test_partial_block_tile_rounds_up_the_grid():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<192x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<192x128xf32>) {
            matmul(%a: gmem<192x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<192x128xf32>) attrs={label: "mm"}
          }
        }"""))
    module = tile_matmul(
        module, tile_dims=["M"], tile_sizes=[128],
        tile_levels=["block.x"], tile_labels=[None],
        label_filter="mm", allow_partial=True,
    )
    assert "ploop block.x (%0: index = 0 to 2)" in repr(module)
    assert "grid_dim=[2, None, None]" in repr(lower_ploop(module))
    tiles = find_ops(module, lambda op: isinstance(op, TileOp))
    assert len(tiles) == 2
    assert all(tile.maybe_partial for tile in tiles)
