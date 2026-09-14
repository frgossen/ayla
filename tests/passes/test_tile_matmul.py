"""Tests for the tile_matmul pass."""
import textwrap

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import ForOp, IfOp, TileOp, gpu_dialect
from ayla.passes.tile_matmul import tile_matmul


def test_tiles_gmem_matmul_along_m_and_k():
    # matmul shapes: lhs=[M, K], rhs=[N, K], dst=[M, N].
    # Here M=128, N=128, K=64.
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) collective=xyz.grid
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = tile_matmul(module, ["M", "K"], [32, 16], ["for", "for"], ["m", "k"])
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            for (%m: index = 0 to 4) attrs={label: "m"} {
              %0: gmem<32x128xf32> = tile(%c: gmem<128x128xf32>, %m: index, 0)
              for (%k: index = 0 to 4) attrs={label: "k"} {
                %1: gmem<32x64xbf16> = tile(%a: gmem<128x64xbf16>, %m: index, 0)
                %2: gmem<32x16xbf16 strides=[64, 1]> = tile(%1: gmem<32x64xbf16>, 0, %k: index)
                %3: gmem<128x16xbf16 strides=[64, 1]> = tile(%b: gmem<128x64xbf16>, 0, %k: index)
                matmul(%2: gmem<32x16xbf16 strides=[64, 1]>, %3: gmem<128x16xbf16 strides=[64, 1]>, %0: gmem<32x128xf32>) collective=xyz.grid
              }
            }
          }
        }""")
    assert text == expected


def test_hierarchical_output_tiling_maps_two_waves_to_16_blocks():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<4096x16xbf16>, %b: gmem<256x16xbf16>, %c: gmem<4096x256xbf16>) grid_dim=[16, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<4096x16xbf16>, %b: gmem<256x16xbf16>, %c: gmem<4096x256xbf16>) collective=xyz.grid attrs={label: "mm"}
          }
        }"""))
    module = tile_matmul(
        module,
        op_dims=["M", "M"], tile_sizes=[2048, 128],
        iteration_bindings=["for", "block.x"],
        for_loop_labels=["outm", None],
        op_label_filter="mm",
    )
    text = repr(module)
    assert "grid_dim=[16, 1, 1]" in text
    assert 'for (%outm: index = 0 to 2) attrs={label: "outm"}' in text
    assert "tile(%a: gmem<4096x16xbf16>, %outm: index, 0)" in text
    assert "tile(%0: gmem<2048x16xbf16>, %bx: index, 0)" in text


def test_warp_group_tiling_sets_block_dim():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) collective=xyz.grid
          }
        }"""))

    module = tile_matmul(
        module,
        op_dims=["M"], tile_sizes=[64],
        iteration_bindings=["wg.x"], for_loop_labels=[None],
    )

    [kernel] = module.body.ops
    assert kernel.block_dim == (256, 1, 1)
    assert kernel.num_warp_groups == 2


def test_partial_block_tile_rounds_up_the_grid():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<192x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<192x128xf32>) {
            matmul(%a: gmem<192x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<192x128xf32>) collective=xyz.grid attrs={label: "mm"}
          }
        }"""))
    tile_matmul(
        module, op_dims=["M"], tile_sizes=[128],
        iteration_bindings=["block.x"], for_loop_labels=[None],
        op_label_filter="mm", allow_partial_tiles=True,
    )
    assert "grid_dim=[2, None, None]" in repr(module)
    tiles = find_ops_of_type(module, TileOp)
    assert len(tiles) == 2
    assert all(tile.maybe_partial for tile in tiles)


def test_tiles_a_sigmoid_epilogue_with_the_matmul():
    """The epilogue is cut to the matmul's tile and left outside the K loop.

    It has M and N but no K, so it runs once per output tile rather than once
    per K step. The destination tile goes with it: an op standing after the
    loop cannot name a value defined inside it.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) block_dim=[128, 1, 1] {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) collective=xyz.grid attrs={label: "mm"}
            elementwise sigmoid(%c: gmem<128x128xf32>, %c: gmem<128x128xf32>) attrs={label: "epilogue"}
          }
        }""")  # noqa: E501

    module = tile_matmul(
        gpu_dialect.parse(module_str),
        ["M", "K"], [32, 16], ["block.x", "for"], [None, "k"],
        op_label_filter="mm", epilogue_label_filter="epilogue",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: gmem<128x128xf32>) grid_dim=[4, None, None] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %0: gmem<32x128xf32> = tile(%c: gmem<128x128xf32>, %bx: index, 0)
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %1: gmem<32x64xbf16> = tile(%a: gmem<128x64xbf16>, %bx: index, 0)
              %2: gmem<32x16xbf16 strides=[64, 1]> = tile(%1: gmem<32x64xbf16>, 0, %k: index)
              %3: gmem<128x16xbf16 strides=[64, 1]> = tile(%b: gmem<128x64xbf16>, 0, %k: index)
              matmul(%2: gmem<32x16xbf16 strides=[64, 1]>, %3: gmem<128x16xbf16 strides=[64, 1]>, %0: gmem<32x128xf32>) collective=[x.block, yz.grid] attrs={label: "mm"}
            }
            elementwise sigmoid(%0: gmem<32x128xf32>, %0: gmem<32x128xf32>) attrs={label: "epilogue"}
          }
        }""")  # noqa: E501


def test_block_x_tiling_guards_more_blocks_than_tiles():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x64xbf16>, %b: gmem<192x64xbf16>, %c: gmem<64x192xf32>) grid_dim=[4, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<64x64xbf16>, %b: gmem<192x64xbf16>, %c: gmem<64x192xf32>) collective=xyz.grid attrs={label: "mm"}
            elementwise sigmoid(%c: gmem<64x192xf32>, %c: gmem<64x192xf32>) attrs={label: "epi"}
          }
        }"""))

    tile_matmul(
        module,
        op_dims=["N"], tile_sizes=[64],
        iteration_bindings=["block.x"], for_loop_labels=[None],
        op_label_filter="mm", epilogue_label_filter="epi",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x64xbf16>, %b: gmem<192x64xbf16>, %c: gmem<64x192xf32>) grid_dim=[4, 1, 1] block_dim=[128, 1, 1] {
            %bx: index, %by: index, %bz: index = block_ids()
            %c3: index = index_constant() attrs={value: 3}
            %0: pred = index_lt(%bx: index, %c3: index)
            if(%0: pred) {
            ^bb0:
              %1: gmem<64x64xbf16> = tile(%b: gmem<192x64xbf16>, %bx: index, 0)
              %2: gmem<64x64xf32 strides=[192, 1]> = tile(%c: gmem<64x192xf32>, 0, %bx: index)
              matmul(%a: gmem<64x64xbf16>, %1: gmem<64x64xbf16>, %2: gmem<64x64xf32 strides=[192, 1]>) collective=[x.block, yz.grid] attrs={label: "mm"}
              elementwise sigmoid(%2: gmem<64x64xf32 strides=[192, 1]>, %2: gmem<64x64xf32 strides=[192, 1]>) attrs={label: "epi"}
            }
          }
        }""")  # noqa: E501


def test_block_x_tiling_rejects_a_smaller_fixed_grid():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x64xbf16>, %b: gmem<384x64xbf16>, %c: gmem<64x384xf32>) grid_dim=[4, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<64x64xbf16>, %b: gmem<384x64xbf16>, %c: gmem<64x384xf32>) collective=xyz.grid attrs={label: "mm"}
          }
        }"""))

    with pytest.raises(ValueError, match="launch dim conflict"):
        tile_matmul(
            module,
            op_dims=["N"], tile_sizes=[64],
            iteration_bindings=["block.x"], for_loop_labels=[None],
            op_label_filter="mm",
        )


def test_block_x_tiling_sets_an_unconstrained_grid():
    source = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x64xbf16>, %b: gmem<192x64xbf16>, %c: gmem<64x192xf32>) {
            matmul(%a: gmem<64x64xbf16>, %b: gmem<192x64xbf16>, %c: gmem<64x192xf32>) collective=xyz.grid
          }
        }""")
    module = gpu_dialect.parse(source)
    tile_matmul(
        module,
        op_dims=["N"], tile_sizes=[64],
        iteration_bindings=["block.x"], for_loop_labels=[None],
    )
    [kernel] = module.body.ops
    assert kernel.grid_dim == (3, None, None)
    assert not find_ops_of_type(module, ForOp)
    assert not find_ops_of_type(module, IfOp)

    unmatched = gpu_dialect.parse(source)
    before = repr(unmatched)
    tile_matmul(
        unmatched,
        op_dims=["N"], tile_sizes=[64],
        iteration_bindings=["block.x"], for_loop_labels=[None],
        op_label_filter="missing",
    )
    assert repr(unmatched) == before
