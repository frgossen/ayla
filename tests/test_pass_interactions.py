"""What a stack of passes produces when run together.

Each pass has unit tests over a handful of ops. This covers the part none of
them can: the composition. Tiling decides what promotion sees, promotion
decides what multi-buffering has to duplicate, and a pass can satisfy its own
tests while changing what the next one receives.

The sequence is a copy of the one auto_matmul_4096_wgmma256_wg is built from,
kept here rather than imported so the passes under test can be reordered,
dropped or given different parameters without touching a kernel that has to
keep working.

The expectation is generated, not written. To refresh it after an intended
change, print `repr(module)` in place of the assertion and run with -s. The
alloc ids come out stable because conftest resets the shared counter for each
test.
"""
# flake8: noqa: E501
import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    ElementwiseOp,
    ForOp,
    IfOp,
    MatmulOp,
    TileOp,
    gpu_dialect,
)
from ayla.passes.canonicalize import Canonicalize
from ayla.passes.promote_buffers import promote_buffers
from ayla.passes.tile_elementwise import tile_elementwise
from ayla.passes.tile_matmul import tile_matmul
from ayla.passes.unroll_loop import unroll_loop

MODULE_STR = textwrap.dedent("""\
    module {
      kernel @mm(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) {
        matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) collective=xyz.grid attrs={init: 0.0, label: "mm"}
        elementwise sigmoid(%c: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) attrs={label: "epilogue"}
      }
    }""")


def _block_tiled():
    module = gpu_dialect.parse(MODULE_STR)
    return tile_matmul(
        module,
        op_dims=["M", "N"], tile_sizes=[128, 256],
        iteration_bindings=["block.x", "block.y"],
        for_loop_labels=[None, None],
        op_label_filter="mm", epilogue_label_filter="epilogue",
    )


def _tiled():
    """The module after both matmul tilings -- the input to promotion."""
    module = _block_tiled()
    return tile_matmul(
        module,
        op_dims=["M"], tile_sizes=[64],
        iteration_bindings=["wg.x"], for_loop_labels=[None],
        op_label_filter="mm", epilogue_label_filter="epilogue",
    )


def test_block_and_warp_group_tiling_carry_the_epilogue():
    """Two tilings in sequence, each cutting the epilogue along with the matmul.

    The second reads what the first produced: it tiles the 128-row block tile
    into two warp-group tiles, and the epilogue follows onto the same one. The
    launch geometry is derived rather than declared -- 4096/128 by 4096/256
    blocks, two warp groups of 128 threads.
    """
    assert repr(_tiled()) == TILED_IR


def test_promotion_moves_the_epilogue_onto_the_accumulator():
    """Promotion reads what tiling produced, and takes the epilogue with it.

    Tiling leaves the epilogue on the destination tile in gmem. Promoting that
    destination into registers moves the epilogue onto the buffer, ahead of the
    convert and store -- so it happens on the way out rather than as a second
    pass over what was just written.
    """
    module = _tiled()
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32",
        label_filter="mm", epilogue_label_filter="epilogue",
    )

    assert repr(module) == PROMOTED_IR


def test_elementwise_tail_is_only_materialized_by_explicit_loop_passes():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @epilogue(%value: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) {
            elementwise sigmoid(%value: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, %value: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "epilogue"}
          }
        }"""))  # noqa: E501
    tile_elementwise(
        module,
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
        label_filter="epilogue",
    )

    [loop] = find_ops_of_type(module, ForOp)
    assert (loop.lb, loop.ub, loop.step) == (0, 4, 1)
    [dynamic_tile] = find_ops_of_type(module, TileOp)
    assert dynamic_tile.static_indices == [0, None]

    unroll_loop(module, unroll_factor=4, label_filter="outn")
    Canonicalize().run(module)
    Canonicalize().run(module)

    assert not find_ops_of_type(module, ForOp)
    tiles = find_ops_of_type(module, TileOp)
    assert [tile.static_indices for tile in tiles] == [
        [0, 0], [0, 1], [0, 2], [0, 3],
    ]
    elementwise = find_ops_of_type(module, ElementwiseOp)
    assert [op.attrs["label"] for op in elementwise] == [
        "epilogue_iter0",
        "epilogue_iter1",
        "epilogue_iter2",
        "epilogue_iter3",
    ]


def test_output_tiling_remains_in_explicit_loops_after_promotion():
    """Promotion establishes memory scope before output tiling adds loops."""
    module = _block_tiled()
    promote_buffers(
        module, target_type="smem", operands=["dst"], tma=True,
        label_filter="mm", epilogue_label_filter="epilogue",
    )
    tile_matmul(
        module,
        op_dims=["M"], tile_sizes=[64],
        iteration_bindings=["wg.x"], for_loop_labels=[None],
        op_label_filter="mm", epilogue_label_filter="epilogue",
    )
    promote_buffers(
        module, target_type="wrmem", target_dtype="f32", operands=["dst"],
        layout="wgmma_layout_0", warps=4,
        label_filter="mm", epilogue_label_filter="epilogue",
    )
    tile_elementwise(
        module,
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
        label_filter=(
            "epilogue|mm_dst_convert|mm_dst_wrmem|mm_dst_smem"
        ),
    )

    loops = find_ops_of_type(module, ForOp)
    assert len(loops) == 3
    assert all((loop.lb, loop.ub, loop.step) == (0, 4, 1) for loop in loops)
    assert all(loop.attrs["label"] == "outn" for loop in loops)
    for loop in loops:
        tiles = find_ops_of_type(loop, TileOp)
        assert tiles
        assert all(tile.indices[1] is loop.iv for tile in tiles)
        assert all(tile.result().type.sizes[1] == 64 for tile in tiles)


def test_two_matmuls_share_block_x_with_different_n_tile_counts():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @chain(%a: gmem<64x64xbf16>, %w_up: gmem<192x64xbf16>, %intermediate: gmem<64x192xbf16>, %w_down: gmem<256x192xbf16>, %out: gmem<64x256xbf16>) {
            matmul(%a: gmem<64x64xbf16>, %w_up: gmem<192x64xbf16>, %intermediate: gmem<64x192xbf16>) collective=xyz.grid attrs={label: "up"}
            matmul(%intermediate: gmem<64x192xbf16>, %w_down: gmem<256x192xbf16>, %out: gmem<64x256xbf16>) collective=xyz.grid attrs={label: "down"}
          }
        }"""))  # noqa: E501

    tile_matmul(
        module,
        op_dims=["N"], tile_sizes=[64],
        iteration_bindings=["block.x"], for_loop_labels=[None],
    )

    [kernel] = module.body.ops
    assert kernel.grid_dim == (4, None, None)
    [guard] = find_ops_of_type(module, IfOp)
    matmuls = find_ops_of_type(module, MatmulOp)
    assert matmuls[0].parent_block is guard.body
    assert matmuls[1].parent_block is kernel.body


TILED_IR = textwrap.dedent("""\
    module {
      kernel @mm(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, None] block_dim=[256, 1, 1] num_warp_groups=2 {
        %bx: index, %by: index, %bz: index = block_ids()
        %0: gmem<128x4096xbf16> = tile(%a: gmem<4096x4096xbf16>, %bx: index, 0)
        %1: gmem<256x4096xbf16> = tile(%b: gmem<4096x4096xbf16>, %by: index, 0)
        %2: gmem<128x4096xbf16> = tile(%c: gmem<4096x4096xbf16>, %bx: index, 0)
        %3: gmem<128x256xbf16 strides=[4096, 1]> = tile(%2: gmem<128x4096xbf16>, 0, %by: index)
        %wgx: index, %wgy: index, %wgz: index = warp_group_id()
        %4: gmem<64x4096xbf16> = tile(%0: gmem<128x4096xbf16>, %wgx: index, 0)
        %5: gmem<64x256xbf16 strides=[4096, 1]> = tile(%3: gmem<128x256xbf16 strides=[4096, 1]>, %wgx: index, 0)
        matmul(%4: gmem<64x4096xbf16>, %1: gmem<256x4096xbf16>, %5: gmem<64x256xbf16 strides=[4096, 1]>) collective=[x.wg, y.block, z.grid] attrs={init: 0.0, label: "mm"}
        elementwise sigmoid(%5: gmem<64x256xbf16 strides=[4096, 1]>, %5: gmem<64x256xbf16 strides=[4096, 1]>) attrs={label: "epilogue"}
      }
    }""")

PROMOTED_IR = textwrap.dedent("""\
    module {
      kernel @mm(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, None] block_dim=[256, 1, 1] num_warp_groups=2 {
        %bx: index, %by: index, %bz: index = block_ids()
        %0: gmem<128x4096xbf16> = tile(%a: gmem<4096x4096xbf16>, %bx: index, 0)
        %1: gmem<256x4096xbf16> = tile(%b: gmem<4096x4096xbf16>, %by: index, 0)
        %2: gmem<128x4096xbf16> = tile(%c: gmem<4096x4096xbf16>, %bx: index, 0)
        %3: gmem<128x256xbf16 strides=[4096, 1]> = tile(%2: gmem<128x4096xbf16>, 0, %by: index)
        %wgx: index, %wgy: index, %wgz: index = warp_group_id()
        %4: gmem<64x4096xbf16> = tile(%0: gmem<128x4096xbf16>, %wgx: index, 0)
        %5: gmem<64x256xbf16 strides=[4096, 1]> = tile(%3: gmem<128x256xbf16 strides=[4096, 1]>, %wgx: index, 0)
        %6: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0, init: 0.0, label: "mm_dst_wrmem"}
        matmul(%4: gmem<64x4096xbf16>, %1: gmem<256x4096xbf16>, %6: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, y.block, z.grid] attrs={label: "mm"}
        %7: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
        elementwise sigmoid(%7: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %6: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "epilogue"}
        move(%5: gmem<64x256xbf16 strides=[4096, 1]>, %7: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_wrmem"}
      }
    }""")
