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

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.tile_matmul import tile_matmul

MODULE_STR = textwrap.dedent("""\
    module {
      kernel @mm(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) {
        matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
        elementwise(%c: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) attrs={fn: "sigmoid", label: "epilogue"}
      }
    }""")


def test_block_and_warp_group_tiling_carry_the_epilogue():
    """Two tilings in sequence, each cutting the epilogue along with the matmul.

    The second reads what the first produced: it tiles the 128-row block tile
    into two warp-group tiles, and the epilogue follows onto the same one. The
    launch geometry is derived rather than declared -- 4096/128 by 4096/256
    blocks, two warp groups of 128 threads.
    """
    module = gpu_dialect.parse(MODULE_STR)
    module = tile_matmul(
        module,
        tile_dims=["M", "N"], tile_sizes=[128, 256],
        tile_levels=["block.x", "block.y"], tile_labels=[None, None],
        label_filter="mm", epilogue_label_filter="epilogue",
    )
    module = tile_matmul(
        module,
        tile_dims=["M"], tile_sizes=[64],
        tile_levels=["wg.x"], tile_labels=[None],
        label_filter="mm", epilogue_label_filter="epilogue",
    )

    assert repr(module) == EXPECTED_IR


EXPECTED_IR = textwrap.dedent("""\
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
        matmul(%4: gmem<64x4096xbf16>, %1: gmem<256x4096xbf16>, %5: gmem<64x256xbf16 strides=[4096, 1]>) attrs={init: 0.0, label: "mm"}
        elementwise(%5: gmem<64x256xbf16 strides=[4096, 1]>, %5: gmem<64x256xbf16 strides=[4096, 1]>) attrs={fn: "sigmoid", label: "epilogue"}
      }
    }""")
