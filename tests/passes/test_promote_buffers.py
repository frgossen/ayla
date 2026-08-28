"""Tests for the promote_buffers pass."""
import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.promote_buffers import promote_buffers


def test_wrmem_tma_writeback_stages_block_tile_in_smem():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %c: gmem<128x256xbf16>) grid_dim=[1, 1, 1] block_dim=[256, 1, 1] {
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %c_wg: gmem<64x256xbf16 strides=[256, 1]> = tile(%c: gmem<128x256xbf16>, %wgx: index, 0)
            matmul(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %c_wg: gmem<64x256xbf16 strides=[256, 1]>) attrs={init: 0.0, label: "mm"}
          }
        }"""))
    module = promote_buffers(
        module, target_type="wrmem", target_dtype="f32", operands=["dst"],
        layout="wgmma_layout_0", warps=4, tma=True, label_filter="mm",
    )
    text = repr(module)
    assert "smem<128x256xbf16> = alloc()" in text
    # SMEM is tiled by warp-group id, so the gather move matches the fragment
    # shape (64x256 <- 64x256) instead of a size-mismatched 128x256 <- 64x256.
    assert "smem<64x256xbf16> = tile(" in text
    assert "smem<128x256xbf16>, %wgx: index, 0)" in text
    assert "move(%3: smem<64x256xbf16>, %1: wrmem<64x256xbf16" in text
    assert "move(%c: gmem<128x256xbf16>" in text
    assert 'attrs={label: "mm_dst_tma", tma: True}' in text


def test_wrmem_tma_writeback_can_use_one_arena_per_slice():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %c: gmem<64x256xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %c: gmem<64x256xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }"""))
    module = promote_buffers(
        module, target_type="wrmem", target_dtype="f32", operands=["dst"],
        layout="wgmma_layout_0", warps=4, tma=True,
        writeback_slices=2, writeback_buffers=2, label_filter="mm",
    )
    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mm(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %c: gmem<64x256xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %0: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0, init: 0.0, label: "mm_dst_wrmem"}
            matmul(%a: smem<64x16xbf16>, %b: smem<256x16xbf16>, %0: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm"}
            %1: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            convert(%1: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %0: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">)
            %2: smem<64x128xbf16> = alloc() attrs={id: 2}
            %3: smem<64x128xbf16> = alloc() attrs={id: 3}
            move(%2: smem<64x128xbf16>, %1: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_smem", wgmma_col_start: 0}
            %4: gmem<64x128xbf16 strides=[256, 1]> = tile(%c: gmem<64x256xbf16>, 0, 0)
            move(%4: gmem<64x128xbf16 strides=[256, 1]>, %2: smem<64x128xbf16>) attrs={label: "mm_dst_tma", tma: True}
            move(%3: smem<64x128xbf16>, %1: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_smem", wgmma_col_start: 128}
            %5: gmem<64x128xbf16 strides=[256, 1]> = tile(%c: gmem<64x256xbf16>, 0, 1)
            move(%5: gmem<64x128xbf16 strides=[256, 1]>, %3: smem<64x128xbf16>) attrs={label: "mm_dst_tma", tma: True}
          }
        }""")


def test_promote_gmem_matmul_operands_to_smem():
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: rmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: rmem<128x128xf32>) attrs={init: 0.0}
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = promote_buffers(module, target_type="smem", operands=["lhs", "rhs", "dst"], swizzle="128B", tma=True)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: rmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %0: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            move(%0: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            %1: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            move(%1: smem<128x64xbf16 swizzle="128B">, %b: gmem<128x64xbf16>) attrs={tma: True}
            %2: smem<128x128xf32 swizzle="128B"> = alloc() attrs={id: 2, init: 0.0}
            matmul(%0: smem<128x64xbf16 swizzle="128B">, %1: smem<128x64xbf16 swizzle="128B">, %2: smem<128x128xf32 swizzle="128B">)
            move(%c: rmem<128x128xf32>, %2: smem<128x128xf32 swizzle="128B">) attrs={tma: True}
          }
        }""")
    assert text == expected


def test_promote_can_slice_tma_operand_loads():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<384x64xbf16>, %c: rmem<128x384xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<384x64xbf16>, %c: rmem<128x384xf32>) attrs={label: "mm"}
          }
        }"""))
    promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", tma=True, tma_load_slices={"rhs": 3},
        move_attrs={"rhs": {"l2_cache_hint": "evict_last"}},
        label_filter="mm",
    )
    text = repr(module)
    # One label belongs to the allocation; the other three are sliced moves.
    assert text.count('label: "mm_rhs_smem"') == 4
    assert text.count('l2_cache_hint: "evict_last"') == 3
    assert text.count("gmem<128x64xbf16> = tile(%b") == 3
    assert text.count("smem<128x64xbf16") >= 3


def test_promote_lhs_rhs_preserves_init_on_matmul():
    """When dst is NOT promoted, init stays on the matmul."""
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: rmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: rmem<128x128xf32>) attrs={init: 0.0}
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = promote_buffers(module, target_type="smem", operands=["lhs", "rhs"], swizzle="128B", tma=True)
    text = repr(module)

    expected = textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>, %c: rmem<128x128xf32>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %0: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            move(%0: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>) attrs={tma: True}
            %1: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 1}
            move(%1: smem<128x64xbf16 swizzle="128B">, %b: gmem<128x64xbf16>) attrs={tma: True}
            matmul(%0: smem<128x64xbf16 swizzle="128B">, %1: smem<128x64xbf16 swizzle="128B">, %c: rmem<128x128xf32>) attrs={init: 0.0}
          }
        }""")
    assert text == expected


def test_promote_smem_with_warp_group_dependent_lhs():
    """Promote lhs/rhs to smem when the lhs tile depends on warp_group_id.

    The lhs gmem tile (%6) is wg.x-dependent (via %3 = tile(%0, %wgx, 0)),
    so each warp group loads different rows. The promote pass must allocate
    smem large enough for all warp groups (128x64, not 64x64) and tile it
    by %wgx so each warp group reads its own portion.

    The rhs gmem tile (%7) is wg.x-independent, so the 256x64 smem is
    shared across warp groups — no issue there.
    """
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[256, 1, 1] num_warp_groups=2 {
            %bx: index, %by: index, %bz: index = block_ids()
            %0: gmem<128x4096xbf16> = tile(%a: gmem<4096x4096xbf16>, %bx: index, 0)
            %1: gmem<256x4096xbf16> = tile(%b: gmem<4096x4096xbf16>, %by: index, 0)
            %2: gmem<128x256xbf16 strides=[4096, 1]> = tile(%c: gmem<4096x4096xbf16>, %bx: index, %by: index)
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %3: gmem<64x4096xbf16> = tile(%0: gmem<128x4096xbf16>, %wgx: index, 0)
            %4: gmem<64x256xbf16 strides=[4096, 1]> = tile(%2: gmem<128x256xbf16 strides=[4096, 1]>, %wgx: index, 0)
            %5: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0, init: 0.0}
            for (%k: index = 0 to 64) attrs={label: "k"} {
              %6: gmem<64x64xbf16 strides=[4096, 1]> = tile(%3: gmem<64x4096xbf16>, 0, %k: index)
              %7: gmem<256x64xbf16 strides=[4096, 1]> = tile(%1: gmem<256x4096xbf16>, 0, %k: index)
              matmul(%6: gmem<64x64xbf16 strides=[4096, 1]>, %7: gmem<256x64xbf16 strides=[4096, 1]>, %5: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm"}
            }
            %8: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            convert(%8: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %5: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">)
            move(%4: gmem<64x256xbf16 strides=[4096, 1]>, %8: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_wrmem"}
          }
        }""")

    module = gpu_dialect.parse(module_str)
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    text = repr(module)

    # The pass must detect that lhs (%6) depends on %wgx (via %3) and:
    #  1. Walk up to find the block-collective supertile (%0, 128x4096)
    #  2. Allocate smem for the full block tile (128x64, not 64x64)
    #  3. Load the block-collective gmem tile into smem via TMA
    #  4. Tile the smem by %wgx to produce the per-wg view (64x64)
    #  5. Use the per-wg smem view in the matmul
    #
    # The rhs (%7) does not depend on %wgx, so it's promoted directly
    # as 256x64 smem — no supertile needed.
    expected = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[256, 1, 1] num_warp_groups=2 {
            %bx: index, %by: index, %bz: index = block_ids()
            %0: gmem<128x4096xbf16> = tile(%a: gmem<4096x4096xbf16>, %bx: index, 0)
            %1: gmem<256x4096xbf16> = tile(%b: gmem<4096x4096xbf16>, %by: index, 0)
            %2: gmem<128x256xbf16 strides=[4096, 1]> = tile(%c: gmem<4096x4096xbf16>, %bx: index, %by: index)
            %wgx: index, %wgy: index, %wgz: index = warp_group_id()
            %3: gmem<64x4096xbf16> = tile(%0: gmem<128x4096xbf16>, %wgx: index, 0)
            %4: gmem<64x256xbf16 strides=[4096, 1]> = tile(%2: gmem<128x256xbf16 strides=[4096, 1]>, %wgx: index, 0)
            %5: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 0, init: 0.0}
            for (%k: index = 0 to 64) attrs={label: "k"} {
              %6: gmem<64x64xbf16 strides=[4096, 1]> = tile(%3: gmem<64x4096xbf16>, 0, %k: index)
              %7: gmem<256x64xbf16 strides=[4096, 1]> = tile(%1: gmem<256x4096xbf16>, 0, %k: index)
              %8: gmem<128x64xbf16 strides=[4096, 1]> = tile(%0: gmem<128x4096xbf16>, 0, %k: index)
              %9: smem<128x64xbf16 swizzle="128B"> = alloc() attrs={id: 0, label: "mm_lhs_smem"}
              move(%9: smem<128x64xbf16 swizzle="128B">, %8: gmem<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem", nc: True, tma: True}
              %10: smem<64x64xbf16 swizzle="128B"> = tile(%9: smem<128x64xbf16 swizzle="128B">, %wgx: index, 0)
              %11: smem<256x64xbf16 swizzle="128B"> = alloc() attrs={id: 1, label: "mm_rhs_smem"}
              move(%11: smem<256x64xbf16 swizzle="128B">, %7: gmem<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem", nc: True, tma: True}
              matmul(%10: smem<64x64xbf16 swizzle="128B">, %11: smem<256x64xbf16 swizzle="128B">, %5: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm"}
            }
            %12: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 1}
            convert(%12: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %5: wrmem<64x256xf32 warps=4 layout="wgmma_layout_0">)
            move(%4: gmem<64x256xbf16 strides=[4096, 1]>, %12: wrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_wrmem"}
          }
        }""")
    assert text == expected


def test_wg_ploop_iv_is_not_block_uniform():
    """A ploop's induction variable varies at its own level.

    The lhs here is tiled by a warp-group ploop index, so the load has to stay
    block-wide and the tile be replayed on the promoted buffer. Reading the iv
    as block-uniform would stage only one warp group's 64 rows.
    """
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<128x64xbf16>, %b: gmem<256x64xbf16>, %c: gmem<128x256xf32>) {
            ploop wg.x (%w: index = 0 to 2) {
              %a_wg: gmem<64x64xbf16 strides=[64, 1]> = tile(%a: gmem<128x64xbf16>, %w: index, 0)
              %c_wg: gmem<64x256xf32 strides=[256, 1]> = tile(%c: gmem<128x256xf32>, %w: index, 0)
              matmul(%a_wg: gmem<64x64xbf16 strides=[64, 1]>, %b: gmem<256x64xbf16>, %c_wg: gmem<64x256xf32 strides=[256, 1]>) attrs={label: "mm"}
            }
          }
        }"""))  # noqa: E501
    module = promote_buffers(
        module, target_type="smem", operands=["lhs"], swizzle="128B",
        tma=True, nc=True, label_filter="mm",
    )
    text = repr(module)

    # All 128 rows are staged block-collectively ...
    assert 'smem<128x64xbf16 swizzle="128B"> = alloc()' in text
    assert 'move(%0: smem<128x64xbf16 swizzle="128B">, %a: gmem<128x64xbf16>)' in text
    # ... and the warp-group tile is replayed on the promoted buffer.
    assert (
        'smem<64x64xbf16 swizzle="128B"> = tile('
        '%0: smem<128x64xbf16 swizzle="128B">, %w: index, 0)'
    ) in text
