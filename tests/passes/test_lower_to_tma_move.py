import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    ModuleOp,
    TmaDescType,
    TmaMoveOp,
    gpu_dialect,
)
from ayla.passes.lower_to_tma_move import lower_to_tma_move


def test_lower_move_preserves_collective():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>) {
            %s: smem<16x16xf32> = alloc() attrs={id: 0}
            move(%s: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.cluster attrs={multicast: True, tma: True}
          }
        }"""))
    assert isinstance(module, ModuleOp)

    lower_to_tma_move(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>, %a_desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) {
            %s: smem<16x16xf32> = alloc() attrs={id: 0}
            %c0: index = index_constant() attrs={value: 0}
            tma_move(%s: smem<16x16xf32>, %a_desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={multicast: True, tma: True}
          }
        }""")


def test_lower_tma_through_singleton_batch_reshape():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<2x16x16xf32>) grid_dim=[2, 1, 2] {
            %bx: index, %by: index, %bz: index = block_ids()
            %batch: gmem<1x16x16xf32> = tile(%a: gmem<2x16x16xf32>, %bz: index, 0, 0)
            %matrix: gmem<16x16xf32> = reshape(%batch: gmem<1x16x16xf32>)
            %tile: gmem<8x16xf32 strides=[16, 1]> = tile(%matrix: gmem<16x16xf32>, %bx: index, 0)
            %smem: smem<8x16xf32> = alloc() attrs={id: 0}
            move(%smem: smem<8x16xf32>, %tile: gmem<8x16xf32 strides=[16, 1]>) attrs={tma: True}
          }
        }"""))

    lower_to_tma_move(module)

    [move] = find_ops_of_type(module, TmaMoveOp)
    assert isinstance(move.src.type, TmaDescType)
    assert move.src.type.gmem_type.sizes == [2, 16, 16]
    assert move.src.type.tile_sizes == (1, 8, 16)
    assert len(move.indices) == 3


def _multicast_load(sizes: str) -> ModuleOp:
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @test(%a: gmem<256x64xf16>) grid_dim=[2, 1, 1] cluster_dim=[2, 1, 1] {{
            %s: smem<{sizes}xf16 swizzle="128B"> = alloc() attrs={{id: 0}}
            %t: gmem<{sizes}xf16 strides=[64, 1]> = tile(%a: gmem<256x64xf16>, 0, 0)
            move(%s: smem<{sizes}xf16 swizzle="128B">, %t: gmem<{sizes}xf16 strides=[64, 1]>) collective=xyz.cluster attrs={{multicast: True, tma: True}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return module


def test_multicast_load_splits_into_one_row_slice_per_cluster_cta():
    module = lower_to_tma_move(_multicast_load("128x64"))

    text = repr(module)
    assert text.count("tma_move(") == 2
    # Two 64-row boxes, the second offset by 64 rows into tile and buffer.
    assert text.count("tile=64x64 swizzle=\"128B\">, ") == 2
    assert 'smem<64x64xf16 swizzle="128B"> = tile(%s: smem<128x64xf16 swizzle="128B">, 1, 0)' in text
    assert "index_constant() attrs={value: 64}" in text


def test_multicast_load_without_whole_swizzle_atoms_stays_whole():
    # 24 rows split into two 12-row slices would end mid swizzle atom.
    module = lower_to_tma_move(_multicast_load("24x64"))

    text = repr(module)
    assert text.count("tma_move(") == 1
    assert "tile=24x64 swizzle=\"128B\">" in text
