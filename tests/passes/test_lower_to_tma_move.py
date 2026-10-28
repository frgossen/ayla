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
    module = gpu_dialect.parse_module(textwrap.dedent("""\
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
    module = gpu_dialect.parse_module(textwrap.dedent("""\
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
