import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.lower_to_tma_move import lower_to_tma_move


def test_lower_move_preserves_collective():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>) {
            %s: smem<16x16xf32> = alloc() attrs={id: 0}
            move(%s: smem<16x16xf32>, %a: gmem<16x16xf32>) collective=xyz.cluster attrs={multicast: True, tma: True}
          }
        }"""))

    lower_to_tma_move(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<16x16xf32>, %a_desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>) {
            %s: smem<16x16xf32> = alloc() attrs={id: 0}
            %c0: index = index_constant() attrs={value: 0}
            tma_move(%s: smem<16x16xf32>, %a_desc: tma_desc<arg_idx=0 gmem<16x16xf32> tile=16x16>, %c0: index, %c0: index) collective=xyz.cluster attrs={multicast: True, tma: True}
          }
        }""")
