import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.pipelines.tile_reduce import TileReducePipeline


def test_reduction_parallel_and_sequential_tiling():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x1024xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x1024xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = TileReducePipeline().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x1024xf32>, %output: gmem<8x1xf32>) grid_dim=[8, None, None] {
            %bx: index, %by: index, %bz: index = block_ids()
            %c8: index = index_constant() attrs={value: 8}
            %0: index = index_mod(%bx: index, %c8: index)
            %1: gmem<1x1024xf32> = tile(%src: gmem<8x1024xf32>, %0: index, 0)
            %2: gmem<1x1xf32> = tile(%output: gmem<8x1xf32>, %0: index, 0)
            %3: wrmem<1x1xf32 layout="blocked"> = alloc() attrs={id: 0, init: 0.0, label: "mean_dst_wrmem"}
            %4: wrmem<1x128xf32 layout="blocked"> = alloc() attrs={id: 1, init: 0.0}
            for (%5: index = 0 to 8) attrs={label: "d1"} {
              %6: gmem<1x128xf32 strides=[1024, 1]> = tile(%1: gmem<1x1024xf32>, 0, %5: index)
              reduce sum(%4: wrmem<1x128xf32 layout="blocked">, %6: gmem<1x128xf32 strides=[1024, 1]>) collective=[x.block, yz.grid] attrs={label: "mean_partial", operand_tiling_scheme: [None, 1]}
            }
            reduce sum(%3: wrmem<1x1xf32 layout="blocked">, %4: wrmem<1x128xf32 layout="blocked">) collective=[x.block, yz.grid] attrs={label: "mean"}
            move(%2: gmem<1x1xf32>, %3: wrmem<1x1xf32 layout="blocked">) attrs={label: "mean_dst_wrmem"}
          }
        }""")
