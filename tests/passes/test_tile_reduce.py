import textwrap

from ayla.gpu_dialect import (
    CollectiveAxis,
    CollectiveLevel,
    gpu_dialect,
)
from ayla.passes.canonicalize import Canonicalize
from ayla.passes.tile_reduce import tile_reduce


def test_tile_reduce_maps_one_dimension_to_block_y():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) {
            reduce sum(%output: gmem<256x128x1xf32>, %src: gmem<256x128x6656xf32>) attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<256x128x1xf32>, %output: gmem<256x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 128},
        parallel_iteration_spaces={
            0: (CollectiveLevel.BLOCK, CollectiveAxis.Y),
        },
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) grid_dim=[None, 2, None] {
            %bx: index, %by: index, %bz: index = block_ids()
            %c2: index = index_constant() attrs={value: 2}
            %0: index = index_mod(%by: index, %c2: index)
            %1: gmem<128x128x6656xf32> = tile(%src: gmem<256x128x6656xf32>, %0: index, 0, 0)
            %2: gmem<128x128x1xf32> = tile(%output: gmem<256x128x1xf32>, %0: index, 0, 0)
            reduce sum(%2: gmem<128x128x1xf32>, %1: gmem<128x128x6656xf32>) attrs={init: 0.0, label: "mean"}
            elementwise div(%2: gmem<128x128x1xf32>, %2: gmem<128x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }""")


def test_tile_reduce_combines_parallel_and_sequential_dimensions():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) {
            reduce sum(%output: gmem<256x128x1xf32>, %src: gmem<256x128x6656xf32>) attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<256x128x1xf32>, %output: gmem<256x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 128, 1: 64},
        parallel_iteration_spaces={
            0: (CollectiveLevel.BLOCK, CollectiveAxis.Y),
        },
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) grid_dim=[None, 2, None] {
            %bx: index, %by: index, %bz: index = block_ids()
            %c2: index = index_constant() attrs={value: 2}
            %0: index = index_mod(%by: index, %c2: index)
            for (%1: index = 0 to 2) attrs={label: "d1"} {
              %2: gmem<128x64x6656xf32 strides=[851968, 6656, 1]> = tile(%src: gmem<256x128x6656xf32>, %0: index, %1: index, 0)
              %3: gmem<128x64x1xf32 strides=[128, 1, 1]> = tile(%output: gmem<256x128x1xf32>, %0: index, %1: index, 0)
              reduce sum(%3: gmem<128x64x1xf32 strides=[128, 1, 1]>, %2: gmem<128x64x6656xf32 strides=[851968, 6656, 1]>) attrs={init: 0.0, label: "mean"}
              elementwise div(%3: gmem<128x64x1xf32 strides=[128, 1, 1]>, %3: gmem<128x64x1xf32 strides=[128, 1, 1]>) attrs={divisor: 6656, label: "mean_epilogue"}
            }
          }
        }""")


def test_tile_reduce_maps_two_dimensions_to_one_parallel_iteration_space():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) {
            reduce sum(%output: gmem<256x128x1xf32>, %src: gmem<256x128x6656xf32>) attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<256x128x1xf32>, %output: gmem<256x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 128, 1: 64},
        parallel_iteration_spaces={
            0: (CollectiveLevel.BLOCK, CollectiveAxis.X),
            1: (CollectiveLevel.BLOCK, CollectiveAxis.X),
        },
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) grid_dim=[4, None, None] {
            %bx: index, %by: index, %bz: index = block_ids()
            %c2: index = index_constant() attrs={value: 2}
            %0: index = index_div(%bx: index, %c2: index)
            %1: index = index_mod(%0: index, %c2: index)
            %2: index = index_mod(%bx: index, %c2: index)
            %3: gmem<128x64x6656xf32 strides=[851968, 6656, 1]> = tile(%src: gmem<256x128x6656xf32>, %1: index, %2: index, 0)
            %4: gmem<128x64x1xf32 strides=[128, 1, 1]> = tile(%output: gmem<256x128x1xf32>, %1: index, %2: index, 0)
            reduce sum(%4: gmem<128x64x1xf32 strides=[128, 1, 1]>, %3: gmem<128x64x6656xf32 strides=[851968, 6656, 1]>) attrs={init: 0.0, label: "mean"}
            elementwise div(%4: gmem<128x64x1xf32 strides=[128, 1, 1]>, %4: gmem<128x64x1xf32 strides=[128, 1, 1]>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }""")
