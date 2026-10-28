import textwrap

import pytest

from ayla.gpu_dialect import CollectiveAxis, CollectiveLevel, gpu_dialect
from ayla.pipelines.canonicalize import Canonicalize
from ayla.passes.tile_reduce import tile_reduce


def test_tile_reduce_without_hero_filter_skips_non_reductions():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            elementwise sigmoid(%output: gmem<8x1xf32>, %output: gmem<8x1xf32>) attrs={label: "before"}
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 4},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
    )

    text = repr(module)
    assert text.count('label: "before"') == 1
    assert text.count("reduce sum(") == 1
    assert "gmem<4x64xf32>" in text


def test_tile_reduce_tiles_reduction_dimension():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            %0: gmem<8x2xf32> = alloc() attrs={id: 0}
            %1: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%0: gmem<8x2xf32>, %1: rmem<f32>)
            for (%2: index = 0 to 2) attrs={label: "d1"} {
              %3: gmem<8x1xf32 strides=[2, 1]> = tile(%0: gmem<8x2xf32>, 0, %2: index)
              %4: gmem<8x32xf32 strides=[64, 1]> = tile(%src: gmem<8x64xf32>, 0, %2: index)
              reduce sum(%3: gmem<8x1xf32 strides=[2, 1]>, %4: gmem<8x32xf32 strides=[64, 1]>) collective=xyz.grid attrs={label: "mean_partial"}
            }
            reduce sum(%output: gmem<8x1xf32>, %0: gmem<8x2xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_reduction_dimension_accumulates_into_output():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            %0: gmem<8x2xf32> = alloc() attrs={id: 0}
            %1: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%0: gmem<8x2xf32>, %1: rmem<f32>)
            for (%2: index = 0 to 2) attrs={label: "d1"} {
              %3: gmem<8x1xf32 strides=[2, 1]> = tile(%0: gmem<8x2xf32>, 0, %2: index)
              %4: gmem<8x32xf32 strides=[64, 1]> = tile(%src: gmem<8x64xf32>, 0, %2: index)
              reduce sum(%3: gmem<8x1xf32 strides=[2, 1]>, %4: gmem<8x32xf32 strides=[64, 1]>) collective=xyz.grid attrs={label: "mean_partial"}
            }
            reduce sum(%output: gmem<8x1xf32>, %0: gmem<8x2xf32>) collective=xyz.grid attrs={label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_reduction_tiling_only_retargets_final_reduction():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )
    module = tile_reduce(
        module,
        tile_sizes={1: 1},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )

    text = repr(module)
    assert text.count('label: "mean_partial"') == 2
    assert text.count('label: "mean"') == 1


def test_tile_reduce_combines_parallel_and_reduction_dimensions():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 4},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )
    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) grid_dim=[2, None, None] {
            %bx: index = block_id.x
            %c1: index = index_constant() attrs={value: 1}
            %0: index = index_div(%bx: index, %c1: index)
            %c2: index = index_constant() attrs={value: 2}
            %1: index = index_mod(%0: index, %c2: index)
            %2: gmem<4x64xf32> = tile(%src: gmem<8x64xf32>, %1: index, 0)
            %3: gmem<4x1xf32> = tile(%output: gmem<8x1xf32>, %1: index, 0)
            %4: gmem<8x2xf32> = alloc() attrs={id: 0}
            %5: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%4: gmem<8x2xf32>, %5: rmem<f32>)
            %6: gmem<4x2xf32> = tile(%4: gmem<8x2xf32>, %1: index, 0)
            for (%7: index = 0 to 2) attrs={label: "d1"} {
              %8: gmem<4x1xf32 strides=[2, 1]> = tile(%6: gmem<4x2xf32>, 0, %7: index)
              %9: gmem<4x32xf32 strides=[64, 1]> = tile(%2: gmem<4x64xf32>, 0, %7: index)
              reduce sum(%8: gmem<4x1xf32 strides=[2, 1]>, %9: gmem<4x32xf32 strides=[64, 1]>) collective=[x.block, yz.grid] attrs={label: "mean_partial"}
            }
            reduce sum(%3: gmem<4x1xf32>, %6: gmem<4x2xf32>) collective=[x.block, yz.grid] attrs={init: 0.0, label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_vector_accumulation_into_block_tiled_wrmem():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x1024xf32>) grid_dim=[8, None, None] {
            %bx: index = block_id.x
            %c8: index = index_constant() attrs={value: 8}
            %0: index = index_mod(%bx: index, %c8: index)
            %1: gmem<1x1024xf32> = tile(%src: gmem<8x1024xf32>, %0: index, 0)
            %2: wrmem<1x1xf32 layout="blocked"> = alloc() attrs={id: 1}
            %2_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%2: wrmem<1x1xf32 layout="blocked">, %2_init: rmem<f32>)
            reduce sum(%2: wrmem<1x1xf32 layout="blocked">, %1: gmem<1x1024xf32>) collective=[x.block, yz.grid] attrs={label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={1: 128},
        parallel_iteration_space=None,
        hero_label_filter="mean",
        vector_accumulation=True,
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x1024xf32>) grid_dim=[8, None, None] {
            %bx: index = block_id.x
            %c8: index = index_constant() attrs={value: 8}
            %0: index = index_mod(%bx: index, %c8: index)
            %1: gmem<1x1024xf32> = tile(%src: gmem<8x1024xf32>, %0: index, 0)
            %2: wrmem<1x1xf32 layout="blocked"> = alloc() attrs={id: 1}
            %2_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%2: wrmem<1x1xf32 layout="blocked">, %2_init: rmem<f32>)
            %3: wrmem<1x128xf32 layout="blocked"> = alloc() attrs={id: 0}
            %4: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%3: wrmem<1x128xf32 layout="blocked">, %4: rmem<f32>)
            for (%5: index = 0 to 8) attrs={label: "d1"} {
              %6: gmem<1x128xf32 strides=[1024, 1]> = tile(%1: gmem<1x1024xf32>, 0, %5: index)
              reduce sum(%3: wrmem<1x128xf32 layout="blocked">, %6: gmem<1x128xf32 strides=[1024, 1]>) collective=[x.block, yz.grid] attrs={label: "mean_partial"}
            }
            reduce sum(%2: wrmem<1x1xf32 layout="blocked">, %3: wrmem<1x128xf32 layout="blocked">) collective=[x.block, yz.grid] attrs={label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_across_block_dimensions():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x1024xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x1024xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 1},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )
    module = Canonicalize().run(module)

    module = tile_reduce(
        module,
        tile_sizes={1: 128},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.Y,
        ),
        hero_label_filter="mean",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x1024xf32>, %output: gmem<8x1xf32>) grid_dim=[8, 8, None] {
            %bx: index = block_id.x
            %c8: index = index_constant() attrs={value: 8}
            %0: index = index_mod(%bx: index, %c8: index)
            %1: gmem<1x1024xf32> = tile(%src: gmem<8x1024xf32>, %0: index, 0)
            %2: gmem<1x1xf32> = tile(%output: gmem<8x1xf32>, %0: index, 0)
            %3: gmem<8x8xf32> = alloc() attrs={id: 0}
            %4: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%3: gmem<8x8xf32>, %4: rmem<f32>)
            %5: gmem<1x8xf32> = tile(%3: gmem<8x8xf32>, %0: index, 0)
            %by: index = block_id.y
            %6: index = index_mod(%by: index, %c8: index)
            %7: gmem<1x1xf32 strides=[8, 1]> = tile(%5: gmem<1x8xf32>, 0, %6: index)
            %8: gmem<1x128xf32 strides=[1024, 1]> = tile(%1: gmem<1x1024xf32>, 0, %6: index)
            reduce sum(%7: gmem<1x1xf32 strides=[8, 1]>, %8: gmem<1x128xf32 strides=[1024, 1]>) collective=[xy.block, z.grid] attrs={label: "mean_partial"}
            reduce sum(%2: gmem<1x1xf32>, %5: gmem<1x8xf32>) collective=[x.block, yz.grid] attrs={init: 0.0, label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_rejects_parallel_and_reduction_dimensions_on_block_x():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="parallel and reduction dimensions in one invocation",
    ):
        tile_reduce(
            module,
            tile_sizes={0: 4, 1: 32},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_parallel_vector_accumulation():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="does not implement parallel vector accumulation",
    ):
        tile_reduce(
            module,
            tile_sizes={1: 32},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
            vector_accumulation=True,
        )


def test_tile_reduce_maps_parallel_dimension_to_block_and_reduction_to_warp():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 4},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )
    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=(
            CollectiveLevel.WARP, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) grid_dim=[2, None, None] block_dim=[64, None, None] {
            %bx: index = block_id.x
            %c1: index = index_constant() attrs={value: 1}
            %0: index = index_div(%bx: index, %c1: index)
            %c2: index = index_constant() attrs={value: 2}
            %1: index = index_mod(%0: index, %c2: index)
            %2: gmem<4x64xf32> = tile(%src: gmem<8x64xf32>, %1: index, 0)
            %3: gmem<4x1xf32> = tile(%output: gmem<8x1xf32>, %1: index, 0)
            %4: gmem<8x2xf32> = alloc() attrs={id: 0}
            %5: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%4: gmem<8x2xf32>, %5: rmem<f32>)
            %6: gmem<4x2xf32> = tile(%4: gmem<8x2xf32>, %1: index, 0)
            %tx: index = thread_id.x
            %c32: index = index_constant() attrs={value: 32}
            %7: index = index_div(%tx: index, %c32: index)
            %c1_1: index = index_constant() attrs={value: 1}
            %8: index = index_div(%7: index, %c1_1: index)
            %c2_1: index = index_constant() attrs={value: 2}
            %9: index = index_mod(%8: index, %c2_1: index)
            %10: gmem<4x1xf32 strides=[2, 1]> = tile(%6: gmem<4x2xf32>, 0, %9: index)
            %11: gmem<4x32xf32 strides=[64, 1]> = tile(%2: gmem<4x64xf32>, 0, %9: index)
            reduce sum(%10: gmem<4x1xf32 strides=[2, 1]>, %11: gmem<4x32xf32 strides=[64, 1]>) collective=[x.warp, yz.grid] attrs={label: "mean_partial"}
            reduce sum(%3: gmem<4x1xf32>, %6: gmem<4x2xf32>) collective=[x.block, yz.grid] attrs={init: 0.0, label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_composes_block_and_warp_tiling():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<64x64xf32>, %output: gmem<64x1xf32>) {
            reduce sum(%output: gmem<64x1xf32>, %src: gmem<64x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 32},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )
    module = tile_reduce(
        module,
        tile_sizes={0: 16},
        parallel_iteration_space=(
            CollectiveLevel.WARP, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )
    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<64x64xf32>, %output: gmem<64x1xf32>) grid_dim=[2, None, None] block_dim=[64, None, None] {
            %bx: index = block_id.x
            %0: gmem<32x64xf32> = tile(%src: gmem<64x64xf32>, %bx: index, 0)
            %1: gmem<32x1xf32> = tile(%output: gmem<64x1xf32>, %bx: index, 0)
            %tx: index = thread_id.x
            %c32: index = index_constant() attrs={value: 32}
            %3: index = index_div(%tx: index, %c32: index)
            %4: index = index_mod(%3: index, %c2: index)
            %5: gmem<16x64xf32> = tile(%1: gmem<32x64xf32>, %4: index, 0)
            %6: gmem<16x1xf32> = tile(%2: gmem<32x1xf32>, %4: index, 0)
            %7: gmem<64x2xf32> = alloc() attrs={id: 0}
            %8: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%7: gmem<64x2xf32>, %8: rmem<f32>)
            %9: gmem<32x2xf32> = tile(%7: gmem<64x2xf32>, %0: index, 0)
            %10: gmem<16x2xf32> = tile(%9: gmem<32x2xf32>, %4: index, 0)
            for (%11: index = 0 to 2) attrs={label: "d1"} {
              %12: gmem<16x1xf32 strides=[2, 1]> = tile(%10: gmem<16x2xf32>, 0, %11: index)
              %13: gmem<16x32xf32 strides=[64, 1]> = tile(%5: gmem<16x64xf32>, 0, %11: index)
              reduce sum(%12: gmem<16x1xf32 strides=[2, 1]>, %13: gmem<16x32xf32 strides=[64, 1]>) collective=[x.warp, yz.grid] attrs={label: "mean_partial"}
            }
            reduce sum(%6: gmem<16x1xf32>, %10: gmem<16x2xf32>) collective=[x.warp, yz.grid] attrs={init: 0.0, label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_rejects_reusing_iteration_space_across_invocations():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x8x64xf32>, %output: gmem<8x8x1xf32>) {
            reduce sum(%output: gmem<8x8x1xf32>, %src: gmem<8x8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 4},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )

    with pytest.raises(
        AssertionError,
        match="grid-to-block and block-to-intra-block",
    ):
        tile_reduce(
            module,
            tile_sizes={1: 4},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_dependent_levels_across_invocations():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<64x64x64xf32>, %output: gmem<64x64x1xf32>) {
            reduce sum(%output: gmem<64x64x1xf32>, %src: gmem<64x64x64xf32>) collective=[x.block, yz.grid] attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 32},
        parallel_iteration_space=(
            CollectiveLevel.WARP, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
    )

    with pytest.raises(
        AssertionError,
        match="grid-to-block and block-to-intra-block",
    ):
        tile_reduce(
            module,
            tile_sizes={1: 1},
            parallel_iteration_space=(
                CollectiveLevel.THREAD, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_intra_block_tiling_before_block_tiling():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<64x64xf32>, %output: gmem<64x1xf32>) {
            reduce sum(%output: gmem<64x1xf32>, %src: gmem<64x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="grid-to-block and block-to-intra-block",
    ):
        tile_reduce(
            module,
            tile_sizes={0: 32},
            parallel_iteration_space=(
                CollectiveLevel.WARP, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        )


@pytest.mark.parametrize(
    ("level", "match"), [
        (CollectiveLevel.CLUSTER, "cluster-aware iteration spaces"),
        (
            CollectiveLevel.GRID,
            "grid-to-block and block-to-intra-block",
        ),
    ],
)
def test_tile_reduce_rejects_unsupported_iteration_space(level, match):
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x8x64xf32>, %output: gmem<8x8x1xf32>) {
            reduce sum(%output: gmem<8x8x1xf32>, %src: gmem<8x8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match=match,
    ):
        tile_reduce(
            module,
            tile_sizes={0: 4},
            parallel_iteration_space=(level, CollectiveAxis.X),
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_clustered_kernel():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x8x64xf32>, %output: gmem<8x8x1xf32>) cluster_dim=[2, 1, 1] {
            reduce sum(%output: gmem<8x8x1xf32>, %src: gmem<8x8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="cluster-aware iteration spaces",
    ):
        tile_reduce(
            module,
            tile_sizes={0: 4},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        )
def test_tile_reduce_rejects_non_divisible_dimension():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x65xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x65xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="dimension 1 size 65 to be divisible by tile size 32",
    ):
        tile_reduce(
            module,
            tile_sizes={1: 32},
            parallel_iteration_space=None,
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_tile_larger_than_dimension():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match=r"dimension 0 tile size to be in \[1, 8\]; got 16",
    ):
        tile_reduce(
            module,
            tile_sizes={0: 16},
            parallel_iteration_space=None,
            hero_label_filter="mean",
        )


@pytest.mark.parametrize("dim", [-1, 2])
def test_tile_reduce_rejects_out_of_range_dimension(dim):
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match=r"dimensions must be in \[0, 2\)",
    ):
        tile_reduce(
            module,
            tile_sizes={dim: 1},
            parallel_iteration_space=None,
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_non_tile_destination_view_chain():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<2x4x1xf32>) {
            %view: gmem<8x1xf32> = reshape(%output: gmem<2x4x1xf32>)
            reduce sum(%view: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="only supports TileOp destination view chains",
    ):
        tile_reduce(
            module,
            tile_sizes={1: 32},
            parallel_iteration_space=None,
            hero_label_filter="mean",
        )


def test_tile_reduce_rejects_parallel_and_reduction_dimensions_sequentially():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="parallel and reduction dimensions in one invocation",
    ):
        tile_reduce(
            module,
            tile_sizes={0: 4, 1: 32},
            parallel_iteration_space=None,
            hero_label_filter="mean",
        )


def test_tile_reduce_replays_tiled_destination_for_reduction_tiling():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            %0: gmem<4x64xf32> = tile(%src: gmem<8x64xf32>, 0, 0)
            %1: gmem<4x1xf32> = tile(%output: gmem<8x1xf32>, 0, 0)
            reduce sum(%1: gmem<4x1xf32>, %0: gmem<4x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="mean",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            %0: gmem<4x64xf32> = tile(%src: gmem<8x64xf32>, 0, 0)
            %1: gmem<4x1xf32> = tile(%output: gmem<8x1xf32>, 0, 0)
            %2: gmem<8x2xf32> = alloc() attrs={id: 0}
            %3: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%2: gmem<8x2xf32>, %3: rmem<f32>)
            %4: gmem<4x2xf32> = tile(%2: gmem<8x2xf32>, 0, 0)
            for (%5: index = 0 to 2) attrs={label: "d1"} {
              %6: gmem<4x1xf32 strides=[2, 1]> = tile(%4: gmem<4x2xf32>, 0, %5: index)
              %7: gmem<4x32xf32 strides=[64, 1]> = tile(%0: gmem<4x64xf32>, 0, %5: index)
              reduce sum(%6: gmem<4x1xf32 strides=[2, 1]>, %7: gmem<4x32xf32 strides=[64, 1]>) collective=xyz.grid attrs={label: "mean_partial"}
            }
            reduce sum(%1: gmem<4x1xf32>, %4: gmem<4x2xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }""")  # noqa: E501


def test_tile_reduce_uses_reduction_neutral_element_for_partials():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @maximum(%src: gmem<8x64xf32>, %output: gmem<8x1xf32>) {
            reduce max(%output: gmem<8x1xf32>, %src: gmem<8x64xf32>) collective=xyz.grid attrs={label: "maximum"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={1: 32},
        parallel_iteration_space=None,
        hero_label_filter="maximum",
    )

    text = repr(module)
    assert "constant() attrs={value: -inf}" in text
    assert text.count("elementwise fill(") == 1
    assert text.count("reduce max(") == 2
    assert repr(gpu_dialect.parse_module(text)) == text


def test_tile_reduce_rejects_multiple_parallel_and_reduction_dimensions_on_block_x():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x8x64xf32>, %output: gmem<8x8x1xf32>) {
            reduce sum(%output: gmem<8x8x1xf32>, %src: gmem<8x8x64xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
          }
        }"""))

    with pytest.raises(
        AssertionError,
        match="parallel and reduction dimensions in one invocation",
    ):
        tile_reduce(
            module,
            tile_sizes={0: 4, 1: 4, 2: 32},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        )


def test_tile_reduce_maps_one_dimension_to_block_y():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) {
            reduce sum(%output: gmem<256x128x1xf32>, %src: gmem<256x128x6656xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<256x128x1xf32>, %output: gmem<256x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 128},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.Y,
        ),
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) grid_dim=[None, 2, None] {
            %by: index = block_id.y
            %0: gmem<128x128x6656xf32> = tile(%src: gmem<256x128x6656xf32>, %by: index, 0, 0)
            %1: gmem<128x128x1xf32> = tile(%output: gmem<256x128x1xf32>, %by: index, 0, 0)
            reduce sum(%1: gmem<128x128x1xf32>, %0: gmem<128x128x6656xf32>) collective=[xz.grid, y.block] attrs={init: 0.0, label: "mean"}
            elementwise div(%1: gmem<128x128x1xf32>, %1: gmem<128x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }""")


def test_tile_reduce_combines_parallel_and_sequential_dimensions():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) {
            reduce sum(%output: gmem<256x128x1xf32>, %src: gmem<256x128x6656xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<256x128x1xf32>, %output: gmem<256x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 128},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.Y,
        ),
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = tile_reduce(
        module,
        tile_sizes={1: 64},
        parallel_iteration_space=None,
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) grid_dim=[None, 2, None] {
            %by: index = block_id.y
            %0: gmem<128x128x6656xf32> = tile(%src: gmem<256x128x6656xf32>, %by: index, 0, 0)
            %1: gmem<128x128x1xf32> = tile(%output: gmem<256x128x1xf32>, %by: index, 0, 0)
            for (%2: index = 0 to 2) attrs={label: "d1"} {
              %3: gmem<128x64x6656xf32 strides=[851968, 6656, 1]> = tile(%0: gmem<128x128x6656xf32>, 0, %2: index, 0)
              %4: gmem<128x64x1xf32 strides=[128, 1, 1]> = tile(%1: gmem<128x128x1xf32>, 0, %2: index, 0)
              reduce sum(%4: gmem<128x64x1xf32 strides=[128, 1, 1]>, %3: gmem<128x64x6656xf32 strides=[851968, 6656, 1]>) collective=[xz.grid, y.block] attrs={init: 0.0, label: "mean"}
              elementwise div(%4: gmem<128x64x1xf32 strides=[128, 1, 1]>, %4: gmem<128x64x1xf32 strides=[128, 1, 1]>) attrs={divisor: 6656, label: "mean_epilogue"}
            }
          }
        }""")


def test_tile_reduce_maps_two_dimensions_to_one_parallel_iteration_space():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) {
            reduce sum(%output: gmem<256x128x1xf32>, %src: gmem<256x128x6656xf32>) collective=xyz.grid attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<256x128x1xf32>, %output: gmem<256x128x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    module = tile_reduce(
        module,
        tile_sizes={0: 128, 1: 64},
        parallel_iteration_space=(
            CollectiveLevel.BLOCK, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<256x128x6656xf32>, %output: gmem<256x128x1xf32>) grid_dim=[4, None, None] {
            %bx: index = block_id.x
            %c2: index = index_constant() attrs={value: 2}
            %0: index = index_mod(%bx: index, %c2: index)
            %1: index = index_div(%bx: index, %c2: index)
            %2: gmem<128x64x6656xf32 strides=[851968, 6656, 1]> = tile(%src: gmem<256x128x6656xf32>, %0: index, %1: index, 0)
            %3: gmem<128x64x1xf32 strides=[128, 1, 1]> = tile(%output: gmem<256x128x1xf32>, %0: index, %1: index, 0)
            reduce sum(%3: gmem<128x64x1xf32 strides=[128, 1, 1]>, %2: gmem<128x64x6656xf32 strides=[851968, 6656, 1]>) collective=[x.block, yz.grid] attrs={init: 0.0, label: "mean"}
            elementwise div(%3: gmem<128x64x1xf32 strides=[128, 1, 1]>, %3: gmem<128x64x1xf32 strides=[128, 1, 1]>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }""")


def test_tile_reduce_assigns_one_mean_row_per_warp():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x6656xf32>, %output: gmem<8x1xf32>) {
            reduce sum(%output: gmem<8x1xf32>, %src: gmem<8x6656xf32>) collective=[x.block, yz.grid] attrs={init: 0.0, label: "mean"}
            elementwise div(%output: gmem<8x1xf32>, %output: gmem<8x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }"""))

    warp_size = 32
    chunk_size = 4
    module = tile_reduce(
        module,
        tile_sizes={0: 1},
        parallel_iteration_space=(
            CollectiveLevel.WARP, CollectiveAxis.X,
        ),
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = tile_reduce(
        module,
        tile_sizes={1: warp_size * chunk_size},
        parallel_iteration_space=None,
        hero_label_filter="mean",
        epilogue_label_filter="mean_epilogue",
    )
    module = Canonicalize().run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @mean(%src: gmem<8x6656xf32>, %output: gmem<8x1xf32>) block_dim=[256, None, None] {
            %tx: index = thread_id.x
            %c32: index = index_constant() attrs={value: 32}
            %0: index = index_div(%tx: index, %c32: index)
            %c8: index = index_constant() attrs={value: 8}
            %1: index = index_mod(%0: index, %c8: index)
            %2: gmem<1x6656xf32> = tile(%src: gmem<8x6656xf32>, %1: index, 0)
            %3: gmem<1x1xf32> = tile(%output: gmem<8x1xf32>, %1: index, 0)
            %4: gmem<8x52xf32> = alloc() attrs={id: 0}
            %5: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%4: gmem<8x52xf32>, %5: rmem<f32>)
            %6: gmem<1x52xf32> = tile(%4: gmem<8x52xf32>, %1: index, 0)
            for (%7: index = 0 to 52) attrs={label: "d1"} {
              %8: gmem<1x1xf32 strides=[52, 1]> = tile(%6: gmem<1x52xf32>, 0, %7: index)
              %9: gmem<1x128xf32 strides=[6656, 1]> = tile(%2: gmem<1x6656xf32>, 0, %7: index)
              reduce sum(%8: gmem<1x1xf32 strides=[52, 1]>, %9: gmem<1x128xf32 strides=[6656, 1]>) collective=[x.warp, yz.grid] attrs={label: "mean_partial"}
            }
            reduce sum(%3: gmem<1x1xf32>, %6: gmem<1x52xf32>) collective=[x.warp, yz.grid] attrs={init: 0.0, label: "mean"}
            elementwise div(%3: gmem<1x1xf32>, %3: gmem<1x1xf32>) attrs={divisor: 6656, label: "mean_epilogue"}
          }
        }""")  # noqa: E501
