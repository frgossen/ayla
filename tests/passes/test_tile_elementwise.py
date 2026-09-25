import textwrap

import pytest

from ayla.base.ir import find_ops_of_type
from ayla.gpu_dialect import (
    ElementwiseOp,
    ForOp,
    MemTensorType,
    ModuleOp,
    MoveOp,
    TileOp,
    gpu_dialect,
)
from ayla.passes.tile_elementwise import tile_elementwise


def _parse_module(text: str) -> ModuleOp:
    module = gpu_dialect.parse(text)
    assert isinstance(module, ModuleOp)
    return module


def test_tiles_elementwise_into_a_sequential_iteration_space():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%value: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) {
            elementwise sigmoid(%value: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, %value: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "epilogue"}
          }
        }"""))  # noqa: E501

    tile_elementwise(
        module,
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
        label_filter="epilogue",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @epilogue(%value: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) {
            for (%outn: index = 0 to 4) attrs={label: "outn"} {
              %0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = tile(%value: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">, 0, %outn: index)
              elementwise sigmoid(%0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %0: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "epilogue"}
            }
          }
        }""")  # noqa: E501


def test_tiles_multiple_dimensions_in_one_invocation():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%value: gmem<128x256xf32>) {
            elementwise sigmoid(%value: gmem<128x256xf32>, %value: gmem<128x256xf32>) attrs={label: "epilogue"}
          }
        }"""))  # noqa: E501

    tile_elementwise(
        module,
        tile_dims=[0, 1], tile_sizes=[64, 64],
        tile_levels=["for", "for"], tile_labels=["row", "column"],
        label_filter="epilogue",
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @epilogue(%value: gmem<128x256xf32>) {
            for (%row: index = 0 to 2) attrs={label: "row"} {
              for (%column: index = 0 to 4) attrs={label: "column"} {
                %0: gmem<64x256xf32> = tile(%value: gmem<128x256xf32>, %row: index, 0)
                %1: gmem<64x64xf32 strides=[256, 1]> = tile(%0: gmem<64x256xf32>, 0, %column: index)
                elementwise sigmoid(%1: gmem<64x64xf32 strides=[256, 1]>, %1: gmem<64x64xf32 strides=[256, 1]>) attrs={label: "epilogue"}
              }
            }
          }
        }""")  # noqa: E501


def test_tiles_distinct_conversion_operands_inside_one_loop():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%dst: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %src: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) {
            elementwise convert(%dst: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">, %src: wgrmem<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "convert"}
          }
        }"""))  # noqa: E501

    tile_elementwise(
        module,
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
        label_filter="convert",
    )

    assert len(find_ops_of_type(module, ForOp)) == 1
    assert len(find_ops_of_type(module, TileOp)) == 2
    [convert] = [
        op for op in find_ops_of_type(module, ElementwiseOp)
        if op.fn == "convert"
    ]
    dst_type = convert.dst.type
    src_type = convert.src(0).type
    assert isinstance(dst_type, MemTensorType)
    assert isinstance(src_type, MemTensorType)
    assert tuple(dst_type.sizes) == (64, 64)
    assert tuple(src_type.sizes) == (64, 64)


def test_rejects_negative_dimension():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%value: gmem<64x256xf32>) {
            elementwise sigmoid(%value: gmem<64x256xf32>, %value: gmem<64x256xf32>) attrs={label: "epilogue"}
          }
        }"""))  # noqa: E501

    with pytest.raises(ValueError, match="tile dimension -1 is invalid"):
        tile_elementwise(
            module,
            tile_dims=[-1], tile_sizes=[64],
            tile_levels=["for"], tile_labels=["outn"],
            label_filter="epilogue",
        )


def test_rejects_nonuniform_partition():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%value: gmem<64x256xf32>) {
            elementwise sigmoid(%value: gmem<64x256xf32>, %value: gmem<64x256xf32>) attrs={label: "epilogue"}
          }
        }"""))  # noqa: E501

    with pytest.raises(ValueError, match="is not divisible by tile size 96"):
        tile_elementwise(
            module,
            tile_dims=[1], tile_sizes=[96],
            tile_levels=["for"], tile_labels=["outn"],
            label_filter="epilogue",
        )


def test_rejects_parallel_iteration_binding():
    with pytest.raises(ValueError, match="only the .*for.* tile level"):
        tile_elementwise(
            _parse_module(textwrap.dedent("""\
                module {
                  kernel @epilogue(%value: gmem<64x256xf32>) {
                    elementwise sigmoid(%value: gmem<64x256xf32>, %value: gmem<64x256xf32>) attrs={label: "epilogue"}
                  }
                }""")),  # noqa: E501
            tile_dims=[1], tile_sizes=[64],
            tile_levels=["block.x"], tile_labels=[None],
            label_filter="epilogue",
        )


def test_tiles_an_explicitly_selected_move():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%dst: smem<64x256xbf16>, %src: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) {
            move(%dst: smem<64x256xbf16>, %src: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "writeback"}
          }
        }"""))  # noqa: E501

    tile_elementwise(
        module,
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
        label_filter="writeback",
    )

    assert len(find_ops_of_type(module, ForOp)) == 1
    [move] = find_ops_of_type(module, MoveOp)
    dst_type = move.dst.type
    src_type = move.src.type
    assert isinstance(dst_type, MemTensorType)
    assert isinstance(src_type, MemTensorType)
    assert tuple(dst_type.sizes) == (64, 64)
    assert tuple(src_type.sizes) == (64, 64)


def test_does_not_tile_moves_without_an_explicit_label_filter():
    module = _parse_module(textwrap.dedent("""\
        module {
          kernel @epilogue(%dst: smem<64x256xbf16>, %src: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">) {
            move(%dst: smem<64x256xbf16>, %src: wgrmem<64x256xbf16 warps=4 layout="wgmma_layout_0">)
          }
        }"""))  # noqa: E501

    tile_elementwise(
        module,
        tile_dims=[1], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["outn"],
    )

    assert not find_ops_of_type(module, TileOp)
    assert len(find_ops_of_type(module, MoveOp)) == 1
