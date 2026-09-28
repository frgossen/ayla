"""Tests for the peel_loop pass."""
import textwrap

import pytest

from ayla.base.ir import iter_ops
from ayla.gpu_dialect import ForOp, ModuleOp, gpu_dialect
from ayla.passes.peel_loop import PeelLoop, peel_loop


def _strided_loop_module() -> ModuleOp:
    return gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x14xf32>) {
            for (%k: index = 2 to 14 step 2) attrs={label: "k", marker: "keep"} {
              %t: gmem<4x1xf32 strides=[14, 1]> = tile(%a: gmem<4x14xf32>, 0, %k: index) attrs={label: "slice"}
            }
            for (%other: index = 0 to 2) attrs={label: "other"} {
            }
          }
        }"""))


def test_label_filter_is_required():
    with pytest.raises(TypeError, match="label_filter"):
        peel_loop(_strided_loop_module())  # type: ignore[call-arg]


def test_peel_front_and_back_with_a_non_unit_step():
    """Peeled IVs are constants; the residual loop keeps attrs and identity."""
    module = _strided_loop_module()
    original_loop = next(
        op for op in iter_ops(module)
        if isinstance(op, ForOp) and op.attrs.get("label") == "k"
    )

    module = peel_loop(
        module,
        label_filter="k",
        front_iterations=2,
        back_iterations=1,
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x14xf32>) {
            %c2: index = index_constant() attrs={value: 2}
            %t: gmem<4x1xf32 strides=[14, 1]> = tile(%a: gmem<4x14xf32>, 0, %c2: index) attrs={label: "slice_front0"}
            %c4: index = index_constant() attrs={value: 4}
            %t1: gmem<4x1xf32 strides=[14, 1]> = tile(%a: gmem<4x14xf32>, 0, %c4: index) attrs={label: "slice_front1"}
            for (%k: index = 6 to 12 step 2) attrs={label: "k", marker: "keep"} {
              %t2: gmem<4x1xf32 strides=[14, 1]> = tile(%a: gmem<4x14xf32>, 0, %k: index) attrs={label: "slice"}
            }
            %c12: index = index_constant() attrs={value: 12}
            %t3: gmem<4x1xf32 strides=[14, 1]> = tile(%a: gmem<4x14xf32>, 0, %c12: index) attrs={label: "slice_back0"}
            for (%other: index = 0 to 2) attrs={label: "other"} {
            }
          }
        }""")

    residual_loop = next(
        op for op in iter_ops(module)
        if isinstance(op, ForOp) and op.attrs.get("label") == "k"
    )
    assert residual_loop is original_loop


def test_exact_peel_omits_the_empty_middle_loop():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
            }
          }
        }"""))

    module = peel_loop(
        module, label_filter="k", front_iterations=2, back_iterations=2,
    )

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) {
            %c0: index = index_constant() attrs={value: 0}
            %t: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c0: index)
            %c1: index = index_constant() attrs={value: 1}
            %t1: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c1: index)
            %c2: index = index_constant() attrs={value: 2}
            %t2: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c2: index)
            %c3: index = index_constant() attrs={value: 3}
            %t3: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c3: index)
          }
        }""")


def test_rejects_invalid_iteration_counts():
    with pytest.raises(ValueError, match="front_iterations must be non-negative"):
        PeelLoop(label_filter="k", front_iterations=-1)
    with pytest.raises(ValueError, match="back_iterations must be non-negative"):
        PeelLoop(label_filter="k", back_iterations=-1)
    with pytest.raises(ValueError, match="trip count 6"):
        peel_loop(
            _strided_loop_module(),
            label_filter="k",
            front_iterations=4,
            back_iterations=3,
        )


def test_matching_nested_loop_is_peeled_through_nonmatching_parent():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<2x3xf32>) {
            for (%i: index = 0 to 2) attrs={label: "outer"} {
              for (%j: index = 0 to 3) attrs={label: "target"} {
                %cell: gmem<1x1xf32 strides=[3, 1]> = tile(%a: gmem<2x3xf32>, %i: index, %j: index) attrs={label: "cell"}
              }
            }
          }
        }"""))

    module = peel_loop(module, label_filter="target", front_iterations=1)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<2x3xf32>) {
            for (%i: index = 0 to 2) attrs={label: "outer"} {
              %c0: index = index_constant() attrs={value: 0}
              %cell: gmem<1x1xf32 strides=[3, 1]> = tile(%a: gmem<2x3xf32>, %i: index, %c0: index) attrs={label: "cell_front0"}
              for (%j: index = 1 to 3) attrs={label: "target"} {
                %cell1: gmem<1x1xf32 strides=[3, 1]> = tile(%a: gmem<2x3xf32>, %i: index, %j: index) attrs={label: "cell"}
              }
            }
          }
        }""")
