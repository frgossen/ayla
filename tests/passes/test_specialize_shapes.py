import textwrap

import pytest

from ayla.base.ir import Op, Value, clone_op
from ayla.gpu_dialect import BF16Type, GmemTensorType, ModuleOp, gpu_dialect
from ayla.passes.specialize_shapes import SpecializeShapes, specialize_shapes


def _template() -> ModuleOp:
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @dynamic(%a: gmem<?x11xbf16>, %b: gmem<13x11xbf16>, %c: gmem<?x13xbf16>) {
            matmul(%a: gmem<?x11xbf16>, %b: gmem<13x11xbf16>, %c: gmem<?x13xbf16>)
          }
        }"""))
    assert isinstance(module, ModuleOp)
    return module


def test_specialization_can_mutate_an_explicit_clone():
    template = _template()
    module = clone_op(template, {})
    assert isinstance(module, ModuleOp)

    specialized = specialize_shapes(
        module,
        [(7, 11), (13, 11), (7, 13)],
    )

    assert specialized is module
    signature = specialized.body.ops[0].signature
    assert [list(typ.sizes) for typ in signature] == [[7, 11], [13, 11], [7, 13]]
    assert "?" not in repr(specialized)
    assert "?" in repr(template)


def test_specialize_shapes_is_a_mutating_pass():
    module = _template()

    result = SpecializeShapes(
        [(7, 11), (13, 11), (7, 13)],
    ).run(module)

    assert result is module
    assert "?" not in repr(module)


def test_specialization_rejects_matmul_dimension_conflicts():
    module = _template()
    with pytest.raises(ValueError, match="inconsistent matmul shapes"):
        specialize_shapes(
            module,
            [(7, 11), (13, 11), (8, 13)],
        )


def test_specialization_rejects_unpropagated_dynamic_results():
    module = _template()
    dynamic_result = Value(
        name="dynamic_result",
        type=GmemTensorType.build([None, 11], BF16Type()),
    )
    module.body.ops[0].body.append(Op(operands=[], results=[dynamic_result]))

    with pytest.raises(NotImplementedError, match="propagate dynamic shape"):
        SpecializeShapes([(7, 11), (13, 11), (7, 13)]).run(module)
