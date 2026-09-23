import pytest
import torch
from torch.fx import Graph, Node

from ayla.base.ir import find_ops_of_type
from ayla.fx_compiler import fx_region_to_module
from ayla.fx_pipeline import FxRegionMatch
from ayla.gpu_dialect import (
    ElementwiseOp,
    KernelOp,
    MatmulOp,
    TileOp,
)
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect


def _meta(node: Node, shape, dtype=torch.bfloat16, strides=None) -> Node:
    if strides is None:
        node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
    else:
        node.meta["val"] = torch.empty_strided(
            shape, strides, dtype=dtype, device="meta",
        )
    return node


def _swiglu_region(
    *, m: int = 4, hidden: int = 8, intermediate: int = 6,
) -> FxRegionMatch:
    graph = Graph()
    packed_intermediate = 2 * intermediate
    lhs = _meta(graph.placeholder("lhs"), (m, hidden))
    packed_weight = _meta(
        graph.placeholder("packed_weight"), (packed_intermediate, hidden),
    )
    down_weight = _meta(
        graph.placeholder("down_weight"), (hidden, intermediate),
    )
    packed_weight_t = _meta(
        graph.call_function(
            torch.ops.aten.permute.default,
            args=(packed_weight, [1, 0]),
        ),
        (hidden, packed_intermediate),
        strides=(1, hidden),
    )
    projection = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, packed_weight_t),
    ), (m, packed_intermediate))
    gate = _meta(graph.call_function(
        torch.ops.aten.slice.Tensor, args=(projection, 1, 0, intermediate),
    ), (m, intermediate), strides=(packed_intermediate, 1))
    up = _meta(graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(
            projection, 1, intermediate, 9_223_372_036_854_775_807,
        ),
    ), (m, intermediate), strides=(packed_intermediate, 1))
    gate_f32 = _meta(graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(gate, torch.float32),
    ), (m, intermediate), dtype=torch.float32)
    neg = _meta(graph.call_function(
        torch.ops.aten.neg.default, args=(gate_f32,),
    ), (m, intermediate), dtype=torch.float32)
    exp = _meta(graph.call_function(
        torch.ops.aten.exp.default, args=(neg,),
    ), (m, intermediate), dtype=torch.float32)
    denominator = _meta(graph.call_function(
        torch.ops.aten.add.Tensor, args=(exp, 1),
    ), (m, intermediate), dtype=torch.float32)
    div = _meta(graph.call_function(
        torch.ops.aten.div.Tensor, args=(gate_f32, denominator),
    ), (m, intermediate), dtype=torch.float32)
    activation = _meta(graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(div, torch.bfloat16),
    ), (m, intermediate))
    product = _meta(graph.call_function(
        torch.ops.aten.mul.Tensor, args=(activation, up),
    ), (m, intermediate))
    down_weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default,
        args=(down_weight, [1, 0]),
    ), (intermediate, hidden), strides=(1, intermediate))
    output = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(product, down_weight_t),
    ), (m, hidden))
    graph.output(output)
    return FxRegionMatch(output, (lhs, packed_weight, down_weight))


def test_swiglu_fx_region_imports_to_high_level_gpu_ir():
    match = _swiglu_region()
    module = fx_region_to_module(match, name="swiglu")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    assert [value.name for value in kernel.body.arguments] == [
        "lhs", "packed_weight", "down_weight", match.output.name,
    ]
    matmuls = find_ops_of_type(module, MatmulOp)
    assert len(matmuls) == 2
    assert matmuls[0].rhs.name == "packed_weight"
    assert matmuls[1].rhs.name == "down_weight"
    assert len(find_ops_of_type(module, TileOp)) == 2
    assert [
        op.fn for op in find_ops_of_type(module, ElementwiseOp)
    ] == ["convert", "silu", "mul"]


def test_fx_region_import_rejects_intermediate_external_users():
    match = _swiglu_region()
    projection = [
        node for node in match.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    match.output.graph.call_function(torch.ops.aten.neg.default, (projection,))

    with pytest.raises(ValueError, match="region value has external users"):
        fx_region_to_module(match, name="swiglu")
