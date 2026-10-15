import pytest
import torch
from torch.fx import Graph, Node

from ayla.base.ir import find_ops_of_type
from ayla.fx_compiler import FxRegionMatch, fx_region_to_module
from ayla.gpu_dialect import (
    AllocOp,
    ConstantOp,
    ElementwiseOp,
    KernelOp,
    MatmulOp,
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


def _gemm_region(
    *, m: int = 128, n: int = 64, k: int = 64, sigmoid: bool = False,
) -> FxRegionMatch:
    graph = Graph()
    lhs = _meta(graph.placeholder("lhs"), (m, k))
    weight = _meta(graph.placeholder("weight"), (n, k))
    weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [1, 0]),
    ), (k, n), strides=(1, k))
    output = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, weight_t),
    ), (m, n))
    if sigmoid:
        output = _meta(graph.call_function(
            torch.ops.aten.sigmoid.default, args=(output,),
        ), (m, n))
    graph.output(output)
    return FxRegionMatch(output, (lhs, weight))


def test_gemm_fx_region_imports_to_high_level_gpu_ir():
    match = _gemm_region()
    module = fx_region_to_module(match, name="gemm")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    lhs, weight, output = kernel.body.arguments
    assert [lhs.name, weight.name, output.name] == [
        "lhs", "weight", match.output.name,
    ]
    zero, matmul = kernel.body.ops
    assert isinstance(zero, ConstantOp)
    assert isinstance(matmul, MatmulOp)
    assert (matmul.lhs, matmul.rhs, matmul.dst) == (lhs, weight, output)
    assert matmul.init is zero.result()


def test_gemm_epilogue_imports_through_an_intermediate_allocation():
    match = _gemm_region(sigmoid=True)
    module = fx_region_to_module(match, name="gemm_sigmoid")

    verify_gpu_dialect(module)
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    allocation, _zero, matmul, epilogue = kernel.body.ops
    assert isinstance(allocation, AllocOp)
    assert isinstance(matmul, MatmulOp)
    assert isinstance(epilogue, ElementwiseOp)
    assert matmul.dst is allocation.result()
    assert epilogue.fn == "sigmoid"
    assert epilogue.src(0) is allocation.result()
    assert epilogue.dst is kernel.body.arguments[-1]
    assert find_ops_of_type(module, MatmulOp) == [matmul]


def test_fx_region_import_rejects_intermediate_external_users():
    match = _gemm_region(sigmoid=True)
    projection = [
        node for node in match.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    match.output.graph.call_function(torch.ops.aten.neg.default, (projection,))

    with pytest.raises(ValueError, match="region value has external users"):
        fx_region_to_module(match, name="gemm_sigmoid")


def test_fx_region_import_rejects_unsupported_targets():
    match = _gemm_region()
    [lhs, _weight] = match.inputs
    mm = match.output
    with mm.graph.inserting_before(mm):
        negated = _meta(mm.graph.call_function(
            torch.ops.aten.neg.default, args=(lhs,),
        ), (128, 64))
    mm.replace_input_with(lhs, negated)

    with pytest.raises(ValueError, match="unsupported FX target"):
        fx_region_to_module(match, name="gemm_neg")
