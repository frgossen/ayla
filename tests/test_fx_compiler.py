import pytest
import torch
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import PatternMatcherPass

from ayla.base.ir import find_ops_of_type
from ayla.fx_compiler import (
    FxRegionMatch,
    discover_supported_fx_region,
    fx_region_to_module,
    make_fx_to_ayla_rewrite,
)
from ayla.gpu_dialect import (
    BF16Type,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    TileOp,
)
from ayla.gemm_collection.gemm import schedule_matmul
from ayla.jit_compiler import PtxRoutePlan
from ayla.passes.verify_gpu_dialect import verify_gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel
from ayla.torch_custom_call import OperandKind


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


def test_discovers_complete_swiglu_region_and_ordered_inputs():
    expected = _swiglu_region()
    # FX/AOT is free to declare placeholders in an order unrelated to use.
    inputs = list(expected.inputs)
    inputs[0].prepend(inputs[1])

    match = discover_supported_fx_region(expected.output)

    assert match is not None
    assert match.output is expected.output
    assert match.inputs == expected.inputs
    projection = [
        node for node in expected.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    assert discover_supported_fx_region(projection) is None


def test_fx_region_import_rejects_intermediate_external_users():
    match = _swiglu_region()
    projection = [
        node for node in match.output.graph.nodes
        if node.target is torch.ops.aten.mm.default
    ][0]
    match.output.graph.call_function(torch.ops.aten.neg.default, (projection,))

    assert discover_supported_fx_region(match.output) is None

    with pytest.raises(ValueError, match="region value has external users"):
        fx_region_to_module(match, name="swiglu")


def _gemm_region() -> FxRegionMatch:
    graph = Graph()
    lhs = _meta(graph.placeholder("lhs"), (128, 64))
    weight = _meta(graph.placeholder("weight"), (64, 64))
    weight_t = _meta(graph.call_function(
        torch.ops.aten.permute.default, args=(weight, [1, 0]),
    ), (64, 64), strides=(1, 64))
    output = _meta(graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, weight_t),
    ), (128, 64))
    graph.output(output)
    return FxRegionMatch(output, (lhs, weight))


def _compile_gemm(module):
    scheduled = schedule_matmul(
        module,
        output_n_tile=64,
        output_tma_inner_size=None,
    )
    return PtxRoutePlan(make_ptx_kernel(scheduled))


def test_compiled_fx_rewrite_prepares_kernel_and_registers_meta_op():
    match = _gemm_region()

    patterns = PatternMatcherPass(pass_name="compiled_fx_gemm_test")
    bf16 = BF16Type()
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.mm.default,
        name="compiled_fx_gemm",
        signature=[
            GmemTensorType.build([128, 64], bf16),
            GmemTensorType.build([64, 64], bf16),
            GmemTensorType.build([128, 64], bf16),
        ],
        signature_mapping=[
            OperandKind.R, OperandKind.R, OperandKind.RESULT,
        ],
        compile_region=_compile_gemm,
        namespace="ayla_compiled_fx_rewrite_test",
    )
    replacement = (
        torch.ops.ayla_compiled_fx_rewrite_test.compiled_fx_gemm.default
    )

    module = GraphModule({}, match.output.graph)
    assert patterns.apply(module) == 1

    calls = [
        node for node in module.graph.nodes
        if node.op == "call_function"
    ]
    assert len(calls) == 1
    assert calls[0].target is replacement
    result = replacement(
        torch.empty((128, 64), dtype=torch.bfloat16, device="meta"),
        torch.empty((64, 64), dtype=torch.bfloat16, device="meta"),
    )
    assert result.shape == (128, 64)
