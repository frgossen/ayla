import torch
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    PatternMatcherPass,
)

from ayla.fx_pipeline import register_fx_kernel_rewrite, register_fx_rewrite
from ayla.gpu_dialect import BF16Type, GmemTensorType
from ayla.jit_compiler import JITCompiler
from ayla.torch_custom_call import OperandKind


def _fused_side_by_side_swiglu(lhs, gate_weight, up_weight):
    raise AssertionError("graph rewrite tests do not execute the replacement")


def _side_by_side_swiglu_pattern():
    lhs = KeywordArg("lhs")
    return CallFunction(
        torch.ops.aten.mul.Tensor,
        CallFunction(
            torch.ops.aten.silu.default,
            CallFunction(
                torch.ops.aten.mm.default,
                lhs,
                KeywordArg("gate_weight"),
            ),
        ),
        CallFunction(
            torch.ops.aten.mm.default,
            lhs,
            KeywordArg("up_weight"),


def _fused_gemm_sigmoid(lhs, weight):
    raise AssertionError("graph rewrite tests do not execute the replacement")


def _gemm_sigmoid_pattern():
    return CallFunction(
        torch.ops.aten.sigmoid.default,
        CallFunction(
            torch.ops.aten.mm.default,
            KeywordArg("lhs"),
            KeywordArg("weight"),
        ),
    )


def _side_by_side_swiglu_graph() -> tuple[GraphModule, Node, Node, Node]:
    graph = Graph()
    lhs = graph.placeholder("lhs")
    gate_weight = graph.placeholder("gate_weight")
    up_weight = graph.placeholder("up_weight")
    gate = graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, gate_weight),
    )
    up = graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, up_weight),
    )
    activation = graph.call_function(
        torch.ops.aten.silu.default, args=(gate,),
    )
    output = graph.call_function(
        torch.ops.aten.mul.Tensor, args=(activation, up),
    )
    output.meta["val"] = "preserved metadata"
    graph.output(output)
    return GraphModule({}, graph), lhs, gate_weight, up_weight


def test_registered_pattern_replaces_side_by_side_swiglu_region():
    module, lhs, gate_weight, up_weight = _side_by_side_swiglu_graph()
    patterns = PatternMatcherPass(pass_name="ayla_test")
    register_fx_rewrite(
        patterns,
        pattern=_side_by_side_swiglu_pattern(),
        input_names=("lhs", "gate_weight", "up_weight"),
        replacement=_fused_side_by_side_swiglu,


def _gemm_sigmoid_graph() -> tuple[GraphModule, Node, Node]:
    graph = Graph()
    lhs = graph.placeholder("lhs")
    weight = graph.placeholder("weight")
    product = graph.call_function(
        torch.ops.aten.mm.default, args=(lhs, weight),
    )
    output = graph.call_function(
        torch.ops.aten.sigmoid.default, args=(product,),
    )
    output.meta["val"] = "preserved metadata"
    graph.output(output)
    return GraphModule({}, graph), lhs, weight


def test_registered_pattern_replaces_gemm_epilogue_region():
    module, lhs, weight = _gemm_sigmoid_graph()
    patterns = PatternMatcherPass(pass_name="ayla_test")
    register_fx_rewrite(
        patterns,
        pattern=_gemm_sigmoid_pattern(),
        input_names=("lhs", "weight"),
        replacement=_fused_gemm_sigmoid,
    )

    assert patterns.apply(module) == 1

    calls = [
        node for node in module.graph.nodes if node.op == "call_function"
    ]
    assert len(calls) == 1
    assert calls[0].target is _fused_side_by_side_swiglu
    assert calls[0].args == (lhs, gate_weight, up_weight)
    assert calls[0].meta["val"] == "preserved metadata"


def test_fx_kernel_rewrite_registers_shape_routed_custom_op():
    module, lhs, gate_weight, up_weight = _side_by_side_swiglu_graph()
    patterns = PatternMatcherPass(pass_name="ayla_kernel_test")
    bf16 = BF16Type()
    signature = [
        GmemTensorType.build([None, 8], bf16),
        GmemTensorType.build([8, 16], bf16),
        GmemTensorType.build([8, 16], bf16),
        GmemTensorType.build([None, 16], bf16),
    ]
    compiler = JITCompiler(
        name="side_by_side_swiglu",
        signature=signature,
        compile_fn=lambda _module: lambda: lambda *_tensors: None,
    )
    register_fx_kernel_rewrite(
        patterns,
        pattern=_side_by_side_swiglu_pattern(),
        input_names=("lhs", "gate_weight", "up_weight"),
        compiler=compiler,
        signature_mapping=[
            OperandKind.R,
            OperandKind.R,
            OperandKind.R,
            OperandKind.RESULT,
        ],
        namespace="ayla_fx_kernel_rewrite_test",
        infer_result_shapes=lambda input, _gate, _up: [
            (input.shape[0], 16),
        ],
    )
    replacement = (
        torch.ops.ayla_fx_kernel_rewrite_test.side_by_side_swiglu.default
    )

    assert patterns.apply(module) == 1

    calls = [
        node for node in module.graph.nodes if node.op == "call_function"
    ]
    assert len(calls) == 1
    assert calls[0].target is replacement
    assert calls[0].args == (lhs, gate_weight, up_weight)

    result = replacement(
        torch.empty((7, 8), dtype=torch.bfloat16, device="meta"),
        torch.empty((8, 16), dtype=torch.bfloat16, device="meta"),
        torch.empty((8, 16), dtype=torch.bfloat16, device="meta"),
    )
    assert result.shape == (7, 16)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "meta"
