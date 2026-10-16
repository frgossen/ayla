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
    assert calls[0].target is _fused_gemm_sigmoid
    assert calls[0].args == (lhs, weight)
    assert calls[0].meta["val"] == "preserved metadata"


def test_fx_kernel_rewrite_registers_shape_routed_custom_op():
    module, lhs, weight = _gemm_sigmoid_graph()
    patterns = PatternMatcherPass(pass_name="ayla_kernel_test")
    bf16 = BF16Type()
    signature = [
        GmemTensorType.build([None, 8], bf16),
        GmemTensorType.build([8, 16], bf16),
        GmemTensorType.build([None, 16], bf16),
    ]
    compiler = JITCompiler(
        name="gemm_sigmoid",
        signature=signature,
        compile_fn=lambda _module: lambda: lambda *_tensors: None,
    )
    register_fx_kernel_rewrite(
        patterns,
        pattern=_gemm_sigmoid_pattern(),
        input_names=("lhs", "weight"),
        compiler=compiler,
        signature_mapping=[
            OperandKind.R,
            OperandKind.R,
            OperandKind.RESULT,
        ],
        namespace="ayla_fx_kernel_rewrite_test",
        infer_result_shapes=lambda input, _weight: [
            (input.shape[0], 16),
        ],
    )
    replacement = torch.ops.ayla_fx_kernel_rewrite_test.gemm_sigmoid.default

    assert patterns.apply(module) == 1

    calls = [
        node for node in module.graph.nodes if node.op == "call_function"
    ]
    assert len(calls) == 1
    assert calls[0].target is replacement
    assert calls[0].args == (lhs, weight)

    result = replacement(
        torch.empty((7, 8), dtype=torch.bfloat16, device="meta"),
        torch.empty((8, 16), dtype=torch.bfloat16, device="meta"),
    )
    assert result.shape == (7, 16)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "meta"
