import torch
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    PatternMatcherPass,
)

from ayla.fx_pipeline import register_fx_rewrite


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
