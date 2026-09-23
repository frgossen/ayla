import torch
from torch.fx import Graph, GraphModule
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    PatternMatcherPass,
)

from ayla.fx_pipeline import register_fx_rewrite


def _fused_side_by_side_swiglu(lhs, gate_weight, up_weight):
    raise AssertionError("graph rewrite tests do not execute the replacement")


def test_registered_pattern_replaces_side_by_side_swiglu_region():
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
    module = GraphModule({}, graph)

    input_pattern = KeywordArg("lhs")
    patterns = PatternMatcherPass(pass_name="ayla_test")
    register_fx_rewrite(
        patterns,
        pattern=CallFunction(
            torch.ops.aten.mul.Tensor,
            CallFunction(
                torch.ops.aten.silu.default,
                CallFunction(
                    torch.ops.aten.mm.default,
                    input_pattern,
                    KeywordArg("gate_weight"),
                ),
            ),
            CallFunction(
                torch.ops.aten.mm.default,
                input_pattern,
                KeywordArg("up_weight"),
            ),
        ),
        input_names=("lhs", "gate_weight", "up_weight"),
        replacement=_fused_side_by_side_swiglu,
    )

    assert patterns.apply(module) == 1

    calls = [node for node in graph.nodes if node.op == "call_function"]
    assert len(calls) == 1
    assert calls[0].target is _fused_side_by_side_swiglu
    assert calls[0].args == (lhs, gate_weight, up_weight)
    assert calls[0].meta["val"] == "preserved metadata"
