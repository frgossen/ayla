"""Register Ayla rewrites with Inductor's post-grad FX pattern matcher."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch.fx import Node
from torch._inductor.pattern_matcher import (
    Match,
    PatternExpr,
    PatternMatcherPass,
    register_graph_pattern,
)

from .jit_compiler import JITCompiler
from .torch_custom_call import (
    DEFAULT_NAMESPACE,
    OperandKind,
    make_torch_custom_call,
)


def register_fx_rewrite(
    patterns: PatternMatcherPass,
    *,
    pattern: PatternExpr,
    input_names: tuple[str, ...],
    replacement: Callable[..., Any],
    extra_check: Callable[[Match], bool] | None = None,
    prepare: Callable[[Match, tuple[Node, ...]], None] | None = None,
) -> None:
    """Replace one registered FX pattern with a callable operation."""

    @register_graph_pattern(
        pattern,
        extra_check=extra_check or (lambda _match: True),
        pass_dict=cast(Any, patterns),
    )
    def rewrite(torch_match: Match, **captures) -> None:
        inputs = tuple(captures[name] for name in input_names)
        if not all(isinstance(value, Node) for value in inputs):
            raise ValueError("FX rewrite inputs must be graph nodes")
        if prepare is not None:
            prepare(torch_match, inputs)
        output = torch_match.output_node()
        with torch_match.graph.inserting_before(output):
            result = torch_match.graph.call_function(replacement, args=inputs)
        result.meta = dict(output.meta)
        output.replace_all_uses_with(result)
        torch_match.erase_nodes()
        torch_match.graph.eliminate_dead_code()


def register_fx_kernel_rewrite(
    patterns: PatternMatcherPass,
    *,
    pattern: PatternExpr,
    input_names: tuple[str, ...],
    compiler: JITCompiler,
    signature_mapping: list[OperandKind],
    extra_check: Callable[[Match], bool] | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    infer_result_shapes: Callable[..., list[tuple]] | None = None,
    prepare: Callable[[Match, tuple[Node, ...]], None] | None = None,
) -> None:
    """Register a tensor-signature-routed kernel as one FX replacement."""
    replacement = make_torch_custom_call(
        compiler,
        signature_mapping,
        namespace=namespace,
        infer_result_shapes=infer_result_shapes,
    )
    register_fx_rewrite(
        patterns,
        pattern=pattern,
        input_names=input_names,
        replacement=replacement,
        extra_check=extra_check,
        prepare=prepare,
    )
