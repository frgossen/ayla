"""Translate matched post-grad FX regions into high-level Ayla GPU IR."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx import Node
from torch._inductor.pattern_matcher import (
    MULTIPLE,
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
)

from .base.ir import Block, Op, Value
from .fx_pipeline import register_fx_kernel_rewrite
from .jit_compiler import (
    JITCompiler,
    LauncherPlan,
    TensorSignature,
)
from .gpu_dialect import (
    AllocOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    ReshapeOp,
    TileOp,
    Type,
)
from .torch_custom_call import (
    DEFAULT_NAMESPACE,
    OperandKind,
    from_torch_dtype,
)


_ELEMENTWISE_OPS: dict[object, tuple[str, int]] = {
    torch.ops.prims.convert_element_type.default: ("convert", 1),
    torch.ops.aten.sigmoid.default: ("sigmoid", 1),
    torch.ops.aten.silu.default: ("silu", 1),
    torch.ops.aten.mul.Tensor: ("mul", 2),
}


@dataclass(frozen=True)
class FxRegionMatch:
    """The boundary of one single-output FX region."""

    output: Node
    inputs: tuple[Node, ...]


# TODO: Derive discovery support from the Ayla importer once primitive math
# operations have complete import and lowering support. Today add, div, exp,
# and neg are included only so discovery can traverse the hardcoded
# decomposed-SiLU pattern; they are not independently importable.
_SUPPORTED_FX_TARGETS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.exp.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.neg.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.view.default,
    torch.ops.prims.convert_element_type.default,
}


def _is_supported_fx_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _SUPPORTED_FX_TARGETS


def discover_supported_fx_region(output: Node) -> FxRegionMatch | None:
    """Find the maximal self-contained supported region ending at ``output``.

    A node is considered an anchor only when no supported operation consumes
    it. Walking backward then identifies supported computation and treats all
    other producers as ordered region inputs. Values escaping from the middle
    of the candidate make it ineligible for replacement.
    """
    if not _is_supported_fx_node(output) or any(
        _is_supported_fx_node(user) for user in output.users
    ):
        return None

    region: set[Node] = set()
    boundary_set: set[Node] = set()
    boundaries: list[Node] = []

    def visit(node: Node) -> None:
        if node in region or node in boundary_set:
            return
        if not _is_supported_fx_node(node):
            boundary_set.add(node)
            boundaries.append(node)
            return
        region.add(node)
        for argument in node.all_input_nodes:
            visit(argument)

    visit(output)
    if any(
        node is not output and set(node.users) - region
        for node in region
    ):
        return None
    return FxRegionMatch(output=output, inputs=tuple(boundaries))


def _node_type(node: Node) -> GmemTensorType:
    tensor = node.meta.get("val")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{node.name}: missing tensor metadata")

    def static(value: object) -> int | None:
        return value if isinstance(value, int) else None

    return GmemTensorType.build(
        [static(size) for size in tensor.shape],
        from_torch_dtype(tensor.dtype),
        strides=[static(stride) for stride in tensor.stride()],
    )


def _region_nodes(match: FxRegionMatch) -> list[Node]:
    inputs = set(match.inputs)
    seen: set[Node] = set()
    ordered: list[Node] = []

    def visit(node: Node) -> None:
        if node in inputs or node in seen:
            return
        seen.add(node)
        for argument in node.all_input_nodes:
            visit(argument)
        ordered.append(node)

    visit(match.output)
    region = set(ordered)
    for node in ordered:
        if node is match.output:
            continue
        outside_users = set(node.users) - region
        if outside_users:
            users = ", ".join(sorted(user.name for user in outside_users))
            raise ValueError(f"{node.name}: region value has external users: {users}")
    return ordered


def _node_argument(value: object, *, owner: Node) -> Node:
    if not isinstance(value, Node):
        raise ValueError(f"{owner.name}: expected tensor node, got {value!r}")
    return value


def _call_node(node: object, target: object, num_args: int) -> Node | None:
    if (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is target
        and len(node.args) == num_args
    ):
        return node
    return None


def _decomposed_silu(
    node: Node,
) -> tuple[Node, set[Node]] | None:
    """Recognize the post-grad ``x * sigmoid(x)`` expansion ending in BF16."""
    # TODO: Import the primitive math operations into Ayla IR, then recognize
    # and lower compound expressions there. Keep this targeted FX match only
    # until neg, exp, add, and div have complete Ayla lowering support.
    conversion = _call_node(
        node, torch.ops.prims.convert_element_type.default, 2,
    )
    if conversion is None or conversion.args[1] is not torch.bfloat16:
        return None
    div = _call_node(conversion.args[0], torch.ops.aten.div.Tensor, 2)
    if div is None:
        return None
    source, denominator = div.args
    denominator = _call_node(
        denominator, torch.ops.aten.add.Tensor, 2,
    )
    if not isinstance(source, Node) or denominator is None:
        return None
    exp, one = denominator.args
    exp = _call_node(exp, torch.ops.aten.exp.default, 1)
    if one != 1 or exp is None:
        return None
    neg = _call_node(exp.args[0], torch.ops.aten.neg.default, 1)
    if neg is None:
        return None
    if (
        neg.args[0] is not source
        or set(neg.users) != {exp}
        or set(exp.users) != {denominator}
        or set(denominator.users) != {div}
        or set(div.users) != {conversion}
    ):
        return None
    return source, {div, denominator, exp, neg}


def _allocate(
    node: Node,
    output: Node,
    output_value: Value,
    ops: list[Op],
) -> Value:
    if node is output:
        return output_value
    alloc = AllocOp.build_with_fresh_id(_node_type(node))
    ops.append(alloc)
    return alloc.result()


def _transpose_source(node: Node) -> Node | None:
    # TODO: Import arbitrary FX permutations once the GPU dialect can express
    # strided aliasing views. For now, recognize only the 2D weight transpose
    # that can be elided because Ayla matmul already expects an [N, K] RHS.
    transpose = _call_node(node, torch.ops.aten.permute.default, 2)
    if transpose is None:
        return None
    if transpose.args[1] not in ([1, 0], (1, 0)):
        return None
    return _node_argument(transpose.args[0], owner=transpose)


def fx_region_to_module(
    match: FxRegionMatch,
    *,
    name: str,
) -> ModuleOp:
    """Import one single-output FX region into unscheduled Ayla GPU IR.

    The importer currently covers the operators needed by the fused GEMM
    examples: transposed-weight matmul, reshape, slicing, BF16/FP32 conversion,
    sigmoid/SiLU, and multiplication. Unsupported operations fail explicitly.
    """
    if match.output in match.inputs:
        raise ValueError("FX region output cannot also be an input")
    nodes = _region_nodes(match)
    values = {
        node: Value(node.name, _node_type(node)) for node in match.inputs
    }
    output_value = Value(match.output.name, _node_type(match.output))
    ops: list[Op] = []
    matmul_index = 0

    silu_sources: dict[Node, Node] = {}
    skipped: set[Node] = set()
    for node in nodes:
        decomposed = _decomposed_silu(node)
        if decomposed is not None:
            source, internal = decomposed
            silu_sources[node] = source
            skipped.update(internal)

    for node in nodes:
        if node in skipped:
            continue
        if node.op != "call_function":
            raise ValueError(
                f"{node.name}: unsupported FX node kind {node.op!r}"
            )

        transpose_source = _transpose_source(node)
        if transpose_source is not None:
            if not all(
                user.target is torch.ops.aten.mm.default
                and len(user.args) == 2
                and user.args[1] is node
                for user in node.users
            ):
                raise ValueError(
                    f"{node.name}: transpose is supported only for matmul weights"
                )
            values[node] = values[transpose_source]
            continue

        if node.target is torch.ops.aten.mm.default:
            lhs_node = _node_argument(node.args[0], owner=node)
            rhs_node = _node_argument(node.args[1], owner=node)
            rhs_source = _transpose_source(rhs_node)
            if rhs_source is None:
                raise ValueError(
                    f"{node.name}: matmul RHS must be a 2D transpose"
                )
            dst = _allocate(node, match.output, output_value, ops)
            label = "mm" if matmul_index == 0 else f"mm_{matmul_index}"
            matmul_index += 1
            ops.append(MatmulOp.build(
                values[lhs_node],
                values[rhs_source],
                dst,
                collective={
                    axis: CollectiveLevel.GRID for axis in CollectiveAxis
                },
                init=0.0,
                attrs={"label": label},
            ))
            values[node] = dst
            continue

        if node.target is torch.ops.aten.view.default:
            source = _node_argument(node.args[0], owner=node)
            reshape = ReshapeOp.build(values[source], _node_type(node))
            ops.append(reshape)
            values[node] = reshape.result()
            continue

        if node.target is torch.ops.aten.slice.Tensor:
            source = cast(Node, node.args[0])
            dim, start, _end = node.args[1:4]
            dim = cast(int, dim)
            step = node.args[4] if len(node.args) == 5 else 1
            if not isinstance(start, int):
                raise ValueError(f"{node.name}: slice start must be static")
            if not isinstance(step, int):
                raise ValueError(f"{node.name}: slice step must be static")
            if step != 1:
                raise ValueError(f"{node.name}: strided slices are unsupported")
            source_type = values[source].type
            if not isinstance(source_type, GmemTensorType):
                raise ValueError(f"{node.name}: slice source must be gmem")
            indices: list[Value | int] = [0] * source_type.rank
            indices[dim] = start
            tile = TileOp.build(values[source], indices, _node_type(node))
            ops.append(tile)
            values[node] = tile.result()
            continue

        if node in silu_sources:
            elementwise_fn = "silu"
            source_nodes = [silu_sources[node]]
        else:
            elementwise = _ELEMENTWISE_OPS.get(node.target)
            if elementwise is None:
                elementwise_fn = None
                source_nodes = []
            else:
                elementwise_fn, num_sources = elementwise
                source_nodes = [
                    _node_argument(source, owner=node)
                    for source in node.args[:num_sources]
                ]

        if elementwise_fn is not None:
            dst = _allocate(node, match.output, output_value, ops)
            ops.append(ElementwiseOp.build(
                dst,
                [values[source] for source in source_nodes],
                elementwise_fn,
                attrs={"label": node.name},
            ))
            values[node] = dst
            continue

        raise ValueError(f"{node.name}: unsupported FX target {node.target}")

    kernel = KernelOp.build(
        name,
        Block(
            arguments=[
                *(values[node] for node in match.inputs), output_value,
            ],
            ops=ops,
        ),
    )
    return ModuleOp.build(Block(ops=[kernel]))


def _match_signature(match: FxRegionMatch) -> TensorSignature:
    tensors = []
    for node in (*match.inputs, match.output):
        tensor = node.meta.get("val")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{node.name}: missing tensor metadata")
        tensors.append((
            tuple(int(size) for size in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
            tensor.dtype,
        ))
    return tuple(tensors)


def make_fx_to_ayla_rewrite(
    patterns: PatternMatcherPass,
    *,
    root: object,
    name: str,
    signature: list[Type],
    signature_mapping: list[OperandKind],
    compile_region: Callable[[ModuleOp], LauncherPlan],
    accept: Callable[[FxRegionMatch], bool] | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    infer_result_shapes: Callable[..., list[tuple]] | None = None,
) -> None:
    """Build a rewrite that compiles matched FX regions through Ayla."""
    if len(signature) != len(signature_mapping):
        raise ValueError("signature and signature mapping must have equal length")
    if signature_mapping[-1:] != [OperandKind.RESULT] or (
        signature_mapping.count(OperandKind.RESULT) != 1
    ):
        raise ValueError("FX regions require exactly one final result")
    if not all(isinstance(typ, GmemTensorType) for typ in signature):
        raise ValueError("FX region signatures require gmem tensors")
    expected_types = [
        typ for typ in signature if isinstance(typ, GmemTensorType)
    ]
    compiler = JITCompiler(
        name=name,
        signature=signature,
        compile_fn=compile_region,
    )

    def region(torch_match: Match) -> FxRegionMatch | None:
        candidate = discover_supported_fx_region(torch_match.output_node())
        if candidate is None or not _matches_signature(
            candidate, expected_types,
        ):
            return None
        if accept is not None and not accept(candidate):
            return None
        return candidate

    def resolve_inputs(torch_match: Match) -> tuple[Node, ...]:
        candidate = region(torch_match)
        if candidate is None:
            raise RuntimeError("accepted FX match no longer describes a region")
        return candidate.inputs

    def prepare(torch_match: Match, inputs: tuple[Node, ...]) -> None:
        candidate = FxRegionMatch(torch_match.output_node(), inputs)
        compiler.prepare_module(
            _match_signature(candidate),
            fx_region_to_module(candidate, name=name),
        )

    register_fx_kernel_rewrite(
        patterns,
        pattern=CallFunctionVarArgs(root, users=MULTIPLE),
        resolve_inputs=resolve_inputs,
        compiler=compiler,
        signature_mapping=signature_mapping,
        extra_check=lambda torch_match: region(torch_match) is not None,
        namespace=namespace,
        infer_result_shapes=infer_result_shapes,
        prepare=prepare,
    )


def _matches_signature(
    match: FxRegionMatch,
    signature: list[GmemTensorType],
) -> bool:
    actual = (*match.inputs, match.output)
    if len(actual) != len(signature):
        return False

    for node, expected in zip(actual, signature, strict=True):
        tensor = node.meta.get("val")
        if not isinstance(tensor, torch.Tensor) or tensor.device.type not in (
            "cuda", "meta",
        ):
            return False
        found = _node_type(node)
        if found.dtype != expected.dtype or found.rank != expected.rank:
            return False
        if any(
            declared is not None and declared != concrete
            for declared, concrete in zip(
                expected.sizes, found.sizes, strict=True,
            )
        ):
            return False
        if any(
            declared is not None and declared != concrete
            for declared, concrete in zip(
                expected.strides, found.strides, strict=True,
            )
        ):
            return False
    return True
