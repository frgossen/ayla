"""Translate matched post-grad FX regions into high-level Ayla GPU IR."""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.fx import Node

from .base.ir import Block, Op, Value
from .fx_pipeline import FxRegionMatch
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
)
from .torch_custom_call import from_torch_dtype


_GRID = {axis: CollectiveLevel.GRID for axis in CollectiveAxis}


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
            ops.append(MatmulOp.build(
                values[lhs_node],
                values[rhs_source],
                dst,
                collective=_GRID,
                init=0.0,
                attrs={"label": node.name},
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
            source = _node_argument(node.args[0], owner=node)
            dim, start, _end = node.args[1:4]
            step = node.args[4] if len(node.args) == 5 else 1
            if not isinstance(dim, int) or not isinstance(start, int):
                raise ValueError(f"{node.name}: slice indices must be static")
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

        elementwise_fn: str | None = None
        sources: Iterable[object] = ()
        if node in silu_sources:
            elementwise_fn = "silu"
            sources = [silu_sources[node]]
        elif node.target is torch.ops.prims.convert_element_type.default:
            elementwise_fn = "convert"
            sources = node.args[:1]
        elif node.target is torch.ops.aten.sigmoid.default:
            elementwise_fn = "sigmoid"
            sources = node.args
        elif node.target is torch.ops.aten.silu.default:
            elementwise_fn = "silu"
            sources = node.args
        elif node.target is torch.ops.aten.mul.Tensor:
            elementwise_fn = "mul"
            sources = node.args

        if elementwise_fn is not None:
            source_nodes = [
                _node_argument(source, owner=node) for source in sources
            ]
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
