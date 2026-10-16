"""Translate matched post-grad FX regions into high-level Ayla GPU IR."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import Node

from .base.ir import Block, Op, Value
from .gpu_dialect import (
    AllocOp,
    CollectiveAxis,
    CollectiveLevel,
    ConstantOp,
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    RmemTensorType,
)
from .torch_custom_call import from_torch_dtype


# Elementwise FX targets mapped to their Ayla function name and source count.
_ELEMENTWISE_OPS: dict[object, tuple[str, int]] = {
    torch.ops.aten.sigmoid.default: ("sigmoid", 1),
}


@dataclass(frozen=True)
class FxRegionMatch:
    """The boundary of one single-output FX region."""

    output: Node
    inputs: tuple[Node, ...]


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
            raise ValueError(
                f"{node.name}: region value has external users: {users}"
            )
    return ordered


def _node_argument(value: object, *, owner: Node) -> Node:
    if not isinstance(value, Node):
        raise ValueError(f"{owner.name}: expected tensor node, got {value!r}")
    return value


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
    if (
        node.op != "call_function"
        or node.target is not torch.ops.aten.permute.default
        or len(node.args) != 2
        or node.args[1] not in ([1, 0], (1, 0))
    ):
        return None
    return _node_argument(node.args[0], owner=node)


def fx_region_to_module(
    match: FxRegionMatch,
    *,
    name: str,
) -> ModuleOp:
    """Import one single-output FX region into unscheduled Ayla GPU IR.

    The importer covers what GEMM regions need: a matmul whose RHS is a
    transposed weight, followed by elementwise epilogues from
    ``_ELEMENTWISE_OPS``. Unsupported operations fail explicitly.
    """
    if match.output in match.inputs:
        raise ValueError("FX region output cannot also be an input")
    nodes = _region_nodes(match)
    values = {
        node: Value(node.name, _node_type(node)) for node in match.inputs
    }
    output_value = Value(match.output.name, _node_type(match.output))
    ops: list[Op] = []

    for node in nodes:
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
            zero = ConstantOp.build(0.0, RmemTensorType.build([], F32Type()))
            ops.append(zero)
            ops.append(MatmulOp.build(
                values[lhs_node],
                values[rhs_source],
                dst,
                collective={
                    axis: CollectiveLevel.GRID for axis in CollectiveAxis
                },
                init=zero.result(),
                attrs={"label": node.name},
            ))
            values[node] = dst
            continue

        elementwise = _ELEMENTWISE_OPS.get(node.target)
        if elementwise is not None:
            elementwise_fn, num_sources = elementwise
            sources = [
                values[_node_argument(source, owner=node)]
                for source in node.args[:num_sources]
            ]
            dst = _allocate(node, match.output, output_value, ops)
            ops.append(ElementwiseOp.build(
                dst, sources, elementwise_fn, attrs={"label": node.name},
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
