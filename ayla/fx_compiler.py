"""Translate matched post-grad FX regions into high-level Ayla GPU IR."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
from .jit_compiler import JITCompiler, LauncherPlan, TensorSignature
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
    Type,
)
from .torch_custom_call import (
    DEFAULT_NAMESPACE,
    OperandKind,
    from_torch_dtype,
)


# Elementwise FX targets mapped to their Ayla function name and source count.
_ELEMENTWISE_OPS: dict[object, tuple[str, int]] = {
    torch.ops.aten.sigmoid.default: ("sigmoid", 1),
}

# Matmul FX targets mapped to the argument positions of their accumulator
# initializer (``None`` for zero), LHS, and transposed-weight RHS.
_MATMUL_OPS: dict[object, tuple[int | None, int, int]] = {
    torch.ops.aten.mm.default: (None, 0, 1),
    torch.ops.aten.addmm.default: (0, 1, 2),
}


@dataclass(frozen=True)
class FxRegionMatch:
    """The boundary of one single-output FX region."""

    output: Node
    inputs: tuple[Node, ...]


# Discovery traverses exactly what ``fx_region_to_module`` can import.
_SUPPORTED_FX_TARGETS = {
    torch.ops.aten.permute.default,
    *_MATMUL_OPS,
    *_ELEMENTWISE_OPS,
}


def _is_supported_fx_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _SUPPORTED_FX_TARGETS


def discover_supported_fx_region(
    output: Node,
    *,
    require_maximal: bool = True,
) -> FxRegionMatch | None:
    """Find the maximal self-contained supported region ending at ``output``.

    A node is considered an anchor only when no supported operation consumes
    it. Walking backward then identifies supported computation and treats all
    other producers as ordered region inputs. Values escaping from the middle
    of the candidate make it ineligible for replacement.
    """
    if not _is_supported_fx_node(output) or (
        require_maximal
        and any(_is_supported_fx_node(user) for user in output.users)
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


def _node_tensor(node: Node) -> torch.Tensor:
    tensor = node.meta.get("val")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{node.name}: missing tensor metadata")
    return tensor


def _tensor_type(tensor: torch.Tensor) -> GmemTensorType:
    def static(value: object) -> int | None:
        return value if isinstance(value, int) else None

    return GmemTensorType.build(
        [static(size) for size in tensor.shape],
        from_torch_dtype(tensor.dtype),
        strides=[static(stride) for stride in tensor.stride()],
    )


def _node_type(node: Node) -> GmemTensorType:
    return _tensor_type(_node_tensor(node))


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

    The importer covers what GEMM regions need: a matmul from
    ``_MATMUL_OPS`` whose RHS is a transposed weight, followed by elementwise
    epilogues from ``_ELEMENTWISE_OPS``. Unsupported operations fail
    explicitly.
    """
    if match.output in match.inputs:
        raise ValueError("FX region output cannot also be an input")
    nodes = _region_nodes(match)
    region = set(nodes)
    values = {
        node: Value(node.name, _node_type(node)) for node in match.inputs
    }
    output_value = Value(match.output.name, _node_type(match.output))
    ops: list[Op] = []
    matmul_index = 0

    for node in nodes:
        if node.op != "call_function":
            raise ValueError(
                f"{node.name}: unsupported FX node kind {node.op!r}"
            )

        transpose_source = _transpose_source(node)
        if transpose_source is not None:
            if not all(
                user.target in _MATMUL_OPS
                and user.args[_MATMUL_OPS[user.target][2]] is node
                for user in node.users
            ):
                raise ValueError(
                    f"{node.name}: transpose is supported only for matmul weights"
                )
            values[node] = values[transpose_source]
            continue

        matmul = _MATMUL_OPS.get(node.target)
        if matmul is not None:
            init_index, lhs_index, rhs_index = matmul
            if any(node.kwargs.get(scale, 1) != 1 for scale in ("alpha", "beta")):
                raise ValueError(f"{node.name}: scaled matmuls are unsupported")
            lhs_node = _node_argument(node.args[lhs_index], owner=node)
            rhs_node = _node_argument(node.args[rhs_index], owner=node)
            rhs_source = _transpose_source(rhs_node)
            if rhs_source is None:
                raise ValueError(
                    f"{node.name}: matmul RHS must be a 2D transpose"
                )
            dst = _allocate(node, match.output, output_value, ops)
            label = "mm" if matmul_index == 0 else f"mm_{matmul_index}"
            matmul_index += 1
            if init_index is None:
                zero = ConstantOp.build(
                    0.0, RmemTensorType.build([], F32Type()),
                )
                ops.append(zero)
                init = zero.result()
            else:
                bias = _node_argument(node.args[init_index], owner=node)
                region_users = set(bias.users) & region
                if bias not in match.inputs or region_users != {node}:
                    raise ValueError(
                        f"{node.name}: matmul bias must be a region input"
                    )
                # Torch broadcasts the bias to the output; Ayla reads the same
                # memory through the zero strides of that broadcast view.
                init = values[bias] = Value(bias.name, _tensor_type(
                    _node_tensor(bias).expand(_node_tensor(node).shape),
                ))
            ops.append(MatmulOp.build(
                values[lhs_node],
                values[rhs_source],
                dst,
                collective={
                    axis: CollectiveLevel.GRID for axis in CollectiveAxis
                },
                init=init,
                attrs={"label": label},
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
    require_maximal: bool = True,
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
        candidate = discover_supported_fx_region(
            torch_match.output_node(), require_maximal=require_maximal,
        )
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
