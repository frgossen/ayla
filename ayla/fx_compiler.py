"""Translate matched post-grad FX regions into high-level Ayla GPU IR."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx import Node
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternExpr,
    PatternMatcherPass,
    is_match,
)

from .base.ir import Block, Op, Value
from .fx_pipeline import register_fx_kernel_rewrite
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
    Type,
)
from .jit_compiler import TensorSignature
from .passes._utils import build_slice_tile
from .ptx_codegen import PtxKernel, make_ptx_kernel
from .ptx_runtime import PtxLauncher, make_ptx_launcher
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


_SILU_SOURCE = KeywordArg("source")
_DECOMPOSED_SILU = CallFunction(
    torch.ops.prims.convert_element_type.default,
    CallFunction(
        torch.ops.aten.div.Tensor,
        _SILU_SOURCE,
        CallFunction(
            torch.ops.aten.add.Tensor,
            CallFunction(
                torch.ops.aten.exp.default,
                CallFunction(torch.ops.aten.neg.default, _SILU_SOURCE),
            ),
            1,
        ),
    ),
    torch.bfloat16,
)


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


def _decomposed_silu(
    node: Node,
) -> tuple[Node, set[Node]] | None:
    """Recognize the post-grad ``x * sigmoid(x)`` expansion ending in BF16."""
    # TODO: Import the primitive math operations into Ayla IR, then recognize
    # and lower compound expressions there. Keep this targeted FX match only
    # until neg, exp, add, and div have complete Ayla lowering support.
    match = _DECOMPOSED_SILU.match(node)
    if not is_match(match):
        return None
    source = match.kwargs["source"]
    assert isinstance(source, Node)
    return source, set(match.nodes) - {node}


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
    ):
        return None
    if node.args[1] not in ([1, 0], (1, 0)):
        return None
    return _node_argument(node.args[0], owner=node)


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
            dim %= source_type.rank
            size = _node_type(node).sizes[dim]
            if not isinstance(size, int):
                raise ValueError(f"{node.name}: slice size must be static")
            tile_index, remainder = divmod(start, size)
            if remainder:
                raise ValueError(f"{node.name}: slice must be tile-aligned")
            tile = build_slice_tile(values[source], dim, tile_index, size)
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


class FxToAylaCompiler:
    """Compile each concrete matched FX signature to one reusable PTX kernel."""

    def __init__(
        self,
        *,
        name: str,
        schedule: Callable[[ModuleOp], ModuleOp],
        compile_module: Callable[[ModuleOp], PtxKernel] = make_ptx_kernel,
    ) -> None:
        self.name = name
        self._schedule = schedule
        self._compile_module = compile_module
        self._kernels: dict[TensorSignature, PtxKernel] = {}

    @property
    def cache_size(self) -> int:
        return len(self._kernels)

    def prepare(
        self,
        torch_match: Match,
        inputs: tuple[Node, ...],
    ) -> None:
        self.prepare_region(FxRegionMatch(torch_match.output_node(), inputs))

    def prepare_region(self, region: FxRegionMatch) -> None:
        signature = _match_signature(region)
        if signature in self._kernels:
            return
        module = fx_region_to_module(region, name=self.name)
        self._kernels[signature] = self._compile_module(self._schedule(module))

    def build_route(self, signature: TensorSignature) -> PtxLauncher:
        kernel = self._kernels.get(signature)
        if kernel is None:
            raise RuntimeError(
                f"{self.name}: no compiled FX region for tensor signature"
            )
        return make_ptx_launcher(kernel)


def make_fx_to_ayla_rewrite(
    patterns: PatternMatcherPass,
    *,
    pattern: PatternExpr,
    input_names: tuple[str, ...],
    name: str,
    signature: list[Type],
    signature_mapping: list[OperandKind],
    schedule: Callable[[ModuleOp], ModuleOp],
    extra_check: Callable[[Match], bool] | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    infer_result_shapes: Callable[..., list[tuple]] | None = None,
) -> torch._ops.OpOverload:
    """Build a rewrite that compiles matched FX regions through Ayla."""
    compiler = FxToAylaCompiler(name=name, schedule=schedule)
    return register_fx_kernel_rewrite(
        patterns,
        pattern=pattern,
        input_names=input_names,
        name=name,
        signature=signature,
        signature_mapping=signature_mapping,
        build_route=compiler.build_route,
        extra_check=extra_check,
        namespace=namespace,
        infer_result_shapes=infer_result_shapes,
        prepare=compiler.prepare,
    )
