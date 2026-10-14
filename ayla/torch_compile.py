"""Route matmuls through ayla under torch.compile, epilogue and all.

Installs a lowering for `aten.mm` that claims one hard-coded shape and dtype,
and lowerings for `aten.mm`, `aten.addmm`, and `aten.bmm` that claim FP16/BF16
calls in ayla's native layout whenever `ayla.gemm_jit` has a measured schedule
for them. Everything else falls through to Inductor's own lowering::

    from ayla import torch_compile
    torch_compile.install()

The node it produces is a `TemplateBuffer` rather than an `ExternKernel`, which
is what lets a consumer fuse into it: Inductor never fuses into an
`ExternKernel`, so a sigmoid after the matmul would become a second kernel
reading 4096x4096 back out of gmem.

Three seams, following `nv_universal_gemm`, the closest thing in Inductor to a
backend whose kernels are prebuilt rather than generated:

- `AylaTemplateBuffer` is the node: a matmul ayla will run.
- `AylaScheduling` answers `can_fuse_vertical` -- yes for a sigmoid on an ayla
  template -- and `codegen_template`, which emits the call.
- `AylaCombinedScheduling` owns the cuda device and hands everything that is
  not an ayla template to the scheduling that would have had it.

What makes this small is that ayla does not generate a fused kernel on demand:
fusing is swapping in a different prebuilt op. A kernel built per epilogue
would need real codegen here.

The Muse Glimmer prototype recognizes the gated attention-output
``view -> mm -> sigmoid -> mul -> mm`` producer region, translates it to
high-level Ayla IR, and dumps that module. Lowering deliberately raises after
the dump; code generation will follow separately.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import torch
from torch._inductor import ir
from torch._inductor.codegen.common import (
    BackendFeature,
    device_codegens,
    init_backend_registration,
    register_backend_for_device,
)
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.ir import TensorBox, is_contiguous_strides_for_shape
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import user_lowerings
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
)
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.utils._ordered_set import OrderedSet

from . import gemm_collection  # noqa: F401 -- defines the ops below
from .base.ir import Block, Value
from .gpu_dialect import (
    AllocOp,
    BF16Type,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    ReshapeOp,
)

_MM_OUT = torch.ops.ayla.auto_matmul_4096_wgmma256_wg_out.default
_MM_SIGMOID_OUT = torch.ops.ayla.auto_matmul_4096_wgmma256_wg_sigmoid_out.default


@dataclass(frozen=True)
class _GatedAttentionOutputProjectionMatch:
    attention_output: Node
    attention_2d: Node
    gate_input: Node
    gate_weight: Node
    gate_logits: Node
    gate_activation: Node
    gated_attention: Node
    output_projection_weight: Node
    projected_attention: Node


def _match_node_val(
    node: object,
    *,
    shape: tuple[int | None, ...] | None = None,
    strides: tuple[int | None, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> TypeGuard[Node]:
    if not isinstance(node, Node):
        return False
    tensor = node.meta.get("val")
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type != "cuda":
        return False
    if dtype is not None and tensor.dtype != dtype:
        return False
    if shape is not None:
        if len(tensor.shape) != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(
                tensor.shape, shape, strict=True,
            )
        ):
            return False
    if strides is not None:
        if len(tensor.stride()) != len(strides) or any(
            expected is not None and actual != expected
            for actual, expected in zip(
                tensor.stride(), strides, strict=True,
            )
        ):
            return False
    return True


def _only_used_by(node: Node, user: Node) -> bool:
    return len(node.users) == 1 and user in node.users


def _match_node_target(
    node: object,
    target: object,
    *,
    num_args: int,
) -> TypeGuard[Node]:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is target
        and len(node.args) == num_args
    )


def _match_mm(
    node: object,
    *,
    m: int | None,
    n: int | None,
    k: int | None,
    dtype: torch.dtype,
) -> tuple[Node, Node] | None:
    if not _match_node_target(node, torch.ops.aten.mm.default, num_args=2):
        return None
    lhs_node, rhs_node = node.args
    if not _match_node_val(
        lhs_node,
        shape=(m, k),
        strides=(k, 1),
        dtype=dtype,
    ):
        return None
    if not _match_node_val(
        rhs_node,
        shape=(k, n),
        strides=(1, k),
        dtype=dtype,
    ):
        return None
    if not _match_node_val(node, shape=(m, n), dtype=dtype):
        return None
    return lhs_node, rhs_node


def _match_mul(
    node: object,
    *,
    shape: tuple[int | None, ...],
    strides: tuple[int | None, ...],
    dtype: torch.dtype,
) -> tuple[Node, Node] | None:
    if not _match_node_target(node, torch.ops.aten.mul.Tensor, num_args=2):
        return None
    lhs_node, rhs_node = node.args
    if not _match_node_val(
        lhs_node, shape=shape, strides=strides, dtype=dtype,
    ):
        return None
    if not _match_node_val(
        rhs_node, shape=shape, strides=strides, dtype=dtype,
    ):
        return None
    if not _match_node_val(
        node, shape=shape, strides=strides, dtype=dtype,
    ):
        return None
    return lhs_node, rhs_node


def _match_sigmoid(
    node: object,
    *,
    shape: tuple[int | None, ...],
    strides: tuple[int | None, ...],
    dtype: torch.dtype,
) -> Node | None:
    if not _match_node_target(
        node, torch.ops.aten.sigmoid.default, num_args=1,
    ):
        return None
    if not _match_node_val(
        node, shape=shape, strides=strides, dtype=dtype,
    ):
        return None
    arg_node = node.args[0]
    return arg_node if isinstance(arg_node, Node) else None


def _match_reshape(
    node: object,
    *,
    arg_shape: tuple[int | None, ...],
    arg_strides: tuple[int | None, ...],
    result_shape: tuple[int | None, ...],
    result_strides: tuple[int | None, ...],
    dtype: torch.dtype,
) -> Node | None:
    if not _match_node_target(
        node,
        torch.ops.aten.reshape.default,
        num_args=2,
    ):
        return None
    arg_node = node.args[0]
    if not _match_node_val(
        arg_node,
        shape=arg_shape,
        strides=arg_strides,
        dtype=dtype,
    ):
        return None
    if not _match_node_val(
        node,
        shape=result_shape,
        strides=result_strides,
        dtype=dtype,
    ):
        return None
    return arg_node


def _match_permute(
    node: object,
    *,
    arg_shape: tuple[int | None, ...],
    arg_strides: tuple[int | None, ...],
    result_shape: tuple[int | None, ...],
    result_strides: tuple[int | None, ...],
    dtype: torch.dtype,
) -> Node | None:
    if not _match_node_target(
        node, torch.ops.aten.permute.default, num_args=2,
    ):
        return None
    src = node.args[0]
    if not _match_node_val(
        src,
        shape=arg_shape,
        strides=arg_strides,
        dtype=dtype,
    ):
        return None
    if not _match_node_val(
        node,
        shape=result_shape,
        strides=result_strides,
        dtype=dtype,
    ):
        return None
    return src


def _match_gated_attention_output_projection(
    node: Node,
    *,
    m: int | None,
    n: int | None,
    k: int | None,
) -> _GatedAttentionOutputProjectionMatch | None:
    """Match a gated attention-output projection."""
    output_mm_args = _match_mm(
        node,
        m=m,
        n=n,
        k=k,
        dtype=torch.bfloat16,
    )
    if output_mm_args is None:
        return None

    gated_attention, output_weight_transpose = output_mm_args
    if not _only_used_by(gated_attention, node):
        return None
    if not _only_used_by(output_weight_transpose, node):
        return None

    output_projection_weight = _match_permute(
        output_weight_transpose,
        arg_shape=(n, k),
        arg_strides=(k, 1),
        result_shape=(k, n),
        result_strides=(1, k),
        dtype=torch.bfloat16,
    )
    if output_projection_weight is None:
        return None

    mul_args = _match_mul(
        gated_attention,
        shape=(m, k),
        strides=(k, 1),
        dtype=torch.bfloat16,
    )
    if mul_args is None:
        return None

    gate_activation, attention_2d = mul_args
    if not _only_used_by(gate_activation, gated_attention):
        return None
    if not _only_used_by(attention_2d, gated_attention):
        return None

    gate_logits = _match_sigmoid(
        gate_activation,
        shape=(m, k),
        strides=(k, 1),
        dtype=torch.bfloat16,
    )
    if gate_logits is None:
        return None
    if not _only_used_by(gate_logits, gate_activation):
        return None

    attention_output = _match_reshape(
        attention_2d,
        arg_shape=(m, 32, 128),
        arg_strides=(k, 128, 1),
        result_shape=(m, k),
        result_strides=(k, 1),
        dtype=torch.bfloat16,
    )
    if attention_output is None:
        return None

    gate_mm_args = _match_mm(
        gate_logits,
        m=m,
        n=k,
        k=n,
        dtype=torch.bfloat16,
    )
    if gate_mm_args is None:
        return None

    gate_input, gate_weight_transpose = gate_mm_args
    if not _only_used_by(gate_weight_transpose, gate_logits):
        return None

    gate_weight = _match_permute(
        gate_weight_transpose,
        arg_shape=(k, n),
        arg_strides=(n, 1),
        result_shape=(n, k),
        result_strides=(1, n),
        dtype=torch.bfloat16,
    )
    if gate_weight is None:
        return None

    match = _GatedAttentionOutputProjectionMatch(
        attention_output=attention_output,
        attention_2d=attention_2d,
        gate_input=gate_input,
        gate_weight=gate_weight,
        gate_logits=gate_logits,
        gate_activation=gate_activation,
        gated_attention=gated_attention,
        output_projection_weight=output_projection_weight,
        projected_attention=node,
    )
    return match


def _build_gmem_type(node: Node) -> GmemTensorType:
    tensor = node.meta.get("val")
    assert isinstance(tensor, torch.Tensor)
    return GmemTensorType.build(
        list(tensor.shape),
        BF16Type(),
        strides=list(tensor.stride()),
    )


def _build_gmem_value(name: str, node: Node) -> Value:
    return Value(name=name, type=_build_gmem_type(node))


def _build_gated_attention_output_projection_module(
    match: _GatedAttentionOutputProjectionMatch,
) -> ModuleOp:
    attention_output = _build_gmem_value(
        "attention_output", match.attention_output,
    )
    gate_input = _build_gmem_value("gate_input", match.gate_input)
    gate_weight = _build_gmem_value("gate_weight", match.gate_weight)
    output_projection_weight = _build_gmem_value(
        "output_projection_weight", match.output_projection_weight,
    )
    projected_attention = _build_gmem_value(
        "projected_attention", match.projected_attention,
    )

    gate_type = _build_gmem_type(match.gate_logits)

    reshape = ReshapeOp.build(attention_output, gate_type)

    alloc = AllocOp.build_with_fresh_id(gate_type)

    matmul = MatmulOp.build(
        gate_input,
        gate_weight,
        alloc.result(),
        collective={
            CollectiveAxis.X: CollectiveLevel.GRID,
            CollectiveAxis.Y: CollectiveLevel.GRID,
            CollectiveAxis.Z: CollectiveLevel.GRID,
        },
        init=0.0,
        attrs={"label": "gate_projection"},
    )
    sigmoid = ElementwiseOp.build(
        alloc.result(),
        srcs=[alloc.result()],
        fn="sigmoid",
        attrs={"label": "gate_activation"},
    )
    mul = ElementwiseOp.build(
        alloc.result(),
        srcs=[alloc.result(), reshape.result()],
        fn="mul",
        attrs={"label": "gated_attention"},
    )
    matmul2 = MatmulOp.build(
        alloc.result(),
        output_projection_weight,
        projected_attention,
        collective={
            CollectiveAxis.X: CollectiveLevel.GRID,
            CollectiveAxis.Y: CollectiveLevel.GRID,
            CollectiveAxis.Z: CollectiveLevel.GRID,
        },
        init=0.0,
        attrs={"label": "output_projection"},
    )

    body = Block(
        arguments=[
            attention_output,
            gate_input,
            gate_weight,
            output_projection_weight,
            projected_attention,
        ],
        ops=[
            reshape,
            alloc,
            matmul,
            sigmoid,
            mul,
            matmul2,
        ],
    )
    kernel = KernelOp.build("gated_attention_output_projection", body)
    return ModuleOp.build(Block(ops=[kernel]))


def _try_lower_glimmer_attention_subgraph(node: Node) -> None:
    match = _match_gated_attention_output_projection(
        node, m=8, n=6656, k=4096,
    )
    if match is None:
        return
    module = _build_gated_attention_output_projection_module(match)
    raise NotImplementedError(
        "Muse Glimmer attention subgraph lowering is not implemented\n"
        f"{module}"
    )

class AylaTemplateBuffer(ir.TemplateBuffer):
    """A matmul ayla will run.

    ``jit_op``, when set, is a ``gemm_jit`` op called with the inputs (the
    last one transposed back to ``bt``) and the output.
    """

    def __init__(self, layout, inputs, jit_op=None) -> None:
        super().__init__(layout, inputs, make_kernel_render=None)
        self.jit_op = jit_op


class AylaScheduling(BaseScheduling):
    """Fusion decisions and codegen for ayla template nodes."""

    def get_backend_features(
        self, device: torch.device,
    ) -> OrderedSet[BackendFeature]:
        # Never asked: AylaCombinedScheduling answers for the device.
        raise NotImplementedError

    @staticmethod
    def is_ayla_template(node: BaseSchedulerNode) -> bool:
        """Whether ``node`` is, or contains, a matmul ayla claimed."""
        return isinstance(node.get_template_node(), AylaTemplateBuffer)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if not self.is_ayla_template(node1) or len(node1.get_nodes()) != 1:
            return False        # one epilogue: there is no kernel for two
        template = node1.get_template_node()
        assert isinstance(template, AylaTemplateBuffer)
        if template.jit_op is not None:
            return False        # the JIT GEMMs have no fused epilogues
        if not _is_elementwise_sigmoid_node(node2):
            return False
        # Only fuse if nothing else reads the matmul.
        # TODO: generalize -- a kernel that stores the matmul as well as the
        # sigmoid would serve a second reader, which is what SiLU wants.
        (output,) = node1.get_outputs()
        return all(user.node is node2 for user in output.users if not user.is_weak)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return False

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> str | None:
        assert not prologue_nodes, "ayla templates do not fuse a prologue"
        assert len(epilogue_nodes) <= 1, "ayla templates fuse at most one epilogue"
        ayla_op = _MM_OUT
        if epilogue_nodes:
            assert _is_elementwise_sigmoid_node(
                epilogue_nodes[0]
            ), "can_fuse_vertical let through what it cannot fuse"
            ayla_op = _MM_SIGMOID_OUT

        template_node.mark_run()
        for node in epilogue_nodes:
            node.mark_run()

        template_ir_node = template_node.node
        assert template_ir_node is not None
        if epilogue_nodes:
            epilogue_ir_node = epilogue_nodes[0].node
            assert epilogue_ir_node is not None
            dest = epilogue_ir_node.get_name()
            # The unfused matmul's buffer has no reader left.
            V.graph.removed_buffers.add(template_ir_node.get_name())
        else:
            dest = template_ir_node.get_name()

        *inputs, b = (inp.codegen_reference() for inp in template_ir_node.inputs)
        assert isinstance(template_ir_node, AylaTemplateBuffer)
        if template_ir_node.jit_op is not None:
            ayla_op = template_ir_node.jit_op
        bt = f"{b}.transpose(-1, -2)"
        V.graph.wrapper_code.writeline(
            f"torch.ops.{ayla_op}({', '.join([*inputs, bt, dest])})"
        )
        self.free_buffers_in_scheduler()
        return None


def _is_elementwise_sigmoid_node(node: BaseSchedulerNode) -> bool:
    inner = node.node
    if not isinstance(inner, ir.ComputedBuffer):
        return False
    inner = cast(ir.ComputedBuffer, inner)
    data = inner.data
    if not isinstance(data, ir.Pointwise):
        return False
    _, body, _ = inner.get_default_sizes_body()
    ops = [n.target for n in body.root_block.graph.nodes if n.op == "call_method"]
    return ops == ["load", "sigmoid", "store"]


class AylaCombinedScheduling(CUDACombinedScheduling):
    """cuda's scheduling, with ayla templates handled before the rest."""

    def __init__(self, scheduler=None) -> None:
        super().__init__(scheduler)
        self._ayla_scheduling = AylaScheduling(scheduler)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        if AylaScheduling.is_ayla_template(node):
            return self._ayla_scheduling
        return super().choose_node_backend(node)

    def can_fuse_vertical(self, node1, node2) -> bool:
        if AylaScheduling.is_ayla_template(node1):
            return self._ayla_scheduling.can_fuse_vertical(node1, node2)
        if AylaScheduling.is_ayla_template(node2):
            return False
        return super().can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1, node2) -> bool:
        if AylaScheduling.is_ayla_template(node1) or AylaScheduling.is_ayla_template(
            node2
        ):
            return False
        return super().can_fuse_horizontal(node1, node2)

    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes):
        if AylaScheduling.is_ayla_template(template_node):
            return self._ayla_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        return super().codegen_template(
            template_node, epilogue_nodes, prologue_nodes
        )


def _static(*sizes) -> tuple[int, ...] | None:
    try:
        return tuple(int(size) for size in sizes)
    except TypeError:
        return None


def _lower_jit(operation, lhs, rhs, bias=None, *, layout=None):
    """Claim a native-layout GEMM that has a measured ``gemm_jit`` schedule."""
    from .gemm_jit import gemm_op, select_gemm_schedule

    m, n, k, layout, lhs, rhs = mm_args(lhs, rhs, layout=layout)
    dtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(
        lhs.get_dtype(),
    )
    sizes = _static(*lhs.get_size(), n)
    if (
        lhs.get_device().type != "cuda" or dtype is None
        or rhs.get_dtype() != lhs.get_dtype() or sizes is None
    ):
        return None
    *batch, m, k, n = sizes
    # A row-major A and a column-major B: ayla's A @ BT^T with contiguous BT.
    contiguous = list(ir.FlexibleLayout.contiguous_strides(lhs.get_size()))
    if _static(*lhs.get_stride()) != _static(*contiguous):
        return None
    if _static(*rhs.get_stride()) != (*([n * k] if batch else []), 1, k):
        return None
    inputs = [ir.ExternKernel.require_contiguous(lhs), rhs]
    if bias is not None:
        if (
            bias.get_dtype() != lhs.get_dtype()
            or _static(*bias.get_size()) != (n,)
        ):
            return None
        inputs.insert(0, ir.ExternKernel.require_contiguous(bias))
    if select_gemm_schedule(operation, dtype, m, n, k, *batch) is None:
        return None
    return TensorBox.create(AylaTemplateBuffer(
        layout=layout, inputs=inputs, jit_op=gemm_op(operation, dtype),
    ))


def _lower_addmm(bias, lhs, rhs, *, beta=1, alpha=1, layout=None):
    if beta != 1 or alpha != 1:
        return None
    return _lower_jit("addmm", lhs, rhs, bias, layout=layout)


def _lower_bmm(lhs, rhs, *, layout=None):
    return _lower_jit("bmm", lhs, rhs, layout=layout)


def _lower_mm(lhs, rhs, *, layout=None):
    """Claim this matmul as an ayla template, or decline and leave it be."""
    current = V.graph.current_node
    assert isinstance(current, Node)
    _try_lower_glimmer_attention_subgraph(current)

    jit_lhs, jit_rhs = lhs, rhs
    m, n, k, layout, lhs, rhs = mm_args(lhs, rhs, layout=layout)
    if lhs.get_device().type != "cuda":
        return None
    if not is_contiguous_strides_for_shape(
        tuple(reversed(rhs.get_stride())), tuple(reversed(rhs.get_size()))
    ):
        return None

    # TODO: generalize past a single hard-coded kernel. What is missing is
    # deciding between the kernels that could serve a shape, without ayla
    # grading its own kernels inside Inductor's autotuner.
    if (m, k, n, lhs.get_dtype()) != (4096, 4096, 4096, torch.bfloat16):
        return _lower_jit("mm", jit_lhs, jit_rhs)

    return TensorBox.create(
        AylaTemplateBuffer(layout=layout, inputs=[lhs, rhs])
    )


_LOWERINGS = {
    torch.ops.aten.mm.default: _lower_mm,
    torch.ops.aten.addmm.default: _lower_addmm,
    torch.ops.aten.bmm.default: _lower_bmm,
}
_displaced_user_lowerings: dict[object, object] = {}
_displaced_backend = None


def install() -> None:
    """Claim the GEMMs, and the cuda scheduling that decides what fuses into them."""
    global _displaced_backend
    init_backend_registration()
    if _displaced_backend is None:
        _displaced_backend = device_codegens["cuda"]
        register_backend_for_device(
            "cuda", AylaCombinedScheduling, PythonWrapperCodegen, CppWrapperGpu,
        )

    # Register the JIT ops up front: a cached graph can call one without this
    # process ever running the lowering that would create it.
    from .gemm_jit import gemm_op

    for operation in ("mm", "addmm", "bmm"):
        for dtype in ("float16", "bfloat16"):
            gemm_op(operation, dtype)

    for target, lowering in _LOWERINGS.items():
        current = user_lowerings.get(target)
        if current is lowering:
            continue
        if current is not None:
            _displaced_user_lowerings[target] = current
        user_lowerings[target] = lowering


def uninstall() -> None:
    """Hand back the GEMMs and the cuda scheduling."""
    global _displaced_backend
    # Stand down if someone has claimed cuda since, rather than overwrite them.
    installed = device_codegens.get("cuda")
    if installed is not None and installed.scheduling is AylaCombinedScheduling:
        assert _displaced_backend is not None
        device_codegens["cuda"] = _displaced_backend
        _displaced_backend = None

    for target, lowering in _LOWERINGS.items():
        if user_lowerings.get(target) is not lowering:
            continue
        del user_lowerings[target]
        if (displaced := _displaced_user_lowerings.pop(target, None)) is not None:
            user_lowerings[target] = displaced
