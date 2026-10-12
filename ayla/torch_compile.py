"""Route compatible matmuls through Ayla under ``torch.compile``.

Installs a lowering for compatible BF16 `aten.mm` operations and declines
everything else, which falls through to Inductor's own lowering::

    from ayla import torch_compile
    torch_compile.install()

The node it produces is a `TemplateBuffer` rather than an `ExternKernel`, which
is what lets a consumer fuse into it: Inductor never fuses into an
`ExternKernel`, so a sigmoid after the matmul would become a second kernel
reading 4096x4096 back out of gmem.

Three seams, following `nv_universal_gemm`, the closest thing in Inductor to a
backend whose kernels are prebuilt rather than generated:

- `AylaTemplateBuffer` is the node: a matmul ayla will run.
- `AylaScheduling` preserves the existing fixed-shape sigmoid fusion and
  `codegen_template` emits the selected call.
- `AylaCombinedScheduling` owns the cuda device and hands everything that is
  not an ayla template to the scheduling that would have had it.

Fixed 4096 kernels use the tuned collection entries. Other compatible BF16
matmuls use a dynamic-M template selected by their concrete N and K, with the
existing shape-specializing JIT compiling and caching each concrete M.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

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

_MM_OUT = (
    torch.ops.ayla.auto_matmul_4096_wgmma256_wg_out.default
)
_MM_SIGMOID_OUT = (
    torch.ops.ayla.auto_matmul_4096_wgmma256_wg_sigmoid_out.default
)

_DYNAMIC_M_OPS: dict[tuple[int, int], object] = {}


def _dynamic_m_op(n: int, k: int):
    key = (n, k)
    op = _DYNAMIC_M_OPS.get(key)
    if op is None:
        op = gemm_collection.make_dynamic_m_matmul_out(n, k)
        _DYNAMIC_M_OPS[key] = op
    return op


class AylaTemplateBuffer(ir.TemplateBuffer):
    """A matmul ayla will run."""

    def __init__(
        self, layout, inputs, source_node_name: str, ayla_ops,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render=None)
        self.source_node_name = source_node_name
        self.ayla_ops = ayla_ops

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
        if not _is_elementwise_sigmoid_node(node2):
            return False
        if "sigmoid" not in template.ayla_ops:
            return False
        # Repeated reads within one fused epilogue are fine; a different
        # consumer would still require the unfused matmul result.
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
        template_buffer = template_node.get_template_node()
        assert isinstance(template_buffer, AylaTemplateBuffer)
        source_node_names = [template_buffer.source_node_name]
        ayla_op = template_buffer.ayla_ops[None]
        if epilogue_nodes:
            assert _is_elementwise_sigmoid_node(epilogue_nodes[0]), (
                "can_fuse_vertical let through what it cannot fuse"
            )
            ayla_op = template_buffer.ayla_ops["sigmoid"]
            assert epilogue_nodes[0].node is not None
            source_node_names.extend(
                origin.name
                for origin in epilogue_nodes[0].node.get_origins()
                if origin.target is torch.ops.aten.sigmoid.default
            )

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

        a, b = (inp.codegen_reference() for inp in template_ir_node.inputs)
        V.graph.wrapper_code.writeline(
            f"# ayla nodes: {','.join(source_node_names)}"
        )
        V.graph.wrapper_code.writeline(
            f"torch.ops.{ayla_op}({a}, {b}.t(), {dest})"
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


def _lower_mm(lhs, rhs, *, layout=None):
    """Claim this matmul as an ayla template, or decline and leave it be."""
    source_node = V.graph.current_node
    assert isinstance(source_node, Node)
    m, n, k, layout, lhs, rhs = mm_args(lhs, rhs, layout=layout)
    if lhs.get_device().type != "cuda":
        return None
    if not is_contiguous_strides_for_shape(
        tuple(lhs.get_stride()), tuple(lhs.get_size())
    ):
        return None
    if not is_contiguous_strides_for_shape(
        tuple(reversed(rhs.get_stride())), tuple(reversed(rhs.get_size()))
    ):
        return None
    if lhs.get_dtype() != torch.bfloat16:
        return None

    try:
        n_int, k_int = int(n), int(k)
    except TypeError:
        return None
    if n_int <= 0 or k_int <= 0 or k_int % 64:
        return None

    if (m, k_int, n_int) == (4096, 4096, 4096):
        ayla_ops = {
            None: _MM_OUT,
            "sigmoid": _MM_SIGMOID_OUT,
        }
    else:
        ayla_ops = {None: _dynamic_m_op(n_int, k_int)}

    return TensorBox.create(
        AylaTemplateBuffer(
            layout=layout,
            inputs=[lhs, rhs],
            source_node_name=source_node.name,
            ayla_ops=ayla_ops,
        )
    )


_LOWERING_TARGET = torch.ops.aten.mm.default
_displaced_user_lowering = None
_displaced_backend = None


def install() -> None:
    """Claim ``aten.mm``, and the cuda scheduling that decides what fuses into it."""
    global _displaced_user_lowering, _displaced_backend
    init_backend_registration()
    if _displaced_backend is None:
        _displaced_backend = device_codegens["cuda"]
        register_backend_for_device(
            "cuda", AylaCombinedScheduling, PythonWrapperCodegen, CppWrapperGpu,
        )

    current = user_lowerings.get(_LOWERING_TARGET)
    if current is _lower_mm:
        return
    _displaced_user_lowering = current
    user_lowerings[_LOWERING_TARGET] = _lower_mm


def uninstall() -> None:
    """Hand back ``aten.mm`` and the cuda scheduling."""
    global _displaced_backend
    # Stand down if someone has claimed cuda since, rather than overwrite them.
    installed = device_codegens.get("cuda")
    if installed is not None and installed.scheduling is AylaCombinedScheduling:
        assert _displaced_backend is not None
        device_codegens["cuda"] = _displaced_backend
        _displaced_backend = None

    if user_lowerings.get(_LOWERING_TARGET) is not _lower_mm:
        return
    del user_lowerings[_LOWERING_TARGET]
    if _displaced_user_lowering is not None:
        user_lowerings[_LOWERING_TARGET] = _displaced_user_lowering
