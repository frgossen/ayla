"""Route native-layout FP16/BF16 MM, ADDMM, and BMM through measured JIT schedules."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

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
from torch._inductor.ir import TensorBox
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import user_lowerings
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from . import gemm_collection  # noqa: F401 -- registers collection ops


class AylaTemplateBuffer(ir.TemplateBuffer):
    """A single GEMM called with inputs, contiguous BT, and its output."""

    def __init__(self, layout, inputs, jit_op) -> None:
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

    def can_fuse_vertical(self, node1, node2) -> bool:
        return False

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
        assert not prologue_nodes and not epilogue_nodes
        template_node.mark_run()
        template = template_node.node
        assert isinstance(template, AylaTemplateBuffer)
        *inputs, rhs = (inp.codegen_reference() for inp in template.inputs)
        args = [*inputs, f"{rhs}.transpose(-1, -2)", template.get_name()]
        V.graph.wrapper_code.writeline(
            f"torch.ops.{template.jit_op}({', '.join(args)})"
        )
        self.free_buffers_in_scheduler()
        return None


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
    """Claim a native-layout GEMM that has a supported ``gemm_jit`` schedule."""
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
    return _lower_jit("mm", lhs, rhs, layout=layout)


_LOWERINGS = {
    torch.ops.aten.mm.default: _lower_mm,
    torch.ops.aten.addmm.default: _lower_addmm,
    torch.ops.aten.bmm.default: _lower_bmm,
}
_displaced_user_lowerings: dict[object, Callable[..., Any]] = {}
_displaced_backend = None


def install() -> None:
    """Install measured single-GEMM lowerings for Inductor."""
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
