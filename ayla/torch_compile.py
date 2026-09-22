"""Route matmuls through ayla under torch.compile, epilogue and all.

Installs a lowering for `aten.mm` that claims one hard-coded shape and dtype
and declines everything else, which falls through to Inductor's own lowering::

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

"""
from __future__ import annotations

from collections.abc import Sequence
import torch
import torch._inductor.config as inductor_config
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
from torch.utils._ordered_set import OrderedSet

from . import gemm_collection  # noqa: F401 -- defines the ops below

_MM_OUT = torch.ops.ayla.auto_matmul_4096_wgmma256_wg_out.default
_MM_SIGMOID_OUT = torch.ops.ayla.auto_matmul_4096_wgmma256_wg_sigmoid_out.default
_MUSE_SWIGLU_OPS: dict[int, object] = {}


class AylaTemplateBuffer(ir.TemplateBuffer):
    """A matmul ayla will run."""

    def __init__(self, layout, inputs) -> None:
        super().__init__(layout, inputs, make_kernel_render=None)


class AylaScheduling(BaseScheduling):
    """Fusion decisions and codegen for ayla template nodes."""

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
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

        if epilogue_nodes:
            dest = epilogue_nodes[0].node.get_name()
            # The unfused matmul's buffer has no reader left.
            V.graph.removed_buffers.add(template_node.node.get_name())
        else:
            dest = template_node.node.get_name()

        a, b = (inp.codegen_reference() for inp in template_node.node.inputs)
        V.graph.wrapper_code.writeline(
            f"torch.ops.{ayla_op}({a}, {b}.t(), {dest})"
        )
        self.free_buffers_in_scheduler()
        return None


def _is_elementwise_sigmoid_node(node: BaseSchedulerNode) -> bool:
    inner = getattr(node, "node", None)
    if not isinstance(inner, ir.ComputedBuffer):
        return False
    if not isinstance(inner.data, ir.Pointwise):
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
        return None

    return TensorBox.create(
        AylaTemplateBuffer(layout=layout, inputs=[lhs, rhs])
    )


_LOWERING_TARGET = torch.ops.aten.mm.default
_displaced_user_lowering = None
_displaced_backend = None
_displaced_post_grad_pass = None
_post_grad_installed = False


def install() -> None:
    """Install Ayla's graph rewrites, lowering, and CUDA scheduling."""
    global _displaced_user_lowering, _displaced_backend
    global _displaced_post_grad_pass, _post_grad_installed
    init_backend_registration()
    if not _post_grad_installed:
        _displaced_post_grad_pass = inductor_config.post_grad_custom_post_pass
        existing = _displaced_post_grad_pass
        if existing is None:
            passes = [_replace_muse_swiglu]
        elif isinstance(existing, (list, tuple)):
            passes = [*existing, _replace_muse_swiglu]
        else:
            passes = [existing, _replace_muse_swiglu]
        inductor_config.post_grad_custom_post_pass = passes
        _post_grad_installed = True
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
    """Restore the displaced graph pass, ``aten.mm``, and CUDA scheduling."""
    global _displaced_backend, _post_grad_installed
    if _post_grad_installed:
        current = inductor_config.post_grad_custom_post_pass
        if (
            isinstance(current, (list, tuple))
            and _replace_muse_swiglu in current
        ):
            inductor_config.post_grad_custom_post_pass = (
                _displaced_post_grad_pass
            )
        _post_grad_installed = False
    # Stand down if someone has claimed cuda since, rather than overwrite them.
    installed = device_codegens.get("cuda")
