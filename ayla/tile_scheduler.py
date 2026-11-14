"""Callable static scheduling for supported, shape-specialized inference graphs.

    scheduled = compile_workload(compute, example_inputs)
    output = scheduled(*example_inputs)

The compiler diagnoses unsupported operations. Outputs are reused on subsequent
calls; clone them to retain a result. Import profiling separately when needed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch.fx.node import map_arg
from torch.fx.experimental.proxy_tensor import make_fx

from .base.ir import clone_op
from .forward_compile import compile_forward_module
from .fx_compiler import discover_fx_graph, fx_region_to_module
from .gpu_dialect import GmemTensorType, KernelOp, ModuleOp
from .ptx_codegen import PtxKernel
from .torch_custom_call import to_torch_dtype


def _signature(value: torch.Tensor | int):
    if isinstance(value, torch.Tensor):
        return value.device, tuple(value.shape), tuple(value.stride()), value.dtype
    return type(value), value


def _workload_device(inputs):
    if not inputs or any(not isinstance(t, torch.Tensor) and type(t) is not int for t in inputs):
        raise ValueError("inputs must be a nonempty sequence of tensors and static integers")
    tensors = [t for t in inputs if isinstance(t, torch.Tensor)]
    if not tensors or tensors[0].device.type != "cuda" or any(t.device != tensors[0].device for t in tensors):
        raise ValueError("workload tensors must be on one CUDA device")
    return tensors[0].device


class CompiledWorkload:
    """One fixed tensor signature, device, launch plan, and reusable output set."""

    def __init__(self, compute, inputs, *, scheduler, name, emit_kernel):
        device = self.device = _workload_device(inputs)
        self._signatures = tuple(_signature(t) for t in inputs)
        with torch.cuda.device(device), torch.no_grad():
            fx = make_fx(compute)(*inputs)
            match = discover_fx_graph(fx.graph)
            self.module = fx_region_to_module(match, name=name)
            placeholders = [n for n in fx.graph.nodes if n.op == "placeholder"]
            self._bindings: list[int | torch.Tensor] = []
            for node in match.inputs:
                if node.op == "placeholder":
                    self._bindings.append(placeholders.index(node))
                else:
                    value = fx
                    for part in str(node.target).split("."):
                        value = getattr(value, part)
                    if not isinstance(value, torch.Tensor) or value.device != device:
                        raise ValueError(f"{node.name}: expected a tensor on {device}")
                    self._bindings.append(value)
            kernel = cast(KernelOp, self.module.body.ops[0])
            self.outputs = tuple(
                torch.empty_strided(
                    cast(GmemTensorType, typ).static_shape,
                    cast(tuple[int, ...], cast(GmemTensorType, typ).strides),
                    dtype=to_torch_dtype(cast(GmemTensorType, typ).dtype),
                    device=device,
                )
                for typ in kernel.signature[len(match.inputs):]
            )
            self._output_nodes = dict(zip(match.outputs, self.outputs, strict=True))
            self._output_tree = next(n for n in fx.graph.nodes if n.op == "output").args[0]
            self.plan = compile_forward_module(
                cast(ModuleOp, clone_op(self.module, {})),
                tile_scheduler=scheduler,
                emit_kernel=emit_kernel,
            )
            self._launch = self.plan()

    def arguments(self, inputs):
        """Resolve graph inputs (including closed-over tensors) and output buffers."""
        return [inputs[b] if isinstance(b, int) else b for b in self._bindings] + list(self.outputs)

    def __call__(self, *inputs: torch.Tensor | int):
        if tuple(_signature(t) for t in inputs) != self._signatures:
            raise ValueError("inputs differ from the compiled device, shape, stride or dtype")
        self._launch(*self.arguments(inputs))
        return map_arg(self._output_tree, lambda node: self._output_nodes[node])


def compile_workload(
    compute: Callable,
    inputs: Sequence[torch.Tensor | int],
    *,
    scheduler: str | None = "static",
    name: str = "ayla_workload",
    emit_kernel: Callable[[ModuleOp], PtxKernel] | None = None,
) -> CompiledWorkload:
    """Compile an arbitrary supported callable for its example tensor signature.

    Tensor inputs must share a CUDA device; integer shape arguments are
    specialized and checked on each call. ``scheduler=None`` selects the
    resident baseline. ``emit_kernel`` is an
    optional profiling hook receiving scheduled IR; normal callers omit it.
    This is static inference: gradients and signature changes are unsupported.
    """
    if scheduler not in (None, "static"):
        raise ValueError("scheduler must be None or 'static'")
    return CompiledWorkload(
        compute, tuple(inputs), scheduler=scheduler, name=name, emit_kernel=emit_kernel
    )
