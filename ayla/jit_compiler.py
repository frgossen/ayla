"""Shape-specializing JIT compilation and launch dispatch."""
from __future__ import annotations

from collections.abc import Callable

import torch

from .base.ir import clone_op
from .gpu_dialect import GmemTensorType, KernelOp, MatmulOp, ModuleOp
from .passes.specialize_shapes import specialize_shapes
from .ptx_runtime import PtxLauncher
from .torch_custom_call import to_torch_dtype


class JITCompiler:
    """Cache one static PTX launcher per concrete tensor signature."""

    def __init__(
        self,
        template: ModuleOp,
        compile_fn: Callable[[ModuleOp], PtxLauncher],
    ) -> None:
        top_ops = template.body.ops
        if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
            raise ValueError(
                "JITCompiler requires exactly one kernel"
            )
        kernel = top_ops[0]
        if len(kernel.body.ops) != 1 or not isinstance(
            kernel.body.ops[0], MatmulOp,
        ):
            raise ValueError(
                "JITCompiler requires exactly one matmul"
            )
        if not all(
            isinstance(typ, GmemTensorType) for typ in kernel.signature
        ):
            raise ValueError(
                "JITCompiler currently supports only gmem arguments"
            )
        if not all(typ.is_canonical_strides for typ in kernel.signature):
            raise ValueError(
                "JITCompiler requires canonical gmem layouts"
            )

        self._template = template
        self._kernel = kernel
        self._compile_fn = compile_fn
        self._cache: dict[tuple[object, ...], PtxLauncher] = {}

    @property
    def name(self) -> str:
        return self._kernel.name

    @property
    def signature(self):
        return self._kernel.signature

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _metadata(
        self, tensors: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[object, ...], list[tuple[int, ...]]]:
        if len(tensors) != len(self.signature):
            raise ValueError(
                f"{self.name}: expected {len(self.signature)} tensors, "
                f"got {len(tensors)}"
            )
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError(f"{self.name}: all tensors must be CUDA tensors")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError(f"{self.name}: all tensors must be contiguous")
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors[1:]):
            raise ValueError(f"{self.name}: all tensors must be on one CUDA device")
        for tensor, typ in zip(tensors, self.signature, strict=True):
            if tensor.dtype != to_torch_dtype(typ.dtype):
                raise ValueError(
                    f"{self.name}: expected dtype {typ.dtype}, got {tensor.dtype}"
                )

        shapes = [tuple(tensor.shape) for tensor in tensors]
        # Dtype and layout are fixed by the validated template. The cache is
        # process-local, so concrete shapes and CUDA device identify an entry.
        return (device, tuple(shapes)), shapes

    def _launcher(self, tensors: tuple[torch.Tensor, ...]) -> PtxLauncher:
        key, shapes = self._metadata(tensors)
        launcher = self._cache.get(key)
        if launcher is not None:
            return launcher

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{self.name}: specialization cache miss during CUDA graph "
                "capture; invoke the kernel once before capture"
            )
        module = clone_op(self._template, {})
        assert isinstance(module, ModuleOp)
        specialize_shapes(module, shapes)
        launcher = self._compile_fn(module)
        self._cache[key] = launcher
        return launcher

    def __call__(self, *tensors: torch.Tensor) -> None:
        launcher = self._launcher(tuple(tensors))
        launcher(*tensors)
