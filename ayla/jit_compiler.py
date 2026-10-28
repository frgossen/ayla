"""JIT compilation, tensor specialization, and launch dispatch."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base.ir import clone_op
from .gpu_dialect import (
    ConstantOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    Type,
    TmaDescType,
)
from .passes.specialize_shapes import specialize_shapes
from .ptx_codegen import make_ptx_kernel
from .ptx_runtime import PtxLauncher, make_ptx_launcher
from .torch_custom_call import to_torch_dtype


TensorSignature = tuple[tuple[tuple[int, ...], tuple[int, ...], torch.dtype], ...]
Launcher = Callable[..., None]


class JITCompiler:
    """Compile a static kernel module lazily and cache launchers per device.

    The module must already have its scheduling and launch dimensions assigned.
    PTX lowering and launcher construction happen on the first invocation.
    Tensor metadata is available before compilation for Torch registration.
    """

    def __init__(self, module: ModuleOp) -> None:
        top_ops = module.body.ops
        if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
            raise ValueError("JITCompiler requires exactly one kernel")
        if not all(
            isinstance(typ, (GmemTensorType, TmaDescType))
            for typ in top_ops[0].signature
        ):
            raise ValueError("JITCompiler supports only gmem and TMA arguments")
        self._module = module
        self._static_cache: dict[tuple[torch.device, TensorSignature], Launcher] = {}
        self.name = self._kernel.name
        self.signature: list[Type] = [
            typ for typ in self._kernel.signature if isinstance(typ, GmemTensorType)
        ]

    @property
    def _kernel(self) -> KernelOp:
        [kernel] = self._module.body.ops
        assert isinstance(kernel, KernelOp)
        return kernel

    @property
    def cache_size(self) -> int:
        return len(self._static_cache)

    def _metadata(
        self,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.device, TensorSignature]:
        if len(tensors) != len(self.signature):
            raise ValueError(
                f"{self.name}: expected {len(self.signature)} tensors, "
                f"got {len(tensors)}"
            )
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError(f"{self.name}: all tensors must be CUDA tensors")
        device = (
            tensors[0].device
            if tensors
            else torch.device("cuda", torch.cuda.current_device())
        )
        if any(tensor.device != device for tensor in tensors[1:]):
            raise ValueError(f"{self.name}: all tensors must be on one CUDA device")
        for tensor, typ in zip(tensors, self.signature, strict=True):
            assert isinstance(typ, GmemTensorType)
            if typ.is_canonical_strides:
                if not tensor.is_contiguous():
                    raise ValueError(f"{self.name}: expected contiguous tensors")
            elif tuple(tensor.stride()) != tuple(typ.strides):
                raise ValueError(f"{self.name}: tensor strides must match {typ}")
            if tensor.dtype != to_torch_dtype(typ.dtype):
                raise ValueError(
                    f"{self.name}: expected dtype {typ.dtype}, got {tensor.dtype}"
                )
            shape = tuple(tensor.shape)
            if len(shape) != typ.rank or any(
                declared is not None and declared != concrete
                for declared, concrete in zip(typ.sizes, shape)
            ):
                raise ValueError(
                    f"{self.name}: expected shape {typ.sizes}, got {shape}"
                )

        return device, tuple(
            (tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors
        )

    def _prepare_module(
        self,
        module: ModuleOp,
        shapes: list[tuple[int, ...]],
    ) -> ModuleOp:
        if any(
            isinstance(typ, GmemTensorType) and typ.is_dynamic for typ in self.signature
        ):
            raise ValueError(
                "JITCompiler requires static shapes; use ShapeSpecializingJITCompiler"
            )
        return module

    def _compile(self, module: ModuleOp) -> PtxLauncher:
        launcher = make_ptx_launcher(make_ptx_kernel(module))
        launcher.ensure_loaded()
        return launcher

    def _launcher(self, tensors: tuple[torch.Tensor, ...]) -> Launcher:
        device, signature = self._metadata(tensors)
        key = (device, signature)
        shapes = [entry[0] for entry in signature]
        launcher = self._static_cache.get(key)
        if launcher is not None:
            return launcher
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{self.name}: compilation cache miss during CUDA graph "
                "capture; invoke the kernel once before capture"
            )
        module = clone_op(self._module, {})
        assert isinstance(module, ModuleOp)
        module = self._prepare_module(module, shapes)
        launcher = self._compile(module)
        self._static_cache[key] = launcher
        return launcher

    def __call__(self, *tensors: torch.Tensor) -> None:
        device = tensors[0].device if tensors else torch.cuda.current_device()
        with torch.cuda.device(device):
            launcher = self._launcher(tuple(tensors))
            launcher(*tensors)


class ShapeSpecializingJITCompiler(JITCompiler):
    """Cache one static PTX launcher per concrete tensor signature."""

    def __init__(
        self,
        module: ModuleOp,
        compile_fn: Callable[[ModuleOp], PtxLauncher | ModuleOp],
    ) -> None:
        top_ops = module.body.ops
        if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
            raise ValueError("ShapeSpecializingJITCompiler requires exactly one kernel")
        kernel = top_ops[0]
        body_ops = kernel.body.ops
        if sum(isinstance(op, MatmulOp) for op in body_ops) != 1 or not all(
            isinstance(op, (MatmulOp, ConstantOp)) for op in body_ops
        ):
            raise ValueError("ShapeSpecializingJITCompiler requires exactly one matmul")
        if not all(isinstance(typ, GmemTensorType) for typ in kernel.signature):
            raise ValueError(
                "ShapeSpecializingJITCompiler currently supports only gmem arguments"
            )
        for typ in kernel.signature:
            assert isinstance(typ, GmemTensorType)
            if not typ.is_canonical_strides:
                raise ValueError(
                    "ShapeSpecializingJITCompiler requires canonical gmem layouts"
                )

        self._module = module
        self._compile_specialization = compile_fn
        super().__init__(module)

    def _prepare_module(
        self,
        module: ModuleOp,
        shapes: list[tuple[int, ...]],
    ) -> ModuleOp:
        specialize_shapes(module, shapes)
        return module

    def _compile(self, module: ModuleOp) -> PtxLauncher:
        result = self._compile_specialization(module)
        return super()._compile(result) if isinstance(result, ModuleOp) else result
