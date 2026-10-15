"""JIT compilation, tensor specialization, and launch dispatch."""
from __future__ import annotations

from collections.abc import Callable
import torch

from .base.ir import clone_op
from .gpu_dialect import (
    ConstantOp, GmemTensorType, KernelOp, MatmulOp, ModuleOp, Type,
)
from .passes.specialize_shapes import specialize_shapes
from .ptx_codegen import PtxKernel
from .ptx_runtime import PtxLauncher, make_ptx_launcher
from .torch_custom_call import to_torch_dtype


TensorSignature = tuple[
    tuple[tuple[int, ...], tuple[int, ...], torch.dtype], ...
]
Launcher = Callable[..., None]
LauncherPlan = Callable[[], Launcher]


class JITCompiler:
    """Compile and dispatch launchers for concrete tensor signatures."""

    def __init__(
        self,
        *,
        name: str,
        signature: list[Type],
        compile_fn: Callable[[ModuleOp], LauncherPlan],
    ) -> None:
        self.name = name
        self.signature = signature
        self._compile_fn = compile_fn
        self._plans: dict[TensorSignature, LauncherPlan] = {}
        self._cache: dict[
            tuple[torch.device, TensorSignature], Launcher
        ] = {}

    def _metadata(
        self, tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.device, TensorSignature]:
        if not tensors:
            raise ValueError(f"{self.name}: shape dispatch requires a tensor")
        device = tensors[0].device
        if device.type != "cuda":
            raise ValueError(f"{self.name}: expected CUDA tensors")
        if any(tensor.device != device for tensor in tensors[1:]):
            raise ValueError(f"{self.name}: all tensors must be on one CUDA device")
        signature = tuple(
            (
                tuple(int(size) for size in tensor.shape),
                tuple(int(stride) for stride in tensor.stride()),
                tensor.dtype,
            )
            for tensor in tensors
        )
        return device, signature

    def prepare_module(
        self, signature: TensorSignature, module: ModuleOp,
    ) -> None:
        """Compile one specialization before its first runtime invocation."""
        if signature not in self._plans:
            self._plans[signature] = self._compile_fn(module)

    def _prepare_signature(self, signature: TensorSignature) -> None:
        raise RuntimeError(
            f"{self.name}: no compiled specialization for tensor signature"
        )

    def _launcher(self, tensors: tuple[torch.Tensor, ...]) -> Launcher:
        device, signature = self._metadata(tensors)
        key = (device, signature)
        launcher = self._cache.get(key)
        if launcher is not None:
            return launcher
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{self.name}: route cache miss during CUDA graph capture; "
                "invoke the kernel once before capture"
            )
        if signature not in self._plans:
            self._prepare_signature(signature)
        plan = self._plans.get(signature)
        if plan is None:
            raise RuntimeError(
                f"{self.name}: no compiled specialization for tensor signature"
            )
        with torch.cuda.device(device):
            launcher = plan()
        self._cache[key] = launcher
        return launcher

    def __call__(self, *tensors: torch.Tensor) -> None:
        self._launcher(tuple(tensors))(*tensors)


class PtxRoutePlan:
    """A single compiled PTX kernel route."""

    def __init__(self, kernel: PtxKernel) -> None:
        self.kernel = kernel

    def __call__(self) -> Launcher:
        return make_ptx_launcher(self.kernel)


class ShapeSpecializingJITCompiler(JITCompiler):
    """Cache one static PTX launcher per concrete tensor signature."""

    def __init__(
        self,
        module: ModuleOp,
        compile_fn: Callable[[ModuleOp], PtxLauncher],
    ) -> None:
        top_ops = module.body.ops
        if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
            raise ValueError(
                "ShapeSpecializingJITCompiler requires exactly one kernel"
            )
        kernel = top_ops[0]
        body_ops = kernel.body.ops
        if sum(isinstance(op, MatmulOp) for op in body_ops) != 1 or not all(
            isinstance(op, (MatmulOp, ConstantOp)) for op in body_ops
        ):
            raise ValueError(
                "ShapeSpecializingJITCompiler requires exactly one matmul"
            )
        if not all(
            isinstance(typ, GmemTensorType) for typ in kernel.signature
        ):
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
        super().__init__(
            name=kernel.name,
            signature=kernel.signature,
            compile_fn=self._compile_module,
        )

    def _metadata(
        self, tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.device, TensorSignature]:
        if len(tensors) != len(self.signature):
            raise ValueError(
                f"{self.name}: expected {len(self.signature)} tensors, "
                f"got {len(tensors)}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError(f"{self.name}: all tensors must be contiguous")
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors[1:]):
            raise ValueError(f"{self.name}: all tensors must be on one CUDA device")
        for tensor, typ in zip(tensors, self.signature, strict=True):
            assert isinstance(typ, GmemTensorType)
            if tensor.dtype != to_torch_dtype(typ.dtype):
                raise ValueError(
                    f"{self.name}: expected dtype {typ.dtype}, got {tensor.dtype}"
                )

        return super()._metadata(tensors)

    def _compile_module(self, module: ModuleOp) -> LauncherPlan:
        launcher = self._compile_specialization(module)
        return lambda: launcher

    def _prepare_signature(self, signature: TensorSignature) -> None:
        shapes = [shape for shape, _strides, _dtype in signature]
        module = clone_op(self._module, {})
        assert isinstance(module, ModuleOp)
        specialize_shapes(module, shapes)
        self.prepare_module(signature, module)
