"""JIT compilation, tensor specialization, and launch dispatch."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import cast

import torch

from .base.ir import clone_op
from .gpu_dialect import ConstantOp, GmemTensorType, KernelOp, MatmulOp, ModuleOp, Type, TmaDescType
from .passes.specialize_shapes import specialize_shapes
from .ptx_codegen import PtxKernel, make_ptx_kernel
from .ptx_runtime import PtxLauncher, make_ptx_launcher
from .torch_custom_call import to_torch_dtype


TensorSignature = tuple[
    tuple[tuple[int, ...], tuple[int, ...], torch.dtype], ...
]
Launcher = Callable[..., None]
LauncherPlan = Callable[[], Launcher]


class _RegionJITCompiler:
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


class ScratchInit(Enum):
    EMPTY = auto()
    ZERO = auto()


class PtxRoutePlan:
    """A PTX kernel and the runtime source of each gmem argument."""

    def __init__(
        self,
        kernel: PtxKernel,
        bindings: Iterable[int | ScratchInit] | None = None,
    ) -> None:
        self.kernel = kernel
        gmem_types = [
            typ for typ in kernel.signature
            if isinstance(typ, GmemTensorType)
        ]
        self.bindings = tuple(range(len(gmem_types))) if bindings is None else (
            tuple(bindings)
        )
        if len(self.bindings) != len(gmem_types):
            raise ValueError(
                f"{kernel.name}: expected {len(gmem_types)} argument "
                f"bindings, got {len(self.bindings)}"
            )

    def __call__(self) -> Launcher:
        launcher = make_ptx_launcher(self.kernel)
        arguments: list[int | torch.Tensor] = []
        gmem_types = (
            typ for typ in self.kernel.signature
            if isinstance(typ, GmemTensorType)
        )
        for typ, binding in zip(gmem_types, self.bindings, strict=True):
            if type(binding) is int:
                if binding < 0:
                    raise ValueError("runtime tensor index must be non-negative")
                arguments.append(binding)
                continue
            if any(size is None for size in typ.sizes):
                raise ValueError(
                    f"{self.kernel.name}: scratch buffer shape must be static"
                )
            sizes = cast(tuple[int, ...], typ.sizes)
            allocate = (
                torch.zeros if binding is ScratchInit.ZERO else torch.empty
            )
            arguments.append(allocate(
                sizes,
                dtype=to_torch_dtype(typ.dtype),
                device="cuda",
            ))

        def run(*tensors: torch.Tensor) -> None:
            launcher(*(
                tensors[arg] if type(arg) is int else arg
                for arg in arguments
            ))

        return run


class JITCompiler(_RegionJITCompiler):
    """Compile a static kernel module lazily and cache launchers per device.

    The module must already have its scheduling and launch dimensions assigned.
    PTX lowering and launcher construction happen on the first invocation.
    Tensor metadata is available before compilation for Torch registration.
    """

    def __init__(self, module: ModuleOp | None = None, **route_options) -> None:
        self._module_mode = module is not None
        if module is None:
            super().__init__(**route_options)
            return
        top_ops = module.body.ops
        if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
            raise ValueError("JITCompiler requires exactly one kernel")
        if not all(
            isinstance(typ, (GmemTensorType, TmaDescType))
            for typ in top_ops[0].signature
        ):
            raise ValueError("JITCompiler supports only gmem and TMA arguments")
        self._module = module
        self._cache: dict[tuple[object, ...], PtxLauncher] = {}

    @property
    def _kernel(self) -> KernelOp:
        [kernel] = self._module.body.ops
        assert isinstance(kernel, KernelOp)
        return kernel

    @property
    def name(self) -> str:
        return self._kernel.name

    @property
    def signature(self) -> list[Type]:
        # TMA descriptors are synthesized by the launcher, not supplied by Torch.
        return [
            typ for typ in self._kernel.signature
            if isinstance(typ, GmemTensorType)
        ]

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _metadata(
        self, tensors: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[object, ...], list[tuple[int, ...]]]:
        if not self._module_mode:
            return super()._metadata(tensors)
        if len(tensors) != len(self.signature):
            raise ValueError(
                f"{self.name}: expected {len(self.signature)} tensors, "
                f"got {len(tensors)}"
            )
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError(f"{self.name}: all tensors must be CUDA tensors")
        device = (
            tensors[0].device if tensors
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

        shapes = [tuple(tensor.shape) for tensor in tensors]
        return (device, tuple(shapes)), shapes

    def _prepare_module(
        self, module: ModuleOp, shapes: list[tuple[int, ...]],
    ) -> ModuleOp:
        if any(
            isinstance(typ, GmemTensorType) and typ.is_dynamic
            for typ in self.signature
        ):
            raise ValueError(
                "JITCompiler requires static shapes; use ShapeSpecializingJITCompiler"
            )
        return module

    def _compile(self, module: ModuleOp) -> PtxLauncher:
        launcher = make_ptx_launcher(make_ptx_kernel(module))
        launcher.ensure_loaded()
        return launcher

    def _launcher(self, tensors: tuple[torch.Tensor, ...]) -> PtxLauncher:
        if not self._module_mode:
            return super()._launcher(tensors)
        key, shapes = self._metadata(tensors)
        if isinstance(key, torch.device):
            metadata = shapes
            key, shapes = (key, metadata), [entry[0] for entry in metadata]
        launcher = self._cache.get(key)
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
        self._cache[key] = launcher
        return launcher

    def __call__(self, *tensors: torch.Tensor) -> None:
        if not self._module_mode:
            return super().__call__(*tensors)
        device = tensors[0].device if tensors else torch.cuda.current_device()
        with torch.cuda.device(device):
            launcher = self._launcher(tuple(tensors))
            launcher(*tensors)


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
        result = self._compile_specialization(module)
        launcher = self._compile(result) if isinstance(result, ModuleOp) else result
        return lambda: launcher

    def _prepare_signature(self, signature: TensorSignature) -> None:
        shapes = [shape for shape, _strides, _dtype in signature]
        module = clone_op(self._module, {})
        assert isinstance(module, ModuleOp)
        specialize_shapes(module, shapes)
        self.prepare_module(signature, module)
