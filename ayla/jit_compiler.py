"""Shape-specializing JIT compilation and launch dispatch."""
from __future__ import annotations

from collections.abc import Callable

import torch

from .base.ir import clone_op
from .base.pipeline import Pipeline
from .gpu_dialect import (
    CollectiveAxis,
    CollectiveLevel,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
)
from .passes.promote_buffers import PromoteBuffers
from .passes.reuse_buffers import ReuseBuffers
from .passes.specialize_shapes import specialize_shapes
from .passes.tile_reduce import (
    TileReduce,
)
from .passes.verify_gpu_dialect import VerifyGpuDialect
from .ptx_codegen import PtxLoweringPipeline, make_ptx_kernel_from_lowered
from .ptx_runtime import PtxLauncher, make_ptx_launcher
from .torch_custom_call import to_torch_dtype


class JITCompiler:
    """Lazily compile one module and cache its launcher."""

    def __init__(self, module: ModuleOp) -> None:
        self._module = module
        self._launcher: PtxLauncher | None = None

    def _compile_and_make_ptx_launcher(self) -> PtxLauncher:
        module = clone_op(self._module, {})
        assert isinstance(module, ModuleOp)
        pipeline = Pipeline()
        pipeline.add_pass(ReuseBuffers())
        pipeline.add_pass(TileReduce(
            tile_sizes={0: 1},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
            epilogue_label_filter="mean_epilogue",
        ))
        pipeline.add_pass(PromoteBuffers(
            target_type="smem",
            operands=["src"],
        ))
        pipeline.add_pass(PromoteBuffers(
            target_type="wrmem",
            operands=["dst"],
            layout="lane0",
            epilogue_label_filter="mean_epilogue",
        ))
        pipeline.add_pass(TileReduce(
            tile_sizes={1: 32},
            parallel_iteration_space=None,
            hero_label_filter="mean",
            epilogue_label_filter="mean_epilogue",
        ))
        pipeline.add_pass(PtxLoweringPipeline())
        pipeline.add_instrumentation_pass(VerifyGpuDialect())
        module = pipeline.run(module)
        return make_ptx_launcher(make_ptx_kernel_from_lowered(module))

    def _get_launcher(self) -> PtxLauncher:
        if self._launcher is None:
            self._launcher = self._compile_and_make_ptx_launcher()
        return self._launcher

    @property
    def name(self) -> str:
        return self._get_launcher().name

    @property
    def signature(self):
        return self._get_launcher().signature

    def __call__(self, *tensors: torch.Tensor) -> None:
        self._get_launcher()(*tensors)


class ShapeSpecializingJITCompiler:
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
        if len(kernel.body.ops) != 1 or not isinstance(
            kernel.body.ops[0], MatmulOp,
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
        self._compile_fn = compile_fn
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
            assert isinstance(typ, GmemTensorType)
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
        module = clone_op(self._module, {})
        assert isinstance(module, ModuleOp)
        specialize_shapes(module, shapes)
        launcher = self._compile_fn(module)
        self._cache[key] = launcher
        return launcher

    def __call__(self, *tensors: torch.Tensor) -> None:
        launcher = self._launcher(tuple(tensors))
        launcher(*tensors)
