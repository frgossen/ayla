from importlib import import_module
from typing import TYPE_CHECKING

from .gpu_dialect import (
    AsyncOp,
    CollectiveAxis,
    CollectiveLevel,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    PTXTemplateOp,
)
from .base.ir import (
    Block,
    Op,
    Printer,
    Type,
    Value,
)


# Static tools need the real types of lazily exported compiler entry points.
# This condition is false at runtime; it does not control generated kernels.
if TYPE_CHECKING:
    from .ptx_codegen import PtxPrinter, PtxKernel, make_ptx_kernel
    from .jit_compiler import JITCompiler, ShapeSpecializingJITCompiler
    from .gemm_jit import (
        GemmScheduleSeed,
        apply_gemm_schedule_seed_post_build,
        select_gemm_schedule,
    )


# Importing the dialect must not pull in a backend or CUDA runtime. Keep the
# existing convenience API, loading compiler entry points when requested.
_COMPILER_EXPORTS = {
    'PtxPrinter': 'ptx_codegen',
    'PtxKernel': 'ptx_codegen',
    'make_ptx_kernel': 'ptx_codegen',
    'JITCompiler': 'jit_compiler',
    'ShapeSpecializingJITCompiler': 'jit_compiler',
    'GemmScheduleSeed': 'gemm_jit',
    'apply_gemm_schedule_seed_post_build': 'gemm_jit',
    'select_gemm_schedule': 'gemm_jit',
}


def __getattr__(name: str):
    module = _COMPILER_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AsyncOp",
    "CollectiveAxis",
    "CollectiveLevel",
    "Block",
    "GmemTensorType",
    "KernelOp",
    "MatmulOp",
    "ModuleOp",
    "Op",
    "PTXTemplateOp",
    "Printer",
    "Type",
    "Value",
    "PtxPrinter",
    "PtxKernel",
    "make_ptx_kernel",
    "JITCompiler",
    "ShapeSpecializingJITCompiler",
    "select_gemm_schedule",
    "GemmScheduleSeed",
    "apply_gemm_schedule_seed_post_build",
]
