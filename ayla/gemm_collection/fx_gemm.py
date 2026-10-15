"""FX-to-Ayla rewrites for the collection's fixed-size GEMM kernels."""
from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass

from ..fx_compiler import (
    FxRegionMatch,
    fx_region_to_module,
    make_fx_to_ayla_rewrite,
)
from ..gpu_dialect import (
    AllocOp,
    BF16Type,
    ConstantOp,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    Type,
)
from ..jit_compiler import LauncherPlan, PtxRoutePlan
from ..ptx_codegen import make_ptx_kernel
from ..torch_custom_call import OperandKind
from .auto_matmul_4096_wgmma256_wg import (
    auto_matmul_4096_wgmma256_wg_module,
)
from .auto_matmul_4096_wgmma256_wg_sigmoid import (
    auto_matmul_4096_wgmma256_wg_sigmoid_module,
)


_GEMM_4096_SIGNATURE: list[Type] = [
    GmemTensorType.build([4096, 4096], BF16Type()),
    GmemTensorType.build([4096, 4096], BF16Type()),
    GmemTensorType.build([4096, 4096], BF16Type()),
]
_GEMM_OPERANDS = [OperandKind.R, OperandKind.R, OperandKind.RESULT]


def _is_gemm_region(match: FxRegionMatch, *, sigmoid: bool) -> bool:
    try:
        module = fx_region_to_module(match, name="gemm_candidate")
    except ValueError:
        return False
    [kernel] = module.body.ops
    if not isinstance(kernel, KernelOp) or len(kernel.body.arguments) != 3:
        return False
    lhs, weight, output = kernel.body.arguments
    if not sigmoid:
        if len(kernel.body.ops) != 2:
            return False
        zero, matmul = kernel.body.ops
        return (
            isinstance(zero, ConstantOp)
            and isinstance(matmul, MatmulOp)
            and matmul.lhs is lhs
            and matmul.rhs is weight
            and matmul.dst is output
        )
    if len(kernel.body.ops) != 4:
        return False
    allocation, zero, matmul, epilogue = kernel.body.ops
    return (
        isinstance(allocation, AllocOp)
        and isinstance(zero, ConstantOp)
        and isinstance(matmul, MatmulOp)
        and isinstance(epilogue, ElementwiseOp)
        and epilogue.fn == "sigmoid"
        and matmul.lhs is lhs
        and matmul.rhs is weight
        and matmul.dst is allocation.result()
        and epilogue.src(0) is allocation.result()
        and epilogue.dst is output
    )


def _compile_gemm(_module: ModuleOp) -> LauncherPlan:
    return PtxRoutePlan(make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_module(),
    ))


def _compile_gemm_sigmoid(_module: ModuleOp) -> LauncherPlan:
    return PtxRoutePlan(make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_sigmoid_module(),
    ))


def register_gemm_fx_rules(patterns: PatternMatcherPass) -> None:
    """Register fixed-size GEMM and GEMM-sigmoid FX rewrites."""
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.sigmoid.default,
        name="gemm_sigmoid_4096",
        signature=_GEMM_4096_SIGNATURE,
        signature_mapping=_GEMM_OPERANDS,
        compile_region=_compile_gemm_sigmoid,
        accept=lambda match: _is_gemm_region(match, sigmoid=True),
        require_maximal=False,
    )
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.mm.default,
        name="gemm_4096",
        signature=_GEMM_4096_SIGNATURE,
        signature_mapping=_GEMM_OPERANDS,
        compile_region=_compile_gemm,
        accept=lambda match: _is_gemm_region(match, sigmoid=False),
        require_maximal=False,
    )
