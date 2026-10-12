"""FX-to-Ayla rewrites for BF16 GEMM kernels."""
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
from .auto_matmul_4096_wgmma256_wg_sigmoid import (
    auto_matmul_4096_wgmma256_wg_sigmoid_module,
)
from .dynamic_m import _dynamic_m_schedule_heuristic
from .gemm import schedule_matmul


_GEMM_SIGNATURE: list[Type] = [
    GmemTensorType.build([None, None], BF16Type()),
    GmemTensorType.build([None, None], BF16Type()),
    GmemTensorType.build([None, None], BF16Type()),
]
_GEMM_SIGMOID_4096_SIGNATURE: list[Type] = [
    GmemTensorType.build([4096, 4096], BF16Type()),
    GmemTensorType.build([4096, 4096], BF16Type()),
    GmemTensorType.build([4096, 4096], BF16Type()),
]
_GEMM_OPERANDS = [OperandKind.R, OperandKind.R, OperandKind.RESULT]
_ADDMM_SIGNATURE: list[Type] = [
    GmemTensorType.build([None], BF16Type()),
    *_GEMM_SIGNATURE,
]
_ADDMM_OPERANDS = [OperandKind.R, *_GEMM_OPERANDS]


def _static_gemm_shape(op: MatmulOp) -> tuple[int, int, int] | None:
    lhs = op.lhs.type
    rhs = op.rhs.type
    dst = op.dst.type
    if not (
        isinstance(lhs, GmemTensorType)
        and isinstance(rhs, GmemTensorType)
        and isinstance(dst, GmemTensorType)
    ):
        return None
    if lhs.rank != 2 or rhs.rank != 2 or dst.rank != 2:
        return None
    m, k = lhs.sizes
    n, rhs_k = rhs.sizes
    if not (
        isinstance(m, int)
        and isinstance(n, int)
        and isinstance(k, int)
        and isinstance(rhs_k, int)
    ):
        return None
    if min(m, n, k) <= 0 or rhs_k != k or tuple(dst.sizes) != (m, n):
        return None
    return m, n, k


def _gemm(module: ModuleOp) -> MatmulOp | None:
    """The matmul of a kernel that is one GEMM writing its output."""
    [kernel] = module.body.ops
    assert isinstance(kernel, KernelOp)
    *initializers, matmul = kernel.body.ops
    if not (
        isinstance(matmul, MatmulOp)
        and matmul.dst is kernel.body.arguments[-1]
        and all(isinstance(op, ConstantOp) for op in initializers)
    ):
        return None
    return matmul


def _is_gemm_region(match: FxRegionMatch, *, sigmoid: bool) -> bool:
    try:
        module = fx_region_to_module(match, name="gemm_candidate")
    except ValueError:
        return False
    if not sigmoid:
        matmul = _gemm(module)
        shape = None if matmul is None else _static_gemm_shape(matmul)
        return shape is not None and shape[2] % 64 == 0
    [kernel] = module.body.ops
    if not isinstance(kernel, KernelOp) or len(kernel.body.arguments) != 3:
        return False
    lhs, weight, output = kernel.body.arguments
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


def _compile_gemm(module: ModuleOp) -> LauncherPlan:
    matmul = _gemm(module)
    assert matmul is not None
    shape = _static_gemm_shape(matmul)
    assert shape is not None and shape[2] % 64 == 0
    m, n, k = shape
    schedule_matmul(
        module, **_dynamic_m_schedule_heuristic(m, n, k),
    )
    return PtxRoutePlan(make_ptx_kernel(module))


def _compile_gemm_sigmoid(_module: ModuleOp) -> LauncherPlan:
    return PtxRoutePlan(make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_sigmoid_module(),
    ))


def register_gemm_fx_rules(patterns: PatternMatcherPass) -> None:
    """Register plain GEMM, addmm, and fixed-size GEMM-sigmoid FX rewrites."""
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.sigmoid.default,
        name="gemm_sigmoid_4096",
        signature=_GEMM_SIGMOID_4096_SIGNATURE,
        signature_mapping=_GEMM_OPERANDS,
        compile_region=_compile_gemm_sigmoid,
        accept=lambda match: _is_gemm_region(match, sigmoid=True),
        require_maximal=False,
    )
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.mm.default,
        name="gemm",
        signature=_GEMM_SIGNATURE,
        signature_mapping=_GEMM_OPERANDS,
        compile_region=_compile_gemm,
        accept=lambda match: _is_gemm_region(match, sigmoid=False),
        require_maximal=False,
        infer_result_shapes=lambda lhs, weight: [
            (lhs.shape[0], weight.shape[0]),
        ],
    )
    make_fx_to_ayla_rewrite(
        patterns,
        root=torch.ops.aten.addmm.default,
        name="addmm",
        signature=_ADDMM_SIGNATURE,
        signature_mapping=_ADDMM_OPERANDS,
        compile_region=_compile_gemm,
        accept=lambda match: _is_gemm_region(match, sigmoid=False),
        require_maximal=False,
        infer_result_shapes=lambda _bias, lhs, weight: [
            (lhs.shape[0], weight.shape[0]),
        ],
    )
