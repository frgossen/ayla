"""FX-to-Ayla compilation for the Muse Glimmer SwiGLU MLP."""
from __future__ import annotations

from typing import cast

import torch

from ..fx_compiler import FxRoutePlan, make_fx_to_ayla_rewrite
from ..fx_pipeline import FxRegionMatch
from ..fx_schedules import compile_swiglu
from ..gpu_dialect import (
    BF16Type,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    Type,
)
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind
from .gemm import auto_matmul_4096_wgmma256_wg_tma_store_module


MUSE_SWIGLU_HIDDEN = 6_656
MUSE_SWIGLU_INTERMEDIATE = 19_968
MUSE_SWIGLU_PACKED_INTERMEDIATE = 2 * MUSE_SWIGLU_INTERMEDIATE
MUSE_SWIGLU_M_VALUES = (1, 4, 32, 64, 512, 2048)

_KEEP_LHS = {
    "lhs": {"l2_cache_hint": "evict_last", "l2_promotion": "256B"},
    "rhs": {"l2_cache_hint": "evict_first"},
    "dst": {"l2_cache_hint": "evict_first"},
}
_STREAM_INPUTS = {
    "lhs": {"l2_promotion": "256B"},
    "rhs": {"l2_promotion": "256B"},
    "dst": {"l2_cache_hint": "evict_first"},
}

_MUSE_SWIGLU_SIGNATURE: list[Type] = [
    GmemTensorType.build([None, MUSE_SWIGLU_HIDDEN], BF16Type()),
    GmemTensorType.build(
        [MUSE_SWIGLU_PACKED_INTERMEDIATE, MUSE_SWIGLU_HIDDEN],
        BF16Type(),
    ),
    GmemTensorType.build(
        [MUSE_SWIGLU_HIDDEN, MUSE_SWIGLU_INTERMEDIATE], BF16Type(),
    ),
    GmemTensorType.build([None, MUSE_SWIGLU_HIDDEN], BF16Type()),
]
_MUSE_SWIGLU_OPERANDS = [
    OperandKind.R,
    OperandKind.R,
    OperandKind.R,
    OperandKind.RESULT,
]


def _muse_m(match: FxRegionMatch) -> int:
    lhs = match.inputs[0].meta.get("val")
    if not isinstance(lhs, torch.Tensor):
        raise ValueError("Muse SwiGLU match has no lhs tensor metadata")
    return int(lhs.shape[0])


def _compile_large_m(m: int) -> FxRoutePlan:
    # These are the winners retained from the Muse cold-cache sweep, not
    # general-purpose FX compiler heuristics.
    if m == 512:
        projection_kernel = make_ptx_kernel(
            auto_matmul_4096_wgmma256_wg_tma_store_module(
                m=m,
                n=MUSE_SWIGLU_PACKED_INTERMEDIATE,
                k=MUSE_SWIGLU_HIDDEN,
                output_m_tile=128,
                output_n_tile=208,
                operand_k_tile=64,
                operand_stages=4,
                output_tma_inner_size=None,
                operand_move_attrs=_KEEP_LHS,
                ptx_optimization_level=3,
            ),
        )
        down_kernel = make_ptx_kernel(
            auto_matmul_4096_wgmma256_wg_tma_store_module(
                m=m,
                n=MUSE_SWIGLU_HIDDEN,
                k=MUSE_SWIGLU_INTERMEDIATE,
                output_m_tile=128,
                output_n_tile=208,
                operand_k_tile=64,
                operand_stages=4,
                output_tma_inner_size=None,
                operand_move_attrs=_STREAM_INPUTS,
                ptx_optimization_level=3,
            ),
        )
    elif m == 2048:
        projection_kernel = None
        down_kernel = make_ptx_kernel(
            auto_matmul_4096_wgmma256_wg_tma_store_module(
                m=m,
                n=MUSE_SWIGLU_HIDDEN,
                k=MUSE_SWIGLU_INTERMEDIATE,
                output_m_tile=192,
                output_n_tile=192,
                operand_k_tile=64,
                operand_stages=3,
                output_slices=3,
                output_tma_inner_size=64,
                operand_move_attrs=_STREAM_INPUTS,
                ptx_optimization_level=2,
            ),
        )
    else:
        raise ValueError(f"no large-M Muse SwiGLU route for M={m}")

    def build_route():
        down = make_ptx_launcher(down_kernel)

        def activate(projection: torch.Tensor) -> torch.Tensor:
            gate = projection[:, :MUSE_SWIGLU_INTERMEDIATE]
            up = projection[:, MUSE_SWIGLU_INTERMEDIATE:]
            return torch.nn.functional.silu(gate) * up

        if projection_kernel is None:
            def torch_projection(lhs, packed_weight):
                return activate(lhs @ packed_weight.T)

            projection_swiglu = torch.compile(
                torch_projection, fullgraph=True, dynamic=False,
            )
        else:
            compiled_activation = torch.compile(
                activate, fullgraph=True, dynamic=False,
            )
            projection = make_ptx_launcher(projection_kernel)
            projection_output = torch.empty(
                (m, MUSE_SWIGLU_PACKED_INTERMEDIATE),
                device="cuda",
                dtype=torch.bfloat16,
            )

            def projection_swiglu(lhs, packed_weight):
                projection(lhs, packed_weight, projection_output)
                return compiled_activation(projection_output)

        def run(lhs, packed_weight, down_weight, output):
            down(projection_swiglu(lhs, packed_weight), down_weight, output)

        return run

    return build_route


def _compile_muse_swiglu(module: ModuleOp) -> FxRoutePlan:
    [kernel] = module.body.ops
    if not isinstance(kernel, KernelOp):
        raise ValueError("Muse SwiGLU requires one kernel")
    lhs_type = kernel.body.arguments[0].type
    if not isinstance(lhs_type, GmemTensorType):
        raise ValueError("Muse SwiGLU lhs must be a gmem tensor")
    m = cast(int, lhs_type.sizes[0])
    if m <= 64:
        return compile_swiglu(module)
    return _compile_large_m(m)


MUSE_SWIGLU_RULE = make_fx_to_ayla_rewrite(
    name="muse_swiglu",
    signature=_MUSE_SWIGLU_SIGNATURE,
    signature_mapping=_MUSE_SWIGLU_OPERANDS,
    compile_region=_compile_muse_swiglu,
    accept=lambda match: _muse_m(match) in MUSE_SWIGLU_M_VALUES,
    infer_result_shapes=lambda lhs, _packed, _down: [
        (lhs.shape[0], MUSE_SWIGLU_HIDDEN),
    ],
)
