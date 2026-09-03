"""Fused small-M Muse MLP projection kernels."""
from __future__ import annotations

from ..base.ir import Block, Value
from ..gpu_dialect import (
    BF16Type,
    BarrierOp,
    ElementwiseOp,
    F32Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
)
from ..passes.canonicalize import Canonicalize
from ..passes.schedule_async import schedule_async
from .gemm import schedule_persistent_matmul


_HIDDEN = 6_656
_INTERMEDIATE = 19_968


def muse_mlp_gemm_chain_module(*, m: int) -> ModuleOp:
    """Build ``[M,6656] -> GEMM -> SiLU -> GEMM -> [M,6656]``.

    The intermediate width is 19,968.
    """
    if m not in (32, 64):
        raise ValueError(f"unsupported M={m}; expected 32 or 64")

    # This 64x64, K64, four-stage schedule has a 73 KiB footprint. Three CTAs
    # fit per H100 SM, so all 312 up-projection CTAs can reach the grid barrier.
    m_tile = 64
    n_tile = 64
    k_tile = 64
    operand_stages = 4
    resident_blocks = 312
    bf16 = BF16Type()
    a = Value("a", GmemTensorType.build([m, _HIDDEN], bf16))
    w_up = Value("w_up", GmemTensorType.build([_INTERMEDIATE, _HIDDEN], bf16))
    w_down = Value(
        "w_down", GmemTensorType.build([_HIDDEN, _INTERMEDIATE], bf16),
    )
    intermediate = Value(
        "intermediate", GmemTensorType.build([m, _INTERMEDIATE], bf16),
    )
    output = Value("output", GmemTensorType.build([m, _HIDDEN], bf16))
    sync = Value("sync", GmemTensorType.build([2], F32Type()))

    ops = [
        MatmulOp.build(
            a, w_up, intermediate, init=0.0, attrs={"label": "up"},
        ),
        ElementwiseOp.build(
            intermediate, intermediate, "silu",
            attrs={"label": "up_epi"},
        ),
        BarrierOp.build("grid", sync=sync),
        MatmulOp.build(
            intermediate, w_down, output, init=0.0,
            attrs={"label": "down"},
        ),
    ]

    kernel = KernelOp.build(
        f"muse_mlp_gemm_silu_gemm_m{m}",
        Block(
            arguments=[a, w_up, w_down, intermediate, output, sync],
            ops=ops,
        ),
        grid_dim=(resident_blocks, 1, 1),
        ptx_optimization_level=4,
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    module = schedule_persistent_matmul(
        module, label="up", epilogue="silu",
        m_tile=m_tile, n_tile=n_tile,
        resident_blocks=resident_blocks,
        operand_stages=operand_stages, operand_k_tile=k_tile,
        dst_cache_hint="evict_last",
    )
    module = schedule_persistent_matmul(
        module, label="down", epilogue=None,
        m_tile=m_tile, n_tile=n_tile,
        resident_blocks=resident_blocks,
        operand_stages=operand_stages, operand_k_tile=k_tile,
        dst_cache_hint="evict_first",
    )
    module = schedule_async(module)
    module = Canonicalize().run(module)
    return module
