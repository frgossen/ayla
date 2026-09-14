"""Fused small-M Muse MLP projection kernels."""
from __future__ import annotations

from ..base.ir import Block, Op, Value
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
from .gemm import schedule_persistent_matmul, schedule_tiled_matmul
from .split_k import build_persistent_split_k_matmul
from .swiglu_projection import build_persistent_swiglu_projection


_HIDDEN = 6_656
_INTERMEDIATE = 19_968
_OUTPUT_M_TILE = 64
_OUTPUT_N_TILE = 64
_K_TILE = 64
_RESIDENT_BLOCKS = 312
_SWIGLU_OPERAND_STAGES = 3


def _build_down_projection(
    *,
    lhs: Value,
    rhs: Value,
    dst: Value,
    split_k: int | None,
    n_tile: int,
    label: str = "down",
) -> list[Op]:
    if split_k is None:
        return [MatmulOp.build(
            lhs, rhs, dst, init=0.0, attrs={"label": label},
        )]
    return build_persistent_split_k_matmul(
        resident_blocks=_RESIDENT_BLOCKS,
        lhs=lhs,
        rhs=rhs,
        dst=dst,
        m_tile=_OUTPUT_M_TILE,
        n_tile=n_tile,
        split_k=split_k,
        label=label,
    )


def muse_mlp_gemm_chain_module(
    *,
    m: int,
    split_k: int | None = None,
    down_n_tile: int | None = None,
    down_operand_stages: int | None = None,
) -> ModuleOp:
    """Build ``[M,6656] -> GEMM -> SiLU -> GEMM -> [M,6656]``.

    The intermediate width is 19,968.
    """
    if m not in (32, 64):
        raise ValueError(f"unsupported M={m}; expected 32 or 64")
    down_n_tile = 64 if down_n_tile is None else down_n_tile
    down_operand_stages = (
        4 if down_operand_stages is None else down_operand_stages
    )

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
    down = _build_down_projection(
        lhs=intermediate,
        rhs=w_down,
        dst=output,
        split_k=split_k,
        n_tile=down_n_tile,
    )

    ops = [
        MatmulOp.build(
            a, w_up, intermediate, init=0.0, attrs={"label": "up"},
        ),
        ElementwiseOp.build(
            intermediate, intermediate, "silu",
            attrs={"label": "up_epi"},
        ),
        BarrierOp.build("grid", sync=sync),
        *down,
    ]
    kernel = KernelOp.build(
        f"muse_mlp_gemm_silu_gemm_m{m}",
        Block(
            arguments=[a, w_up, w_down, intermediate, output, sync],
            ops=ops,
        ),
        grid_dim=(resident_blocks, 1, 1),
        block_dim=(128, 1, 1) if split_k is not None else (None, None, None),
        num_warp_groups=1 if split_k is not None else None,
        cluster_dim=(split_k, 1, 1) if split_k is not None else None,
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
        m_tile=m_tile, n_tile=down_n_tile,
        resident_blocks=resident_blocks,
        operand_stages=down_operand_stages, operand_k_tile=k_tile,
        dst_cache_hint="evict_first",
    )
    module = schedule_async(module)
    module = Canonicalize().run(module)
    return module


def muse_glimmer_swiglu_module(
    *,
    m: int,
    split_k: int | None = None,
    down_n_tile: int | None = None,
    down_operand_stages: int | None = None,
) -> ModuleOp:
    """Build the dual-projection Muse Glimmer SwiGLU chain.

    The two projection accumulators remain in registers until SiLU and their
    elementwise product have completed, so only the final hidden activation is
    materialized before the down projection.
    """
    if m not in (32, 64):
        raise ValueError(f"unsupported M={m}; expected 32 or 64")
    if split_k is None:
        default_n_tile, default_stages = ((48, 5) if m == 32 else (32, 6))
    else:
        default_n_tile, default_stages = 64, 4
    down_n_tile = default_n_tile if down_n_tile is None else down_n_tile
    down_operand_stages = (
        default_stages
        if down_operand_stages is None else down_operand_stages
    )

    bf16 = BF16Type()
    a = Value("a", GmemTensorType.build([m, _HIDDEN], bf16))
    w_gate = Value(
        "w_gate", GmemTensorType.build([_INTERMEDIATE, _HIDDEN], bf16),
    )
    w_up = Value(
        "w_up", GmemTensorType.build([_INTERMEDIATE, _HIDDEN], bf16),
    )
    w_down = Value(
        "w_down", GmemTensorType.build([_HIDDEN, _INTERMEDIATE], bf16),
    )
    hidden = Value("hidden", GmemTensorType.build([m, _INTERMEDIATE], bf16))
    output = Value("output", GmemTensorType.build([m, _HIDDEN], bf16))
    sync = Value("sync", GmemTensorType.build([2], F32Type()))

    projection = build_persistent_swiglu_projection(
        resident_blocks=_RESIDENT_BLOCKS,
        lhs=a,
        gate_rhs=w_gate,
        up_rhs=w_up,
        dst=hidden,
        m_tile=_OUTPUT_M_TILE,
        n_tile=_OUTPUT_N_TILE,
    )
    grid_barrier = BarrierOp.build("grid", sync=sync)
    down = _build_down_projection(
        lhs=hidden,
        rhs=w_down,
        dst=output,
        split_k=split_k,
        n_tile=down_n_tile,
    )
    kernel = KernelOp.build(
        f"muse_glimmer_swiglu_m{m}",
        Block(
            arguments=[a, w_gate, w_up, w_down, hidden, output, sync],
            ops=[*projection, grid_barrier, *down],
        ),
        grid_dim=(_RESIDENT_BLOCKS, 1, 1),
        block_dim=(128, 1, 1) if split_k is not None else (None, None, None),
        num_warp_groups=1 if split_k is not None else None,
        cluster_dim=(split_k, 1, 1) if split_k is not None else None,
        ptx_optimization_level=4,
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    projection_move_attrs = {
        "lhs": {
            "l2_cache_hint": "evict_last",
            "l2_promotion": "128B",
        },
        "rhs": {
            "l2_cache_hint": "evict_first",
            "l2_promotion": "256B",
        },
        "dst": {"l2_cache_hint": "evict_last"},
    }
    for label in ("gate", "up"):
        # The projection builder already performed the block-grid and
        # warp-group tiling and allocated register-resident accumulators.
        module = schedule_tiled_matmul(
            module,
            label=label,
            output_m_tile=_OUTPUT_M_TILE,
            output_n_tile=_OUTPUT_N_TILE,
            operand_stages=_SWIGLU_OPERAND_STAGES,
            operand_k_tile=_K_TILE,
            output_tma_inner_size=None,
            unsafe_reuse_smem=True,
            operand_move_attrs=projection_move_attrs,
            loop_label_prefix=label,
        )
    module = schedule_persistent_matmul(
        module,
        label="down",
        m_tile=_OUTPUT_M_TILE,
        n_tile=down_n_tile,
        resident_blocks=_RESIDENT_BLOCKS,
        operand_stages=down_operand_stages,
        operand_k_tile=_K_TILE,
        dst_cache_hint="evict_first",
    )
    module = schedule_async(module)
    module = Canonicalize().run(module)
    return module
