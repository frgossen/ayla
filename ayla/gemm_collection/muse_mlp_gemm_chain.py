"""Deferred grid-synchronized Muse SwiGLU/down-projection experiment."""
from __future__ import annotations

from ..base.ir import Block, Value
from ..gpu_dialect import (
    BF16Type,
    BarrierOp,
    CollectiveAxis,
    CollectiveLevel,
    F32Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
)
from ..passes.canonicalize import Canonicalize
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from .gemm import schedule_tiled_matmul
from .muse_swiglu_projection import (
    build_muse_swiglu_projection,
    schedule_muse_swiglu_projection,
)


_HIDDEN = 6_656
_INTERMEDIATE = 19_968
_PACKED_INTERMEDIATE = 2 * _INTERMEDIATE
_OUTPUT_M_TILE = 64
_RESIDENT_BLOCKS = 104


def _schedule_down_projection(
    module: ModuleOp,
    *,
    operand_stages: int,
) -> None:
    TileMatmul(
        op_dims=["M", "N"],
        tile_sizes=[_OUTPUT_M_TILE, 64],
        iteration_bindings=[
            None,
            (CollectiveLevel.BLOCK, CollectiveAxis.X),
        ],
        for_loop_labels=["down_m", None],
        op_label_filter="down",
        allow_partial_tiles=True,
    ).run(module)
    schedule_tiled_matmul(
        label="down",
        output_m_tile=_OUTPUT_M_TILE,
        output_n_tile=64,
        operand_stages=operand_stages,
        operand_k_tile=64,
        output_tma_inner_size=None,
        unsafe_reuse_smem=True,
        operand_move_attrs={
            "lhs": {
                "l2_cache_hint": "evict_last",
                "l2_promotion": "128B",
            },
            "rhs": {
                "l2_cache_hint": "evict_first",
                "l2_promotion": "256B",
            },
            "dst": {"l2_cache_hint": "evict_first"},
        },
        loop_label_prefix="down",
    ).run(module)


def muse_glimmer_swiglu_module(*, m: int) -> ModuleOp:
    """Build the existing one-wave SwiGLU-to-down-projection experiment."""
    if m == 32:
        down_operand_stages = 5
    elif m == 64:
        down_operand_stages = 6
    else:
        raise ValueError(f"unsupported M={m}; expected 32 or 64")

    bf16 = BF16Type()
    lhs = Value("lhs", GmemTensorType.build([m, _HIDDEN], bf16))
    packed_weight = Value(
        "packed_weight",
        GmemTensorType.build([_PACKED_INTERMEDIATE, _HIDDEN], bf16),
    )
    down_weight = Value(
        "down_weight",
        GmemTensorType.build([_HIDDEN, _INTERMEDIATE], bf16),
    )
    hidden = Value(
        "hidden", GmemTensorType.build([m, _INTERMEDIATE], bf16),
    )
    output = Value("output", GmemTensorType.build([m, _HIDDEN], bf16))
    sync = Value("sync", GmemTensorType.build([2], F32Type()))
    down = MatmulOp.build(
        hidden,
        down_weight,
        output,
        collective={axis: CollectiveLevel.GRID for axis in CollectiveAxis},
        init=0.0,
        attrs={"label": "down"},
    )
    projection_ops, projection_writeback = build_muse_swiglu_projection(
        lhs, packed_weight, hidden,
    )
    kernel = KernelOp.build(
        f"muse_glimmer_swiglu_m{m}",
        Block(
            arguments=[
                lhs, packed_weight, down_weight, hidden, output, sync,
            ],
            ops=[
                *projection_ops,
                BarrierOp.build("grid", sync=sync),
                down,
            ],
        ),
        grid_dim=(_RESIDENT_BLOCKS, 1, 1),
        block_dim=(128, 1, 1),
        num_warp_groups=1,
        ptx_optimization_level=4,
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    schedule_muse_swiglu_projection(module, projection_writeback)
    _schedule_down_projection(module, operand_stages=down_operand_stages)
    ScheduleAsync().run(module)
    return Canonicalize().run(module)
