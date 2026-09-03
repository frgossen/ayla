"""Construction of a register-resident SwiGLU projection pair."""
from __future__ import annotations

import math

from ..base.ir import Op, Value
from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    BarrierOp,
    BF16Type,
    ElementwiseOp,
    F16Type,
    F32Type,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    MatmulOp,
    MoveOp,
    SmemTensorType,
    WrmemTensorType,
)
from ..passes.tile_matmul import build_persistent_grid, build_tile_ops


def _store_wgmma_fragment(dst: Value, src: Value, *, label: str) -> list[Op]:
    """Convert one f32 accumulator fragment and issue its TMA writeback."""
    if not isinstance(src.type, WrmemTensorType):
        raise ValueError("fragment writeback requires wrmem source")
    if not isinstance(dst.type, AddrMemTensorType):
        raise ValueError("fragment writeback requires addressable destination")
    if tuple(src.type.sizes) != tuple(dst.type.sizes):
        raise ValueError("fragment and destination must have one shape")
    if not isinstance(src.type.dtype, F32Type):
        raise ValueError("fragment writeback requires f32 source")
    if not isinstance(dst.type.dtype, (BF16Type, F16Type)):
        raise ValueError("fragment writeback requires bf16 or f16 destination")
    packed = AllocOp.build_with_fresh_id(
        WrmemTensorType.build(
            list(src.type.sizes), dst.type.dtype, warps=src.type.warps,
            layout=src.type.layout,
        ),
    )
    staging = AllocOp.build_with_fresh_id(
        SmemTensorType.build(
            list(src.type.sizes), dst.type.dtype, swizzle="128B",
            attrs={
                "physical_layout": {
                    "block_width": 64,
                    "order": "block_major",
                },
            },
        ), attrs={"reuse_group": f"{label}_writeback"},
    )
    return [
        packed,
        staging,
        ElementwiseOp.build(packed.result(), src, "convert"),
        MoveOp.build(
            staging.result(), packed.result(),
            attrs={"label": f"{label}_dst_smem"},
        ),
        MoveOp.build(
            dst, staging.result(),
            attrs={
                "label": f"{label}_dst_tma", "tma": True,
                "l2_cache_hint": "evict_last",
            },
        ),
    ]


def build_persistent_swiglu_projection(
    *,
    resident_blocks: int,
    lhs: Value,
    gate_rhs: Value,
    up_rhs: Value,
    dst: Value,
    m_tile: int,
    n_tile: int,
) -> list[Op]:
    """Build two projections whose paired output tiles stay in registers.

    The two labeled matmuls are scheduled independently. The gate activation
    follows the up projection, allowing its WGMMA await to sink past that
    independent work.
    """
    tiles_m = math.ceil(dst.type.sizes[0] / m_tile)
    tiles_n = math.ceil(dst.type.sizes[1] / n_tile)

    def build_tile(linear: Value):
        c_tiles_n = IndexConstantOp.build(tiles_n)
        tile_m = IndexDivOp.build(linear, c_tiles_n.result())
        tile_n = IndexModOp.build(linear, c_tiles_n.result())
        indices = [tile_m.result(), tile_n.result()]
        lhs_ops, tiled_lhs = build_tile_ops(
            lhs, ["M", "K"], op_dims=["M", "N"],
            tile_idxs=indices, tile_sizes=[m_tile, n_tile],
        )
        gate_ops, tiled_gate = build_tile_ops(
            gate_rhs, ["N", "K"], op_dims=["M", "N"],
            tile_idxs=indices, tile_sizes=[m_tile, n_tile],
        )
        up_ops, tiled_up = build_tile_ops(
            up_rhs, ["N", "K"], op_dims=["M", "N"],
            tile_idxs=indices, tile_sizes=[m_tile, n_tile],
        )
        dst_ops, tiled_dst = build_tile_ops(
            dst, ["M", "N"], op_dims=["M", "N"],
            tile_idxs=indices, tile_sizes=[m_tile, n_tile],
        )
        accumulator_type = WrmemTensorType.build(
            [m_tile, n_tile], F32Type(), warps=4,
            layout="wgmma_layout_0",
        )
        gate = AllocOp.build_with_fresh_id(accumulator_type, init=0.0)
        up = AllocOp.build_with_fresh_id(accumulator_type, init=0.0)
        projection_ops = [
            MatmulOp.build(
                tiled_lhs, tiled_gate, gate.result(),
                attrs={
                    "label": "gate", "tiled": ["block.x", "wg.x"],
                },
            ),
            # Preserve the two sequential shared-memory lifetimes while the
            # async scheduler moves loads. A thread barrier lowers to no PTX.
            BarrierOp.build("thread"),
            MatmulOp.build(
                tiled_lhs, tiled_up, up.result(),
                attrs={
                    "label": "up", "tiled": ["block.x", "wg.x"],
                },
            ),
            # The up WGMMA await can sink past this independent operation,
            # overlapping its tail with activation on capable targets.
            ElementwiseOp.build(
                gate.result(), gate.result(), "silu",
                attrs={"label": "gate_epi"},
            ),
            ElementwiseOp.build(
                up.result(), up.result(), "mul", rhs=gate.result(),
                attrs={"label": "up_epi"},
            ),
            *_store_wgmma_fragment(tiled_dst, up.result(), label="up"),
        ]
        return [
            c_tiles_n, tile_m, tile_n,
            *lhs_ops, *gate_ops, *up_ops, *dst_ops,
            gate, up, *projection_ops,
        ]

    return build_persistent_grid(
        logical_tiles=tiles_m * tiles_n,
        resident_blocks=resident_blocks,
        build_tile=build_tile,
    )
