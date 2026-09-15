"""Construction of a register-resident SwiGLU projection pair."""
from __future__ import annotations

from ..base.ir import Op, Value
from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    BarrierOp,
    BF16Type,
    BlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F16Type,
    F32Type,
    IndexConstantOp,
    MatmulOp,
    MoveOp,
    SmemTensorType,
    WrmemTensorType,
)
from ..passes.tile_matmul import build_tile_ops


# TODO: Generalize TileMatmul and PromoteBuffers to jointly handle explicitly
# selected Matmul heroes and their multi-source elementwise dataflow. That
# transform should share tile coordinates, keep each accumulator in WRMEM, and
# write final live-outs through the standard conversion/SMEM/TMA path. Then
# remove this helper and the one-M-tile/one-N-tile-per-block restriction below.
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
        SmemTensorType.build(list(src.type.sizes), dst.type.dtype),
        attrs={"reuse_group": f"{label}_writeback"},
    )
    return [
        packed,
        staging,
        ElementwiseOp.build(packed.result(), [src], "convert"),
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
    tiles_m = (dst.type.sizes[0] + m_tile - 1) // m_tile
    tiles_n = (dst.type.sizes[1] + n_tile - 1) // n_tile
    if tiles_m != 1 or tiles_n != resident_blocks:
        raise ValueError(
            "SwiGLU projection requires one M tile and one N tile per block"
        )

    block_ids = BlockIdsOp.build()
    tile_m = IndexConstantOp.build(0)
    indices = [tile_m.result(), block_ids.x]
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
    collective = {
        CollectiveAxis.X: CollectiveLevel.WG,
        CollectiveAxis.Y: CollectiveLevel.GRID,
        CollectiveAxis.Z: CollectiveLevel.GRID,
    }
    projection_ops = [
        MatmulOp.build(
            tiled_lhs, tiled_gate, gate.result(),
            collective=collective, attrs={"label": "gate"},
        ),
        # Preserve the two sequential shared-memory lifetimes while the async
        # scheduler moves loads. A thread barrier lowers to no PTX.
        BarrierOp.build("thread"),
        MatmulOp.build(
            tiled_lhs, tiled_up, up.result(),
            collective=collective, attrs={"label": "up"},
        ),
        # The up WGMMA await can sink past this independent operation,
        # overlapping its tail with activation on capable targets.
        ElementwiseOp.build(
            gate.result(), [gate.result()], "silu",
            attrs={"label": "gate_epi"},
        ),
        ElementwiseOp.build(
            up.result(), [up.result(), gate.result()], "mul",
            attrs={"label": "up_epi"},
        ),
        *_store_wgmma_fragment(tiled_dst, up.result(), label="up"),
    ]
    return [
        block_ids, tile_m,
        *lhs_ops, *gate_ops, *up_ops, *dst_ops,
        gate, up, *projection_ops,
    ]
