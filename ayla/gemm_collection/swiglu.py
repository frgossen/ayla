"""Import matching, scheduling, and compilation for packed SwiGLU."""

from __future__ import annotations

import os
from typing import cast

from ..base.ir import Block, Op, Value, clone_type, find_ops_of_type
from ..fx_compiler import (
    FxRegionMatch,
    fx_region_to_module,
)
from ..jit_compiler import PtxRoutePlan, ScratchInit
from .gemm import (
    schedule_register_matmul,
    schedule_tiled_matmul,
)
from ..gpu_dialect import (
    AllocOp,
    BF16Type,
    BarrierOp,
    BlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F32Type,
    ForOp,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    KernelOp,
    MatmulOp,
    MemoryEffect,
    ModuleOp,
    MoveOp,
    TileOp,
    WarpGroupIdOp,
    WrmemTensorType,
)
from ..passes.canonicalize import Canonicalize
from ..passes.fuse_async import FuseAsync
from ..passes.lower_to_async import LowerToAsync
from ..passes.schedule_async import ScheduleAsync
from ..passes.stage_register_writeback import StageRegisterWriteback
from ..passes.tile_matmul import TileMatmul
from ..ptx_codegen import make_ptx_kernel


def _gmem(value: Value) -> GmemTensorType:
    if not isinstance(value.type, GmemTensorType):
        raise ValueError(f"expected gmem value, got {value.type}")
    return value.type


def _static_pair(value: Value) -> tuple[int, int]:
    typ = _gmem(value)
    if typ.rank != 2 or any(size is None for size in typ.sizes):
        raise ValueError(f"expected a static rank-2 tensor, got {typ}")
    sizes = cast(tuple[int, int], typ.sizes)
    return sizes[0], sizes[1]


def _match_swiglu_chain(
    module: ModuleOp,
) -> tuple[KernelOp, MatmulOp, MatmulOp]:
    top_ops = module.body.ops
    if len(top_ops) != 1 or not isinstance(top_ops[0], KernelOp):
        raise ValueError("SwiGLU schedule requires one kernel")
    kernel = top_ops[0]
    if len(kernel.body.arguments) != 4:
        raise ValueError("SwiGLU schedule requires three inputs and one result")
    lhs, packed_weight, down_weight, output = kernel.body.arguments
    matmuls = find_ops_of_type(module, MatmulOp)
    tiles = find_ops_of_type(module, TileOp)
    elementwise = find_ops_of_type(module, ElementwiseOp)
    if len(matmuls) != 2 or len(tiles) != 2:
        raise ValueError("expected two matmuls and two projection slices")
    if [op.fn for op in elementwise] != ["convert", "silu", "mul"]:
        raise ValueError("expected convert, SiLU, and multiply epilogue")
    projection, down = matmuls
    gate_convert, silu, multiply = elementwise
    gate = next((tile for tile in tiles if tile.indices[1] == 0), None)
    up = next((tile for tile in tiles if tile is not gate), None)
    if gate is None or up is None:
        raise ValueError("expected gate and up projection slices")
    intermediate = _static_pair(gate.result())[1]
    if not (
        projection.lhs is lhs
        and projection.rhs is packed_weight
        and gate.arg is projection.dst
        and up.arg is projection.dst
        and up.indices[1] == intermediate
        and gate_convert.src(0) is gate.result()
        and silu.src(0) is gate_convert.dst
        and set((multiply.src(0), multiply.src(1))) == {silu.dst, up.result()}
        and down.lhs is multiply.dst
        and down.rhs is down_weight
        and down.dst is output
    ):
        raise ValueError("module is not a packed SwiGLU GEMM chain")
    return kernel, projection, down


def is_swiglu_region(match: FxRegionMatch) -> bool:
    """Whether an FX region imports as the packed SwiGLU GEMM chain."""
    try:
        module = fx_region_to_module(match, name="swiglu_candidate")
        _match_swiglu_chain(module)
    except ValueError:
        return False
    return True


def _projection_ops(
    lhs: Value,
    packed_weight: Value,
    hidden: Value,
    m_index: Value,
    n_index: Value,
    *,
    m: int,
    k: int,
    intermediate: int,
    m_tile: int,
    n_tile: int,
) -> tuple[list[Op], MoveOp]:
    packed_type = _gmem(packed_weight)
    lhs_type = _gmem(lhs)
    hidden_type = _gmem(hidden)
    gate_weight = TileOp.build(
        packed_weight,
        [0, 0],
        GmemTensorType.build(
            [intermediate, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
    )
    up_weight = TileOp.build(
        packed_weight,
        [1, 0],
        GmemTensorType.build(
            [intermediate, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
    )
    lhs_block_tile = TileOp.build(
        lhs,
        [m_index, 0],
        GmemTensorType.build(
            [m_tile, k],
            lhs_type.dtype,
            strides=list(lhs_type.strides),
        ),
        maybe_partial=m % m_tile != 0,
    )
    gate_tile = TileOp.build(
        gate_weight.result(),
        [n_index, 0],
        GmemTensorType.build(
            [n_tile, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
        maybe_partial=intermediate % n_tile != 0,
    )
    up_tile = TileOp.build(
        up_weight.result(),
        [n_index, 0],
        GmemTensorType.build(
            [n_tile, k],
            packed_type.dtype,
            strides=list(packed_type.strides),
        ),
        maybe_partial=intermediate % n_tile != 0,
    )
    hidden_block_tile = TileOp.build(
        hidden,
        [m_index, n_index],
        GmemTensorType.build(
            [m_tile, n_tile],
            hidden_type.dtype,
            strides=list(hidden_type.strides),
        ),
        maybe_partial=m % m_tile != 0,
    )
    m_warp_groups = m_tile // 64
    parallel_ops: list[Op] = []
    if m_warp_groups > 1:
        warp_group_ids = WarpGroupIdOp.build()
        parallel_ops.append(warp_group_ids)
        lhs_tile = TileOp.build(
            lhs_block_tile.result(),
            [warp_group_ids.x, 0],
            GmemTensorType.build(
                [64, k],
                lhs_type.dtype,
                strides=list(lhs_type.strides),
            ),
            maybe_partial=m % m_tile != 0,
        )
        hidden_tile = TileOp.build(
            hidden_block_tile.result(),
            [warp_group_ids.x, 0],
            GmemTensorType.build(
                [64, n_tile],
                hidden_type.dtype,
                strides=list(hidden_type.strides),
            ),
            maybe_partial=m % m_tile != 0,
        )
    else:
        lhs_tile = lhs_block_tile
        hidden_tile = hidden_block_tile

    accumulator_type = WrmemTensorType.build(
        [64, n_tile],
        F32Type(),
        warps=4,
        layout="wgmma_layout_0",
    )
    epilogue_type = WrmemTensorType.build(
        [64, n_tile],
        _gmem(hidden).dtype,
        warps=4,
        layout="wgmma_layout_0",
    )
    gate_acc = AllocOp.build_with_fresh_id(accumulator_type, init=0.0)
    up_acc = AllocOp.build_with_fresh_id(accumulator_type, init=0.0)
    activation = AllocOp.build_with_fresh_id(epilogue_type)
    product = AllocOp.build_with_fresh_id(epilogue_type)
    collective = {
        CollectiveAxis.X: CollectiveLevel.WG,
        CollectiveAxis.Y: CollectiveLevel.GRID,
        CollectiveAxis.Z: CollectiveLevel.GRID,
    }
    gate_matmul = MatmulOp.build(
        lhs_tile.result(),
        gate_tile.result(),
        gate_acc.result(),
        collective=collective,
        attrs={"label": "gate"},
    )
    up_matmul = MatmulOp.build(
        lhs_tile.result(),
        up_tile.result(),
        up_acc.result(),
        collective=collective,
        attrs={"label": "up"},
    )
    round_gate = ElementwiseOp.build(
        activation.result(),
        [gate_acc.result()],
        "convert",
    )
    widen_gate = ElementwiseOp.build(
        gate_acc.result(),
        [activation.result()],
        "convert",
    )
    activate = ElementwiseOp.build(
        activation.result(),
        [gate_acc.result()],
        "silu",
        attrs={"label": "gate_epi"},
    )
    round_up = ElementwiseOp.build(
        product.result(),
        [up_acc.result()],
        "convert",
    )
    multiply = ElementwiseOp.build(
        product.result(),
        [product.result(), activation.result()],
        "mul",
        attrs={"label": "up_epi"},
    )
    writeback = MoveOp.build(
        hidden_tile.result(),
        product.result(),
        collective=collective,
        attrs={
            "label": "up_dst_wrmem",
            "l2_cache_hint": "evict_last",
            "dst_tiling_scheme": MemoryEffect.fresh_tiling_scheme_id(),
        },
    )
    return [
        gate_weight,
        up_weight,
        lhs_block_tile,
        gate_tile,
        up_tile,
        hidden_block_tile,
        *parallel_ops,
        *([] if lhs_tile is lhs_block_tile else [lhs_tile]),
        *([] if hidden_tile is hidden_block_tile else [hidden_tile]),
        gate_acc,
        up_acc,
        activation,
        product,
        gate_matmul,
        round_gate,
        widen_gate,
        activate,
        up_matmul,
        round_up,
        multiply,
        writeback,
    ], writeback


def _schedule_projection(
    module: ModuleOp,
    writeback: MoveOp,
    *,
    stages: int,
    num_warp_groups: int,
    k_tail_tiles: int,
) -> None:
    move_attrs: dict[str, dict[str, object]] = {
        "lhs": {
            "l2_cache_hint": "evict_last",
            "l2_promotion": "128B",
        },
        "rhs": {
            "l2_cache_hint": "evict_first",
            "l2_promotion": "256B",
        },
    }
    schedule_register_matmul(
        label="up|gate",
        operand_stages=stages,
        operand_k_tile=64,
        unsafe_reuse_smem=True,
        operand_move_attrs=move_attrs,
        loop_label_prefix="projection",
        k_tail_tiles=k_tail_tiles,
    ).run(module)
    if k_tail_tiles:
        FuseAsync(fuse_wgmma_groups=True).run(module)

    # Both projections execute concurrently, so their private operand buffers
    # must not be overlaid. They may still share an arena with later phases.
    for alloc in find_ops_of_type(module, AllocOp):
        reuse_group = alloc.attrs.get("reuse_group")
        if (
            isinstance(reuse_group, str)
            and reuse_group.startswith(("up", "gate"))
            and reuse_group.endswith("_compute")
        ):
            alloc.attrs["reuse_group"] = (
                "projection_tail_compute"
                if alloc.attrs.get("peeled") is True
                else "projection_compute"
            )

    if num_warp_groups > 1:
        for barrier in find_ops_of_type(module, BarrierOp):
            if barrier.scope == "thread":
                barrier.attrs["scope"] = "block"

    if num_warp_groups == 1:
        writeback.attrs["tma"] = True
        StageRegisterWriteback(
            unsafe_reuse_smem=True,
            label_filter="up_dst_wrmem",
        ).run(module)
    LowerToAsync().run(module)
    FuseAsync().run(module)
    ScheduleAsync().run(module)
    Canonicalize().run(module)


def _schedule_down(
    module: ModuleOp,
    *,
    m_tile: int,
    stages: int,
) -> None:
    TileMatmul(
        op_dims=["M", "N"],
        tile_sizes=[m_tile, 64],
        iteration_bindings=[None, (CollectiveLevel.BLOCK, CollectiveAxis.X)],
        for_loop_labels=["down_m", None],
        op_label_filter="down",
        allow_partial_tiles=True,
    ).run(module)
    for loop in find_ops_of_type(module, ForOp):
        if loop.attrs.get("label") != "down_m":
            continue
        if loop.trip_count > 1:
            loop.body.ops.append(BarrierOp.build("block"))
    schedule_tiled_matmul(
        label="down",
        output_m_tile=m_tile,
        output_n_tile=64,
        operand_stages=stages,
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


def schedule_swiglu(
    module: ModuleOp,
    *,
    output_m_tile: int = 64,
    projection_n_tile: int = 192,
    projection_k_tail_tiles: int = 0,
) -> ModuleOp:
    """Schedule a packed SwiGLU GEMM chain as one kernel."""
    source_kernel, projection, down = _match_swiglu_chain(module)
    lhs_type = _gmem(projection.lhs)
    packed_type = _gmem(projection.rhs)
    down_type = _gmem(down.rhs)
    output_type = _gmem(down.dst)
    if not all(
        isinstance(typ.dtype, BF16Type)
        for typ in (lhs_type, packed_type, down_type, output_type)
    ):
        raise ValueError("SwiGLU schedule currently requires BF16 tensors")
    m, k = _static_pair(projection.lhs)
    packed_n, packed_k = _static_pair(projection.rhs)
    output_n, intermediate = _static_pair(down.rhs)
    if (
        packed_k != k
        or packed_n != 2 * intermediate
        or _static_pair(down.dst) != (m, output_n)
    ):
        raise ValueError("inconsistent packed SwiGLU dimensions")
    if output_m_tile not in (64, 128, 192, 256):
        raise ValueError("SwiGLU M tile must be 64, 128, 192, or 256")
    num_warp_groups = output_m_tile // 64
    if intermediate % projection_n_tile or output_n % 64:
        raise ValueError("SwiGLU output dimensions must divide schedule tiles")
    projection_blocks = intermediate // projection_n_tile
    down_blocks = output_n // 64
    if projection_blocks % down_blocks:
        raise ValueError("SwiGLU projection blocks must be a multiple of down blocks")
    grid_blocks = down_blocks

    lhs = Value("lhs", clone_type(lhs_type))
    packed_weight = Value("packed_weight", clone_type(packed_type))
    down_weight = Value("down_weight", clone_type(down_type))
    output = Value("output", clone_type(output_type))
    hidden = Value(
        "hidden",
        GmemTensorType.build([m, intermediate], lhs_type.dtype),
    )
    sync = Value("sync", GmemTensorType.build([2], F32Type()))
    block_ids = BlockIdsOp.build()
    projection_tiles = (m + output_m_tile - 1) // output_m_tile
    projection_n_waves = projection_blocks // grid_blocks
    projection_m = (
        block_ids.y
        if projection_tiles == 1
        else Value("projection_m", block_ids.y.type)
    )
    projection_n_wave = Value("projection_n", block_ids.x.type)
    projection_index_ops: list[Op] = []
    if projection_n_waves == 1:
        projection_n = block_ids.x
    else:
        grid_size = IndexConstantOp.build(grid_blocks)
        wave_offset = IndexMulOp.build(
            projection_n_wave,
            grid_size.result(),
        )
        projection_n_op = IndexAddOp.build(
            wave_offset.result(),
            block_ids.x,
        )
        projection_index_ops = [grid_size, wave_offset, projection_n_op]
        projection_n = projection_n_op.result()
    projection_ops, writeback = _projection_ops(
        lhs,
        packed_weight,
        hidden,
        projection_m,
        projection_n,
        m=m,
        k=k,
        intermediate=intermediate,
        m_tile=output_m_tile,
        n_tile=projection_n_tile,
    )
    projection_phase: list[Op] = [*projection_index_ops, *projection_ops]
    if projection_n_waves > 1:
        projection_phase.append(BarrierOp.build("block"))
        projection_phase = [
            ForOp.build(
                0,
                projection_n_waves,
                body=Block(
                    arguments=[projection_n_wave],
                    ops=projection_phase,
                ),
                attrs={"label": "projection_n"},
            ),
        ]
    if projection_tiles > 1:
        if projection_n_waves == 1:
            projection_phase.append(BarrierOp.build("block"))
        projection_phase = [
            ForOp.build(
                0,
                projection_tiles,
                body=Block(arguments=[projection_m], ops=projection_phase),
                attrs={"label": "projection_m"},
            ),
        ]
    down_op = MatmulOp.build(
        hidden,
        down_weight,
        output,
        collective={axis: CollectiveLevel.GRID for axis in CollectiveAxis},
        init=0.0,
        attrs={"label": "down"},
    )
    kernel = KernelOp.build(
        source_kernel.name,
        Block(
            arguments=[
                lhs,
                packed_weight,
                down_weight,
                hidden,
                output,
                sync,
            ],
            ops=[
                block_ids,
                *projection_phase,
                BarrierOp.build("grid", sync=sync),
                down_op,
            ],
        ),
        grid_dim=(grid_blocks, 1, 1),
        block_dim=(128 * num_warp_groups, 1, 1),
        num_warp_groups=num_warp_groups,
        ptx_optimization_level=1 if m == 512 else 4,
    )
    scheduled = ModuleOp.build(Block(ops=[kernel]))
    _schedule_projection(
        scheduled,
        writeback,
        stages=4,
        num_warp_groups=num_warp_groups,
        k_tail_tiles=projection_k_tail_tiles,
    )
    _schedule_down(
        scheduled,
        m_tile=output_m_tile,
        stages=5 if m == 512 else (5 if m <= 32 else 6),
    )
    ScheduleAsync().run(scheduled)
    return Canonicalize().run(scheduled)


def compile_swiglu(module: ModuleOp) -> PtxRoutePlan:
    """Compile an imported packed SwiGLU region."""
    _, projection, _ = _match_swiglu_chain(module)
    m, _ = _static_pair(projection.lhs)
    return PtxRoutePlan(
        make_ptx_kernel(
            schedule_swiglu(
                module,
                output_m_tile=256 if m == 512 else 64,
                projection_n_tile=96 if m == 512 else 192,
                projection_k_tail_tiles=int(
                    os.environ.get("AYLA_SWIGLU_K_TAIL_TILES", "0")
                ),
            )
        ),
        bindings=[
            0,
            1,
            2,
            ScratchInit.EMPTY,
            3,
            ScratchInit.ZERO,
        ],
    )
