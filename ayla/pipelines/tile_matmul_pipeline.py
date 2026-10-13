"""Tile matmul pipeline."""
from __future__ import annotations

from ..base.pipeline import Pipeline
from ..gpu_dialect import CollectiveAxis, CollectiveLevel
from ..passes.fuse_async import FuseAsync
from ..passes.licm import LICM
from ..passes.lower_to_async import LowerToAsync
from ..passes.multi_buffer import MultiBuffer
from ..passes.pipeline_loop import PipelineLoop
from ..passes.promote_buffers import PromoteBuffers
from ..passes.schedule_async import ScheduleAsync
from ..passes.tile_matmul import TileMatmul
from ..passes.unroll_loop import UnrollLoop
from .canonicalize import Canonicalize


class TileMatmulPipeline(Pipeline):
    """Tile and schedule a matmul with concrete M, N, and K dimensions.

    Register accumulators already own output tiling and promotion; their
    schedule starts with reduction tiling and operand movement.
    """

    def __init__(
        self,
        *,
        m: int,
        n: int,
        k: int,
        register_accumulator: bool = False,
        output_m_tiles_per_cta: int = 1,
        output_n_tiles_per_cta: int = 1,
        output_m_tile: int = 128,
        output_n_tile: int = 256,
        warpgroup_n_tile: int | None = None,
        operand_stages: int = 3,
        operand_k_tile: int = 64,
        output_slices: int = 1,
        output_store_buffers: int = 1,
        # Split each 256-column output row into two 128-element TMA dimensions.
        # Shared-only padding controls the physical row stride seen by the TMA
        # store; the best overall stride may trade some STMatrix bank expansion
        # for faster TMA reads of the arena.
        output_tma_inner_size: int | None = 128,
        output_store: str = "stmatrix",
        operand_move_attrs: dict[str, dict[str, object]] | None = None,
        label: str = "mm",
        epilogue_label_filter: str | None = None,
        loop_label_prefix: str | None = None,
        unsafe_reuse_smem: bool = False,
    ) -> None:
        super().__init__()
        group_size = 1
        group_label = "outn" if output_n_tiles_per_cta > 1 else "outm"
        rhs_tma_load_slices = 1
        if not register_accumulator:
            output_groups = (output_m_tiles_per_cta, output_n_tiles_per_cta)
            if any(group not in (1, 2, 3, 4, 6, 8, 16) for group in output_groups):
                raise ValueError(
                    "output tile groups must each be 1, 2, 3, 4, 6, 8, or 16"
                )
            if output_m_tiles_per_cta > 1 and output_n_tiles_per_cta > 1:
                raise ValueError("output tiles may be grouped along only one axis")
            if min(m, n, k) <= 0:
                raise ValueError("problem dimensions must be positive")
            if output_m_tile not in (64, 128, 192, 256):
                raise ValueError("output_m_tile must be 64, 128, 192, or 256")
            if not 48 <= output_n_tile <= 384 or output_n_tile % 8:
                raise ValueError(
                    "output_n_tile must be a multiple of 8 between 48 and 384"
                )
            if operand_k_tile not in (64, 128):
                raise ValueError("operand_k_tile must be 64 or 128")
            if warpgroup_n_tile is None:
                warpgroup_n_tile = output_n_tile if output_n_tile <= 256 else 128
            if output_n_tile % warpgroup_n_tile:
                raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
            if output_store not in ("scalar", "stmatrix"):
                raise ValueError("output_store must be 'scalar' or 'stmatrix'")
            if output_store == "scalar":
                output_tma_inner_size = None
            m_blocks = (m + output_m_tile - 1) // output_m_tile
            n_blocks = (n + output_n_tile - 1) // output_n_tile
            if output_n_tiles_per_cta > 1:
                group_dim = "N"
                blocks = n_blocks
                tile_size = output_n_tile
                group_size = output_n_tiles_per_cta
            else:
                group_dim = "M"
                blocks = m_blocks
                tile_size = output_m_tile
                group_size = output_m_tiles_per_cta
            if group_size == 1:
                self.add_pass(TileMatmul(
                    op_dims=["M", "N"],
                    tile_sizes=[output_m_tile, output_n_tile],
                    iteration_bindings=[
                        (CollectiveLevel.BLOCK, CollectiveAxis.X),
                        (CollectiveLevel.BLOCK, CollectiveAxis.Y),
                    ],
                    for_loop_labels=[None, None],
                    op_label_filter=label,
                    epilogue_label_filter=epilogue_label_filter,
                    allow_partial_tiles=(
                        m % output_m_tile != 0
                        or n % output_n_tile != 0
                    ),
                ))
            else:
                grid = (blocks + group_size - 1) // group_size
                self.add_pass(TileMatmul(
                    op_dims=[group_dim, "M", "N"],
                    tile_sizes=[
                        grid * tile_size, output_m_tile, output_n_tile,
                    ],
                    iteration_bindings=[
                        None,
                        (CollectiveLevel.BLOCK, CollectiveAxis.X),
                        (CollectiveLevel.BLOCK, CollectiveAxis.Y),
                    ],
                    for_loop_labels=[group_label, None, None], op_label_filter=label,
                    epilogue_label_filter=epilogue_label_filter,
                    allow_partial_tiles=(
                        m % output_m_tile != 0
                        or n % output_n_tile != 0
                        or blocks % group_size != 0
                    ),
                ))
            wg_dim, wg_size = (
                ("N", warpgroup_n_tile)
                if output_m_tile == 64 or output_n_tile > 256
                else ("M", 64)
            )

            self.add_pass(TileMatmul(
                op_dims=[wg_dim], tile_sizes=[wg_size],
                iteration_bindings=[(CollectiveLevel.WG, CollectiveAxis.X)],
                for_loop_labels=[None],
                op_label_filter=label,
                epilogue_label_filter=epilogue_label_filter,
                allow_partial_tiles=(wg_dim == "M" and output_m_tile % wg_size != 0),
            ))
            self.add_pass(PromoteBuffers(
                target_type="wgrmem", operands=["dst"], swizzle=None,
                layout="wgmma_layout_0", warps=4, tma=True,
                physical_layout=(
                    {
                        "swizzle": "128B",
                        "block_width": 64,
                        "order": "block_major",
                    }
                    if output_tma_inner_size is not None else None
                ),
                writeback_slices=output_slices,
                writeback_buffers=output_store_buffers,
                move_attrs=operand_move_attrs,
                target_dtype="f32", label_filter=label,
                epilogue_label_filter=epilogue_label_filter,
                unsafe_reuse_smem=unsafe_reuse_smem,
            ))
            # Preserve three 128-row RHS loads for the tuned N=384 schedule.
            rhs_tma_load_slices = (
                output_n_tile // warpgroup_n_tile if output_n_tile == 384 else 1
            )

        k_label = f"{loop_label_prefix}_k" if loop_label_prefix else "k"
        kk_label = f"{loop_label_prefix}_kk" if loop_label_prefix else "kk"

        self.add_pass(TileMatmul(
            op_dims=["K"], tile_sizes=[operand_k_tile],
            iteration_bindings=[None], for_loop_labels=[k_label],
            op_label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
        ))
        self.add_pass(PromoteBuffers(
            target_type="smem", operands=["lhs", "rhs"],
            swizzle="128B", layout=None, warps=None, tma=True,
            physical_layout=(
                {"block_width": 64, "order": "block_major"}
                if operand_k_tile > 64 else None
            ),
            tma_load_slices=(
                {"rhs": rhs_tma_load_slices}
                if rhs_tma_load_slices > 1 else None
            ),
            move_attrs=operand_move_attrs,
            unsafe_reuse_smem=unsafe_reuse_smem,
            label_filter=label,
        ))
        self.add_pass(TileMatmul(
            op_dims=["K"], tile_sizes=[16],
            iteration_bindings=[None], for_loop_labels=[kk_label],
            op_label_filter=label,
            epilogue_label_filter=epilogue_label_filter,
        ))
        self.add_pass(MultiBuffer(
            depth=operand_stages, label_filter=f"{label}_lhs_smem",
        ))
        self.add_pass(MultiBuffer(
            depth=operand_stages, label_filter=f"{label}_rhs_smem",
        ))
        self.add_pass(LICM())
        self.add_pass(UnrollLoop(
            unroll_factor=operand_stages, label_filter=k_label,
        ))
        self.add_pass(Canonicalize())
        self.add_pass(LowerToAsync())
        self.add_pass(FuseAsync())
        self.add_pass(LICM())
        self.add_pass(Canonicalize())
        self.add_pass(PipelineLoop(
            label_filter_loop=k_label,
            label_filter_op=rf"{label}_(lhs|rhs)_smem_iter\d+_async",
            pipelining_offset=1,
        ))
        self.add_pass(UnrollLoop(
            unroll_factor=operand_k_tile // 16,
            label_filter=rf"{kk_label}(?:_tail\d+)?(?:_iter\d+)?",
        ))
        self.add_pass(FuseAsync())
        for _ in range(2):
            self.add_pass(LICM())
            self.add_pass(Canonicalize())

        if not register_accumulator:
            if group_size > 1:
                self.add_pass(UnrollLoop(
                    unroll_factor=group_size,
                    label_filter=group_label,
                ))
            self.add_pass(ScheduleAsync())
            self.add_pass(Canonicalize())
