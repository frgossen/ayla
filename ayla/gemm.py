"""Configurable GEMM module builders."""
import textwrap

from .base.ir import Block, Value
from .gpu_dialect import (
    AllocOp,
    AsyncCommitOp,
    AsyncOp,
    AwaitOp,
    BF16Type,
    BlockIdsOp,
    DeallocOp,
    F32Type,
    ForOp,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LockType,
    MatmulOp,
    ModuleOp,
    MoveOp,
    PartialSplitBlockIdsOp,
    SplitNBlockIdOp,
    SmemTensorType,
    TileOp,
    TmemTensorType,
    gpu_dialect,
)
from .targets import BLACKWELL
from .passes.canonicalize import Canonicalize
from .passes.fuse_async import fuse_async
from .passes.licm import licm
from .passes.lower_to_async import lower_to_async
from .passes.multi_buffer import multi_buffer
from .passes.pipeline_loop import pipeline_loop
from .passes.promote_buffers import promote_buffers
from .passes.schedule_async import schedule_async
from .passes.tile_matmul import tile_matmul
from .passes.unroll_loop import unroll_loop


def blackwell_tcgen05_gemm_module(
    *, m: int = 4096, n: int = 4096, k: int = 4096,
    operand_stages: int = 4, tma_staging: str | None = "direct",
    output_m_tile: int = 128, output_n_tile: int = 128,
    ptx_optimization_level: int | None = 1,
    cluster_multicast_a: bool = False,
    branch_tcgen05_issue: bool = True,
    transpose_cta_grid: bool = False,
    split_last_n_tile: bool = False,
    split_n_tile: int | None = None,
    split_last_n_tile_rows: int | None = None,
    tail_n_tile: int | None = None,
    balanced_n_tile: int | None = None,
    predicated_tail_stores: bool = False,
    lhs_l2_cache_hint: str | None = None,
    rhs_l2_cache_hint: str | None = None,
    lhs_l2_promotion: str | None = None,
    rhs_l2_promotion: str | None = None,
    split_tma_barriers: bool = False,
    split_rhs_tma: bool = False,
    prefetch_lhs_next: bool = False,
    prefetch_rhs_next: bool = False,
    output_l1_cache_policy: str | None = None,
    output_l2_eviction_priority: str | None = None,
    output_tmem_load_shape: str = "32x32b",
    persistent_n_pairs: bool = False,
    lean_bookkeeping: bool = False,
) -> ModuleOp:
    """Build the first native SM100 BF16 GEMM pipeline.

    Each CTA computes one MxN output tile. All four warps cooperatively
    stage both operands and read the tensor-memory accumulator; only the
    elected issue sequence required by ``tcgen05.mma`` is single-threaded.
    ``lean_bookkeeping`` preserves that shared software pipeline while
    reusing its statically elected thread-zero predicate, deriving adjacent
    K16 descriptors, and joining the already-overlapped next-A0/B0 completion.
    With ``cluster_multicast_a``, paired CTAs still execute this same pipeline;
    cluster rank 0 only performs the instruction-mandated issue of their shared
    A transaction, while each CTA independently loads B and computes its tile.
    """
    if output_m_tile not in (64, 128):
        raise ValueError("output_m_tile must be 64 or 128")
    if output_n_tile not in (64, 128, 232, 240, 248, 256):
        raise ValueError(
            "output_n_tile must be 64, 128, 232, 240, 248, or 256"
        )
    if m % output_m_tile or k % 16:
        raise ValueError(
            "tcgen05 GEMM dimensions must divide the M and K tile sizes"
        )
    if tail_n_tile is None:
        if balanced_n_tile is None and n % output_n_tile:
            raise ValueError("N must divide the requested tcgen05 output tile")
    elif (
        not isinstance(tail_n_tile, int)
        or tail_n_tile % 8
        or tail_n_tile < 8
        or tail_n_tile >= output_n_tile
        or n % output_n_tile != tail_n_tile
    ):
        raise ValueError(
            "tail_n_tile must be a positive 8-column multiple smaller than "
            "the output tile and match N's remainder"
        )
    if operand_stages < 1 or k % (16 * operand_stages):
        raise ValueError("K must divide the requested tcgen05 operand stages")
    if tma_staging not in (None, "repack", "direct"):
        raise ValueError("tma_staging must be None, 'repack', or 'direct'")
    if tma_staging == "direct" and operand_stages not in (4, 8):
        raise ValueError(
            "direct TMA staging currently requires four or eight K=16 stages"
        )
    if output_n_tile != 64 and tma_staging != "direct":
        raise ValueError("wide output tiles currently require direct TMA staging")
    if output_m_tile != 128 and tma_staging != "direct":
        raise ValueError("M=64 output tiles currently require direct TMA staging")
    if cluster_multicast_a and (
        tma_staging != "direct" or output_m_tile != 128 or output_n_tile != 128
    ):
        raise ValueError("cluster A multicast requires direct M=128/N=128 tiles")
    if cluster_multicast_a and n % 256:
        raise ValueError("cluster A multicast requires N divisible by 256")
    if cluster_multicast_a and transpose_cta_grid:
        raise ValueError("cluster A multicast requires the default CTA grid order")
    if sum(
        option is not None and option is not False
        for option in (
            split_last_n_tile, split_n_tile, split_last_n_tile_rows,
            tail_n_tile, balanced_n_tile,
        )
    ) > 1:
        raise ValueError(
            "choose one N-tile splitting mode"
        )
    split_last_n_tile_enabled = (
        split_last_n_tile
        or split_n_tile is not None
        or split_last_n_tile_rows is not None
    )
    if split_last_n_tile_enabled and (
        tma_staging != "direct"
        or output_n_tile != 256
        or not transpose_cta_grid
        or n // output_n_tile < 2
    ):
        raise ValueError(
            "splitting the last N tile requires direct transposed-grid "
            "N=256 staging with at least two N tiles"
        )
    if tail_n_tile is not None and (
        tma_staging != "direct" or not transpose_cta_grid
    ):
        raise ValueError(
            "a narrow N tail requires direct staging and a transposed CTA grid"
        )
    if predicated_tail_stores and (
        tail_n_tile is None or output_m_tile != 128
    ):
        raise ValueError(
            "predicated tail stores require an M128 narrow-tail launch"
        )
    m_tiles = m // output_m_tile
    n_tiles = (n + output_n_tile - 1) // output_n_tile
    balanced_wide_tiles = None
    if balanced_n_tile is not None:
        delta = output_n_tile - balanced_n_tile
        extra = n - n_tiles * balanced_n_tile
        if (
            tma_staging != "direct"
            or not transpose_cta_grid
            or type(balanced_n_tile) is not int
            or balanced_n_tile % 8
            or not 8 <= balanced_n_tile < output_n_tile
            or extra <= 0
            or extra % delta
            or not 0 < extra // delta < n_tiles
        ):
            raise ValueError(
                "balanced_n_tile requires direct transposed-grid staging and "
                "an exact mix of valid narrow/full N tiles"
            )
        balanced_wide_tiles = extra // delta
    if persistent_n_pairs and (
        balanced_n_tile is None
        or not transpose_cta_grid
        or n_tiles % 2
        or balanced_wide_tiles is None
        or balanced_wide_tiles > n_tiles // 2
        or (n_tiles - 2 * balanced_wide_tiles) % 2
    ):
        raise ValueError(
            "persistent N pairing requires an even exact balanced-N launch "
            "whose wide tiles can each pair with a narrow tile"
        )
    if split_n_tile is not None and (
        type(split_n_tile) is not int or not 0 <= split_n_tile < n_tiles
    ):
        raise ValueError("split_n_tile must select an existing logical N tile")
    if split_last_n_tile_rows is not None and (
        type(split_last_n_tile_rows) is not int
        or not 0 < split_last_n_tile_rows < m_tiles
    ):
        raise ValueError(
            "split_last_n_tile_rows must be an integer strictly between "
            "zero and the number of M tiles"
        )
    if lhs_l2_cache_hint not in (None, "evict_first", "evict_last"):
        raise ValueError("unsupported lhs_l2_cache_hint")
    if rhs_l2_cache_hint not in (None, "evict_first", "evict_last"):
        raise ValueError("unsupported rhs_l2_cache_hint")
    if lhs_l2_promotion not in (None, "64B", "128B", "256B"):
        raise ValueError("unsupported lhs_l2_promotion")
    if rhs_l2_promotion not in (None, "64B", "128B", "256B"):
        raise ValueError("unsupported rhs_l2_promotion")
    if output_l1_cache_policy not in (
        None, "no_allocate", "evict_first", "evict_last",
    ):
        raise ValueError("unsupported output_l1_cache_policy")
    if output_l2_eviction_priority not in (None, "evict_first", "evict_last"):
        raise ValueError("unsupported output_l2_eviction_priority")
    if output_tmem_load_shape not in ("32x32b", "16x32bx2"):
        raise ValueError("unsupported output_tmem_load_shape")
    if output_tmem_load_shape == "16x32bx2" and (
        output_m_tile != 128
        or output_n_tile != 232
        or balanced_n_tile != 224
    ):
        raise ValueError(
            "16x32bx2 output requires balanced M128/N232/N224 tiles"
        )
    if split_tma_barriers and (
        tma_staging != "direct"
        or operand_stages != 8
        or cluster_multicast_a
    ):
        raise ValueError(
            "split TMA barriers require direct eight-stage non-multicast staging"
        )
    if split_rhs_tma and (
        tma_staging != "direct" or output_n_tile not in (232, 240, 248, 256)
    ):
        raise ValueError(
            "split RHS TMA transactions require direct N=232/N=240/N=248/"
            "N=256 staging"
        )
    if prefetch_lhs_next and (
        tma_staging != "direct"
        or operand_stages != 8
        or not split_tma_barriers
        or cluster_multicast_a
        or k < 256
    ):
        raise ValueError(
            "next-iteration LHS prefetch requires direct eight-stage split-"
            "barrier staging, no cluster multicast, and at least K=256"
        )
    if prefetch_rhs_next and (
        not prefetch_lhs_next
        or not split_rhs_tma
        or output_n_tile not in (232, 240, 248, 256)
    ):
        raise ValueError(
            "next-iteration RHS prefetch requires next-iteration LHS "
            "prefetch and split direct N=232/N=240/N=248/N=256 RHS "
            "transactions"
        )
    if lean_bookkeeping and (
        not prefetch_rhs_next
        or not branch_tcgen05_issue
        or tma_staging != "direct"
        or operand_stages != 8
    ):
        raise ValueError(
            "lean bookkeeping requires the branched direct K128 pipeline "
            "with next-iteration LHS and RHS prefetch"
        )

    bf16 = BF16Type()
    a = Value(name="a", type=GmemTensorType.build([m, k], bf16))
    bt = Value(name="bt", type=GmemTensorType.build([n, k], bf16))
    c = Value(name="c", type=GmemTensorType.build([m, n], bf16))
    block_ids = BlockIdsOp.build()
    grid_index_ops = [block_ids]
    persistent_loop = (
        ForOp.build(0, 2, attrs={"disjoint_grid_iterations": True})
        if persistent_n_pairs else None
    )
    logical_block_x = block_ids.x
    if persistent_loop is not None:
        assert balanced_wide_tiles is not None
        pair_count = IndexConstantOp.build(n_tiles // 2)
        threshold_bias = IndexConstantOp.build(
            n_tiles // 2 - balanced_wide_tiles,
        )
        biased_pair = IndexAddOp.build(
            block_ids.x, threshold_bias.result(),
        )
        leftover_pair = IndexDivOp.build(
            biased_pair.result(), pair_count.result(),
        )
        doubled_pair = IndexMulOp.build(
            block_ids.x, leftover_pair.result(),
        )
        first_tile = IndexAddOp.build(
            block_ids.x, doubled_pair.result(),
        )
        wide_count = IndexConstantOp.build(balanced_wide_tiles)
        leftover_delta = IndexConstantOp.build(1 - balanced_wide_tiles)
        delta_adjustment = IndexMulOp.build(
            leftover_pair.result(), leftover_delta.result(),
        )
        tile_delta = IndexAddOp.build(
            wide_count.result(), delta_adjustment.result(),
        )
        iteration_delta = IndexMulOp.build(
            persistent_loop.iv, tile_delta.result(),
        )
        paired_tile = IndexAddOp.build(
            first_tile.result(), iteration_delta.result(),
        )
        grid_index_ops.extend([
            pair_count, threshold_bias, biased_pair, leftover_pair,
            doubled_pair, first_tile, wide_count, leftover_delta,
            delta_adjustment, tile_delta, iteration_delta, paired_tile,
        ])
        logical_block_x = paired_tile.result()
    hybrid_n_tiles = None
    hybrid_n_width = None
    hybrid_split_rows = None
    hybrid_split_tile = None
    hybrid_element_offsets = False
    hybrid_controls = None
    if balanced_n_tile is not None:
        assert balanced_wide_tiles is not None
        # Put the wider CTAs first, then derive one CTA-uniform narrow flag and
        # element offset for the remaining blocks. For N4096 this maps eight
        # N232 and ten N224 tiles into the same 18-CTA row as the retained
        # N232/N152 launch while avoiding a short tail.
        width_delta = IndexConstantOp.build(output_n_tile - balanced_n_tile)
        minus_one = IndexConstantOp.build(-1)
        tile_count = IndexConstantOp.build(n_tiles)
        full_width = IndexConstantOp.build(output_n_tile)
        full_base = IndexMulOp.build(logical_block_x, full_width.result())
        threshold_bias = IndexConstantOp.build(n_tiles - balanced_wide_tiles)
        biased_x = IndexAddOp.build(logical_block_x, threshold_bias.result())
        narrow = IndexDivOp.build(biased_x.result(), tile_count.result())
        wide_count = IndexConstantOp.build(balanced_wide_tiles)
        neg_wide_count = IndexMulOp.build(
            wide_count.result(), minus_one.result(),
        )
        after_wide = IndexAddOp.build(
            logical_block_x, neg_wide_count.result(),
        )
        narrow_distance = IndexMulOp.build(
            narrow.result(), after_wide.result(),
        )
        adjustment = IndexMulOp.build(
            narrow_distance.result(), width_delta.result(),
        )
        neg_adjustment = IndexMulOp.build(
            adjustment.result(), minus_one.result(),
        )
        n_offset = IndexAddOp.build(
            full_base.result(), neg_adjustment.result(),
        )
        zero = IndexConstantOp.build(0)
        grid_index_ops.extend([
            width_delta, minus_one, tile_count, full_width, full_base,
            threshold_bias, biased_x, narrow, wide_count, neg_wide_count,
            after_wide, narrow_distance, adjustment, neg_adjustment,
            n_offset, zero,
        ])
        m_block_id = block_ids.y
        n_block_id = n_offset.result()
        hybrid_n_tiles = n_tiles
        hybrid_n_width = balanced_n_tile
        hybrid_element_offsets = True
        hybrid_controls = [narrow.result(), zero.result()]
    elif tail_n_tile is not None:
        # A final naturally narrow tile avoids duplicating a full N-wide TMA
        # operand solely to improve launch-wave balance.  Integer division is
        # CTA-uniform: blocks before the last produce zero and the tail block
        # produces one, while its half selector remains zero.
        full_n_tiles = IndexConstantOp.build(n // output_n_tile)
        narrow = IndexDivOp.build(block_ids.x, full_n_tiles.result())
        zero = IndexConstantOp.build(0)
        grid_index_ops.extend([full_n_tiles, narrow, zero])
        m_block_id = block_ids.y
        n_block_id = block_ids.x
        hybrid_n_tiles = n_tiles
        hybrid_n_width = tail_n_tile
        hybrid_split_tile = n_tiles - 1
        hybrid_controls = [narrow.result(), zero.result()]
    elif split_last_n_tile_rows is not None:
        # A linear mapping splits the last N256 tile only for a prefix of M
        # rows. This adds enough N128 CTAs to improve launch-wave balance while
        # avoiding the duplicated TMA work of splitting every row.
        partial_ids = PartialSplitBlockIdsOp.build(
            m_tiles=m_tiles, n_tiles=n_tiles,
            split_rows=split_last_n_tile_rows,
        )
        grid_index_ops = [partial_ids]
        m_block_id = partial_ids.m
        n_block_id = partial_ids.n
        hybrid_n_tiles = n_tiles
        hybrid_n_width = output_n_tile // 2
        hybrid_split_rows = split_last_n_tile_rows
        hybrid_controls = [partial_ids.narrow, partial_ids.half]
    elif split_n_tile is not None:
        split_ids = SplitNBlockIdOp.build(
            n_tiles=n_tiles, split_tile=split_n_tile,
        )
        grid_index_ops.append(split_ids)
        m_block_id = block_ids.y
        n_block_id = split_ids.n
        hybrid_n_tiles = n_tiles
        hybrid_n_width = output_n_tile // 2
        hybrid_split_tile = split_n_tile
        hybrid_controls = [split_ids.narrow, split_ids.half]
    elif split_last_n_tile:
        # Split only the final N256 tile into two N128 CTAs. For a 4096-wide
        # output this changes 512 uniform CTAs into 544 mixed-width CTAs,
        # reducing the severe two-CTA/SM tail-wave imbalance. Every CTA still
        # executes the same load/compute/store pipeline; the uniform block id
        # only selects its tcgen05 N extent and output half.
        n_tile_count = IndexConstantOp.build(n_tiles)
        half = IndexDivOp.build(block_ids.x, n_tile_count.result())
        minus_one = IndexConstantOp.build(-1)
        neg_half = IndexMulOp.build(half.result(), minus_one.result())
        logical_n = IndexAddOp.build(block_ids.x, neg_half.result())
        grid_index_ops.extend([
            n_tile_count, half, minus_one, neg_half, logical_n,
        ])
        m_block_id = block_ids.y
        n_block_id = logical_n.result()
        hybrid_n_tiles = n_tiles
        hybrid_n_width = output_n_tile // 2
    else:
        m_block_id = block_ids.y if transpose_cta_grid else block_ids.x
        n_block_id = block_ids.x if transpose_cta_grid else block_ids.y
    a_block = TileOp.build(
        a, [m_block_id, 0],
        GmemTensorType.build([output_m_tile, k], bf16, strides=[k, 1]),
    )
    bt_block = TileOp.build(
        bt, [n_block_id, 0],
        GmemTensorType.build([output_n_tile, k], bf16, strides=[k, 1]),
        attrs={"element_offset_dims": [0]} if hybrid_element_offsets else None,
    )
    c_block = TileOp.build(
        c, [m_block_id, n_block_id],
        GmemTensorType.build(
            [output_m_tile, output_n_tile], bf16, strides=[n, 1],
        ),
        attrs={"element_offset_dims": [1]} if hybrid_element_offsets else None,
    )
    accumulator = AllocOp.build_with_fresh_id(
        TmemTensorType.build(
            [output_m_tile, output_n_tile], F32Type(),
            columns=(
                256 if output_n_tile in (232, 240, 248) else output_n_tile
            ),
        ),
        # Each persistent tile starts with an overwrite-mode MMA, so a one-time
        # zero fill is both unnecessary for correctness and dead setup work.
        init=None if persistent_n_pairs else 0.0,
    )
    if prefetch_lhs_next:
        direct_buffers = operand_stages // 4
        lhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [output_m_tile, 64], bf16, swizzle="128B",
                attrs={"tcgen05_matrix": True},
            ))
            for _ in range(direct_buffers + 1)
        ]
        rhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [output_n_tile, 64], bf16, swizzle="128B",
                attrs={"tcgen05_matrix": True},
            ))
            for _ in range(direct_buffers)
        ]
        lhs_tma_smem = []
        rhs_tma_smem = []
        mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
        tma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
        second_tma_lock = AllocOp.build_with_fresh_id(
            LockType.build(kind="mbar")
        )
        prefetch_tma_lock = (
            None if lean_bookkeeping else
            AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
        )
        half_mma_lock = (
            AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
            if prefetch_rhs_next else None
        )
    mainloop_ops = []
    if tma_staging == "direct" and prefetch_lhs_next:
        assert tma_lock is not None
        assert second_tma_lock is not None
        assert len(lhs_smem) == 3 and len(rhs_smem) == 2

        issue_attrs = (
            {"issue_by_thread_zero": True} if lean_bookkeeping else {}
        )

        lhs_move_attrs = {
            "tma": True,
            **({"l2_cache_hint": lhs_l2_cache_hint}
               if lhs_l2_cache_hint else {}),
            **({"l2_promotion": lhs_l2_promotion}
               if lhs_l2_promotion else {}),
        }
        rhs_move_attrs = {
            "tma": True,
            **({"l2_cache_hint": rhs_l2_cache_hint}
               if rhs_l2_cache_hint else {}),
            **({"l2_promotion": rhs_l2_promotion}
               if rhs_l2_promotion else {}),
        }
        hybrid_attrs = (
            {
                "hybrid_n_tiles": hybrid_n_tiles,
                "hybrid_n_width": hybrid_n_width,
                **({"hybrid_split_rows": hybrid_split_rows}
                   if hybrid_split_rows is not None else {}),
                **({"hybrid_split_tile": hybrid_split_tile}
                   if hybrid_split_tile is not None else {}),
                **({"hybrid_element_offsets": True}
                   if hybrid_element_offsets else {}),
            }
            if hybrid_n_tiles is not None else None
        )

        def prefetch_iteration(
            iteration: Value,
            a0_buffer: Value,
            a1_buffer: Value,
            free_buffer: Value,
            *,
            load_a0: bool,
            issue_next_a0: bool,
            overwrite_first: bool = False,
        ) -> list:
            """Build one K128 step of the three-A-buffer software pipeline."""
            ops = []
            two = IndexConstantOp.build(2)
            first_k = IndexMulOp.build(iteration, two.result())
            one = IndexConstantOp.build(1)
            second_k = IndexAddOp.build(first_k.result(), one.result())
            ops.extend([two, first_k, one, second_k])

            lhs0_tile = TileOp.build(
                a_block.result(), [0, first_k.result()],
                GmemTensorType.build(
                    [output_m_tile, 64], bf16, strides=[k, 1],
                ),
            )
            lhs1_tile = TileOp.build(
                a_block.result(), [0, second_k.result()],
                GmemTensorType.build(
                    [output_m_tile, 64], bf16, strides=[k, 1],
                ),
            )
            ops.extend([lhs0_tile, lhs1_tile])
            bank_moves = [[], [MoveOp.build(
                a1_buffer, lhs1_tile.result(), attrs=lhs_move_attrs,
            )]]
            if load_a0:
                bank_moves[0].append(MoveOp.build(
                    a0_buffer, lhs0_tile.result(), attrs=lhs_move_attrs,
                ))

            for bank, k_index in enumerate(
                (first_k.result(), second_k.result())
            ):
                if bank == 0 and prefetch_rhs_next and not load_a0:
                    continue
                for n_half in range(2):
                    bt_tile = TileOp.build(
                        bt_block.result(), [n_half, k_index],
                        GmemTensorType.build(
                            [output_n_tile // 2, 64], bf16,
                            strides=[k, 1],
                        ),
                    )
                    rhs_tile = TileOp.build(
                        rhs_smem[bank].result(), [n_half, 0],
                        SmemTensorType.build(
                            [output_n_tile // 2, 64], bf16,
                            strides=[64, 1],
                            swizzle="128B",
                            attrs={"tcgen05_matrix": True},
                        ),
                    )
                    ops.extend([bt_tile, rhs_tile])
                    bank_moves[bank].append(MoveOp.build(
                        rhs_tile.result(), bt_tile.result(),
                        attrs=rhs_move_attrs,
                    ))

            if bank_moves[0]:
                ops.append(AsyncOp.build(
                    tma_lock.result(), body=Block(ops=bank_moves[0]),
                    attrs=issue_attrs,
                ))
            ops.append(AsyncOp.build(
                second_tma_lock.result(), body=Block(ops=bank_moves[1]),
                attrs=issue_attrs,
            ))
            if not load_a0:
                # The A0 tile was launched after the prior iteration's tcgen
                # commit. Every warp observes its completion before use.
                if not lean_bookkeeping:
                    assert prefetch_tma_lock is not None
                    ops.append(AwaitOp.build(prefetch_tma_lock.result()))
            ops.append(AwaitOp.build(tma_lock.result()))

            mmas = []
            for stage in range(operand_stages):
                if stage == 4:
                    if prefetch_rhs_next and issue_next_a0:
                        assert half_mma_lock is not None
                        mmas.append(AsyncCommitOp.build(half_mma_lock.result()))
                    mmas.append(AwaitOp.build(second_tma_lock.result()))
                bank = stage // 4
                buffer_stage = stage % 4
                lhs_tile = TileOp.build(
                    a0_buffer if bank == 0 else a1_buffer,
                    [0, buffer_stage],
                    SmemTensorType.build(
                        [output_m_tile, 16], bf16,
                        strides=[64, 1], swizzle="128B",
                        attrs={"tcgen05_tma_layout": True},
                    ),
                )
                rhs_tile = TileOp.build(
                    rhs_smem[bank].result(), [0, buffer_stage],
                    SmemTensorType.build(
                        [output_n_tile, 16], bf16,
                        strides=[64, 1], swizzle="128B",
                        attrs={"tcgen05_tma_layout": True},
                    ),
                )
                ops.extend([lhs_tile, rhs_tile])
                mma_attrs = dict(hybrid_attrs or {})
                if overwrite_first and stage == 0:
                    mma_attrs["overwrite_accumulator"] = True
                mmas.append(MatmulOp.build(
                    lhs_tile.result(), rhs_tile.result(), accumulator.result(),
                    attrs=mma_attrs or None, controls=hybrid_controls,
                ))
            mma_attrs = {
                **issue_attrs,
                **({"derive_tcgen05_descriptors": True}
                   if lean_bookkeeping else {}),
                **({"branch_elected_issue": True}
                   if branch_tcgen05_issue else {}),
            }
            ops.append(AsyncOp.build(
                mma_lock.result(), body=Block(ops=mmas),
                attrs=mma_attrs,
            ))

            if issue_next_a0:
                next_offset = IndexConstantOp.build(2)
                next_k = IndexAddOp.build(first_k.result(), next_offset.result())
                next_a_tile = TileOp.build(
                    a_block.result(), [0, next_k.result()],
                    GmemTensorType.build(
                        [output_m_tile, 64], bf16, strides=[k, 1],
                    ),
                )
                ops.extend([
                    next_offset,
                    next_k,
                    next_a_tile,
                    AsyncOp.build(
                        (
                            tma_lock.result() if lean_bookkeeping else
                            prefetch_tma_lock.result()
                        ),
                        body=Block(ops=[MoveOp.build(
                            free_buffer, next_a_tile.result(),
                            attrs=lhs_move_attrs,
                        )]),
                        attrs={
                            **issue_attrs,
                            # A TMA transaction may name an initialized but
                            # not-yet-armed mbarrier. B0 later supplies the one
                            # arrive whose expected byte count covers both
                            # transactions; no warp gains a persistent role.
                            **({"defer_tma_arrival": True}
                               if lean_bookkeeping else {}),
                        },
                    ),
                ])
                if prefetch_rhs_next:
                    assert half_mma_lock is not None
                    # Once MMAs 1-4 complete, B0 is dead for this iteration.
                    # Refill it for the next K128 step while MMAs 5-8 consume
                    # B1. Every warp observes the same completion wait and
                    # software-pipeline order; only TMA's required leader
                    # issues the two copies.
                    ops.append(AwaitOp.build(half_mma_lock.result()))
                    next_rhs_moves = []
                    for n_half in range(2):
                        next_bt_tile = TileOp.build(
                            bt_block.result(), [n_half, next_k.result()],
                            GmemTensorType.build(
                                [output_n_tile // 2, 64], bf16,
                                strides=[k, 1],
                            ),
                        )
                        next_rhs_tile = TileOp.build(
                            rhs_smem[0].result(), [n_half, 0],
                            SmemTensorType.build(
                                [output_n_tile // 2, 64], bf16,
                                strides=[64, 1],
                                swizzle="128B",
                                attrs={"tcgen05_matrix": True},
                            ),
                        )
                        ops.extend([next_bt_tile, next_rhs_tile])
                        next_rhs_moves.append(MoveOp.build(
                            next_rhs_tile.result(), next_bt_tile.result(),
                            attrs=rhs_move_attrs,
                        ))
                    ops.append(AsyncOp.build(
                        tma_lock.result(), body=Block(ops=next_rhs_moves),
                        attrs={
                            **issue_attrs,
                            **({"tma_extra_expected_bytes": 16384}
                               if lean_bookkeeping else {}),
                        },
                    ))
            ops.append(AwaitOp.build(mma_lock.result()))
            return ops

        # Peel the first and last K128 iterations. The middle 30 iterations are
        # grouped in threes, making the cyclic buffer choice static while
        # keeping the emitted loop compact enough for ptxas.
        first = IndexConstantOp.build(0)
        mainloop_ops.append(first)
        mainloop_ops.extend(prefetch_iteration(
            first.result(),
            lhs_smem[0].result(), lhs_smem[1].result(), lhs_smem[2].result(),
            load_a0=True, issue_next_a0=True,
            overwrite_first=persistent_n_pairs,
        ))

        iteration_count = k // 128
        middle_count = iteration_count - 2
        grouped_middle_count = middle_count - middle_count % 3
        grouped_middle_end = 1 + grouped_middle_count
        middle = ForOp.build(1, grouped_middle_end, step=3)
        middle.attrs["label"] = "prefetch_lhs_middle"
        phases = (
            (lhs_smem[2].result(), lhs_smem[0].result(), lhs_smem[1].result()),
            (lhs_smem[1].result(), lhs_smem[2].result(), lhs_smem[0].result()),
            (lhs_smem[0].result(), lhs_smem[1].result(), lhs_smem[2].result()),
        )
        for offset, buffers in enumerate(phases):
            if offset == 0:
                iteration = middle.iv
            else:
                offset_op = IndexConstantOp.build(offset)
                iteration_op = IndexAddOp.build(middle.iv, offset_op.result())
                middle.body.ops.extend([offset_op, iteration_op])
                iteration = iteration_op.result()
            middle.body.ops.extend(prefetch_iteration(
                iteration, *buffers, load_a0=False, issue_next_a0=True,
            ))
        mainloop_ops.append(middle)

        for iteration_value in range(grouped_middle_end, iteration_count - 1):
            iteration = IndexConstantOp.build(iteration_value)
            mainloop_ops.append(iteration)
            buffers = (
                (lhs_smem[0].result(), lhs_smem[1].result(), lhs_smem[2].result()),
                (lhs_smem[2].result(), lhs_smem[0].result(), lhs_smem[1].result()),
                (lhs_smem[1].result(), lhs_smem[2].result(), lhs_smem[0].result()),
            )[iteration_value % 3]
            mainloop_ops.extend(prefetch_iteration(
                iteration.result(), *buffers,
                load_a0=False, issue_next_a0=True,
            ))

        last = IndexConstantOp.build(iteration_count - 1)
        mainloop_ops.append(last)
        # k/128 is constrained to at least two; 31 mod 3 is one for the
        # selected 4096 kernel, and the generic mapping below handles any tail.
        last_phase = (iteration_count - 1) % 3
        last_buffers = (
            (lhs_smem[0].result(), lhs_smem[1].result(), lhs_smem[2].result()),
            (lhs_smem[2].result(), lhs_smem[0].result(), lhs_smem[1].result()),
            (lhs_smem[1].result(), lhs_smem[2].result(), lhs_smem[0].result()),
        )[last_phase]
        mainloop_ops.extend(prefetch_iteration(
            last.result(), *last_buffers,
            load_a0=False, issue_next_a0=False,
        ))
    elif tma_staging == "direct":
        direct_buffers = operand_stages // 4
        lhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [output_m_tile, 64], bf16, swizzle="128B",
                attrs={"tcgen05_matrix": True},
            ))
            for _ in range(direct_buffers + int(prefetch_lhs_next))
        ]
        rhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [output_n_tile, 64], bf16, swizzle="128B",
                attrs={"tcgen05_matrix": True},
            ))
            for _ in range(direct_buffers)
        ]
    else:
        lhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [128, 16], bf16, swizzle="32B",
                attrs={"tcgen05_layout": "lhs_128x16"},
            ))
            for _ in range(operand_stages)
        ]
        rhs_smem = [
            AllocOp.build_with_fresh_id(SmemTensorType.build(
                [64, 16], bf16, swizzle="128B",
                attrs={"tcgen05_layout": "rhs_64x16"},
            ))
            for _ in range(operand_stages)
        ]
    lhs_tma_smem = [
        AllocOp.build_with_fresh_id(SmemTensorType.build([128, 16], bf16))
        for _ in range(operand_stages)
    ] if tma_staging == "repack" else []
    rhs_tma_smem = [
        AllocOp.build_with_fresh_id(SmemTensorType.build([64, 16], bf16))
        for _ in range(operand_stages)
    ] if tma_staging == "repack" else []
    if not prefetch_lhs_next:
        mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
        tma_lock = (
            AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
            if tma_staging == "direct" else None
        )
        second_tma_lock = (
            AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
            if split_tma_barriers else None
        )
        prefetch_tma_lock = None
    if tma_staging == "direct" and not prefetch_lhs_next:
        direct_k_tile = 16 * operand_stages
        loop = ForOp.build(0, k // direct_k_tile)
        load_prefix = []
        load_groups = []
        if operand_stages == 4:
            k_indices = [loop.iv]
        else:
            buffer_count = operand_stages // 4
            count = IndexConstantOp.build(buffer_count)
            first_k = IndexMulOp.build(loop.iv, count.result())
            load_prefix.extend([count, first_k])
            k_indices = [first_k.result()]
            for half in range(1, buffer_count):
                increment = IndexConstantOp.build(half)
                next_k = IndexAddOp.build(first_k.result(), increment.result())
                load_prefix.extend([increment, next_k])
                k_indices.append(next_k.result())
        for half, k_index in enumerate(k_indices):
            a_tile = TileOp.build(
                a_block.result(), [0, k_index],
                GmemTensorType.build(
                    [output_m_tile, 64], bf16, strides=[k, 1],
                ),
            )
            load_prefix.append(a_tile)
            lhs_move = MoveOp.build(
                lhs_smem[half].result(), a_tile.result(),
                attrs={
                    "tma": True,
                    **({"l2_cache_hint": lhs_l2_cache_hint}
                       if lhs_l2_cache_hint else {}),
                    **({"l2_promotion": lhs_l2_promotion}
                       if lhs_l2_promotion else {}),
                    **({"multicast_mask": 0x3, "cluster_leader": True}
                       if cluster_multicast_a else {}),
                },
            )
            rhs_move_attrs = {
                "tma": True,
                **({"l2_cache_hint": rhs_l2_cache_hint}
                   if rhs_l2_cache_hint else {}),
                **({"l2_promotion": rhs_l2_promotion}
                   if rhs_l2_promotion else {}),
            }
            if split_rhs_tma:
                # Two half-width requests complete sooner than one wide
                # request on B200. They share the bank mbarrier, so the
                # software pipeline and its synchronization are unchanged and
                # all four warps still follow the same stages.
                rhs_moves = []
                for n_half in range(2):
                    bt_tile = TileOp.build(
                        bt_block.result(), [n_half, k_index],
                        GmemTensorType.build(
                            [output_n_tile // 2, 64], bf16,
                            strides=[k, 1],
                        ),
                    )
                    rhs_tile = TileOp.build(
                        rhs_smem[half].result(), [n_half, 0],
                        SmemTensorType.build(
                            [output_n_tile // 2, 64], bf16,
                            strides=[64, 1],
                            swizzle="128B",
                            attrs={"tcgen05_matrix": True},
                        ),
                    )
                    load_prefix.extend([bt_tile, rhs_tile])
                    rhs_moves.append(MoveOp.build(
                        rhs_tile.result(), bt_tile.result(),
                        attrs=rhs_move_attrs,
                    ))
            else:
                bt_tile = TileOp.build(
                    bt_block.result(), [0, k_index],
                    GmemTensorType.build(
                        [output_n_tile, 64], bf16, strides=[k, 1],
                    ),
                )
                load_prefix.append(bt_tile)
                rhs_moves = [MoveOp.build(
                    rhs_smem[half].result(), bt_tile.result(),
                    attrs=rhs_move_attrs,
                )]
            load_groups.append([lhs_move, *rhs_moves])
        assert tma_lock is not None
        loop.body.ops.extend(load_prefix)
        if split_tma_barriers:
            assert second_tma_lock is not None and len(load_groups) == 2
            loop.body.ops.extend([
                AsyncOp.build(tma_lock.result(), body=Block(ops=load_groups[0])),
                AsyncOp.build(
                    second_tma_lock.result(), body=Block(ops=load_groups[1]),
                ),
                AwaitOp.build(tma_lock.result()),
            ])
        else:
            loop.body.ops.extend([
                AsyncOp.build(
                    tma_lock.result(),
                    body=Block(ops=[move for group in load_groups for move in group]),
                ),
                AwaitOp.build(tma_lock.result()),
            ])
        mmas = []
        for stage in range(operand_stages):
            if split_tma_barriers and stage == 4:
                assert second_tma_lock is not None
                mmas.append(AwaitOp.build(second_tma_lock.result()))
            buffer = stage // 4
            buffer_stage = stage % 4
            lhs_tile = TileOp.build(
                lhs_smem[buffer].result(), [0, buffer_stage],
                SmemTensorType.build(
                    [output_m_tile, 16], bf16,
                    strides=[64, 1], swizzle="128B",
                    attrs={"tcgen05_tma_layout": True},
                ),
            )
            rhs_tile = TileOp.build(
                rhs_smem[buffer].result(), [0, buffer_stage],
                SmemTensorType.build(
                    [output_n_tile, 16], bf16,
                    strides=[64, 1], swizzle="128B",
                    attrs={"tcgen05_tma_layout": True},
                ),
            )
            loop.body.ops.extend([lhs_tile, rhs_tile])
            hybrid_attrs = (
                {
                    "hybrid_n_tiles": hybrid_n_tiles,
                    "hybrid_n_width": hybrid_n_width,
                    **({"hybrid_split_rows": hybrid_split_rows}
                       if hybrid_split_rows is not None else {}),
                    **({"hybrid_split_tile": hybrid_split_tile}
                       if hybrid_split_tile is not None else {}),
                    **({"hybrid_element_offsets": True}
                       if hybrid_element_offsets else {}),
                }
                if hybrid_n_tiles is not None else None
            )
            mmas.append(MatmulOp.build(
                lhs_tile.result(), rhs_tile.result(), accumulator.result(),
                attrs=hybrid_attrs,
                controls=hybrid_controls,
            ))
    elif tma_staging != "direct":
        loop = ForOp.build(0, k // 16, step=operand_stages)
        mmas = []
        for stage in range(operand_stages):
            if stage == 0:
                k_index = loop.iv
            else:
                increment = IndexConstantOp.build(stage)
                k_index_op = IndexAddOp.build(loop.iv, increment.result())
                loop.body.ops.extend([increment, k_index_op])
                k_index = k_index_op.result()
            a_tile = TileOp.build(
                a_block.result(), [0, k_index],
                GmemTensorType.build([128, 16], bf16, strides=[k, 1]),
            )
            bt_tile = TileOp.build(
                bt_block.result(), [0, k_index],
                GmemTensorType.build([64, 16], bf16, strides=[k, 1]),
            )
            loop.body.ops.extend([a_tile, bt_tile])
            if tma_staging == "repack":
                loop.body.ops.extend([
                    MoveOp.build(
                        lhs_tma_smem[stage].result(), a_tile.result(),
                        attrs={"tma": True},
                    ),
                    MoveOp.build(
                        rhs_tma_smem[stage].result(), bt_tile.result(),
                        attrs={"tma": True},
                    ),
                    MoveOp.build(
                        lhs_smem[stage].result(), lhs_tma_smem[stage].result(),
                    ),
                    MoveOp.build(
                        rhs_smem[stage].result(), rhs_tma_smem[stage].result(),
                    ),
                ])
            else:
                loop.body.ops.extend([
                    MoveOp.build(lhs_smem[stage].result(), a_tile.result()),
                    MoveOp.build(rhs_smem[stage].result(), bt_tile.result()),
                ])
            mmas.append(MatmulOp.build(
                lhs_smem[stage].result(), rhs_smem[stage].result(),
                accumulator.result(),
            ))
    if not prefetch_lhs_next:
        loop.body.ops.extend([
            AsyncOp.build(
                mma_lock.result(),
                body=Block(ops=mmas),
                attrs={"branch_elected_issue": True}
                if branch_tcgen05_issue else None,
            ),
            AwaitOp.build(mma_lock.result()),
        ])
        mainloop_ops = [loop]
    allocation_ops = (
        [
            accumulator,
            mma_lock,
            *([tma_lock] if tma_lock is not None else []),
            *([second_tma_lock] if second_tma_lock is not None else []),
            *([prefetch_tma_lock] if prefetch_tma_lock is not None else []),
            *([half_mma_lock] if half_mma_lock is not None else []),
            *lhs_smem,
            *rhs_smem,
            *lhs_tma_smem,
            *rhs_tma_smem,
        ]
        if prefetch_lhs_next else
        [
            accumulator,
            *lhs_smem,
            *rhs_smem,
            *lhs_tma_smem,
            *rhs_tma_smem,
            mma_lock,
            *([tma_lock] if tma_lock is not None else []),
            *([second_tma_lock] if second_tma_lock is not None else []),
        ]
    )
    output_move = MoveOp.build(
        c_block.result(), accumulator.result(),
        indices=hybrid_controls,
        attrs=(
            {
                "hybrid_n_tiles": hybrid_n_tiles,
                "hybrid_n_width": hybrid_n_width,
                **({"predicated_tail_stores": True}
                   if predicated_tail_stores else {}),
                **({"hybrid_split_rows": hybrid_split_rows}
                   if hybrid_split_rows is not None else {}),
                **({"hybrid_split_tile": hybrid_split_tile}
                   if hybrid_split_tile is not None else {}),
                **({"hybrid_element_offsets": True}
                   if hybrid_element_offsets else {}),
                **({"l1_cache_policy": output_l1_cache_policy}
                   if output_l1_cache_policy else {}),
                **({"l2_eviction_priority": output_l2_eviction_priority}
                   if output_l2_eviction_priority else {}),
                **({"tmem_load_shape": output_tmem_load_shape}
                   if output_tmem_load_shape != "32x32b" else {}),
            }
            if hybrid_n_tiles is not None else (
                {
                    **({"l1_cache_policy": output_l1_cache_policy}
                       if output_l1_cache_policy else {}),
                    **({"l2_eviction_priority": output_l2_eviction_priority}
                       if output_l2_eviction_priority else {}),
                    **({"tmem_load_shape": output_tmem_load_shape}
                       if output_tmem_load_shape != "32x32b" else {}),
                }
                if (
                    output_l1_cache_policy
                    or output_l2_eviction_priority
                    or output_tmem_load_shape != "32x32b"
                )
                else None
            )
        ),
    )
    tile_ops = [
        *grid_index_ops[1:],
        a_block,
        bt_block,
        c_block,
        *mainloop_ops,
        output_move,
    ]
    if persistent_loop is not None:
        # Logical N tiles are disjoint in global memory. The loop attribute
        # communicates that fact to sync insertion while its normal backedge
        # analysis still inserts barriers for reused shared/tensor memory.
        persistent_loop.body.ops.extend(tile_ops)
        body_ops = [
            block_ids,
            *allocation_ops,
            persistent_loop,
            DeallocOp.build(accumulator.result()),
        ]
    else:
        # Preserve the original dominance/order: allocations precede the
        # mainloop and output move in the non-persistent path.
        body_ops = [
            *grid_index_ops,
            a_block,
            bt_block,
            c_block,
            *allocation_ops,
            *mainloop_ops,
            output_move,
            DeallocOp.build(accumulator.result()),
        ]
    body = Block(arguments=[a, bt, c], ops=body_ops)
    return ModuleOp.build(Block(ops=[KernelOp.build(
        "blackwell_tcgen05_4096",
        body,
        grid_dim=(
            (
                m_tiles * n_tiles + split_last_n_tile_rows,
                1,
                1,
            )
            if split_last_n_tile_rows is not None else
            (n_tiles // 2, m_tiles, 1)
            if persistent_n_pairs else
            (n_tiles + int(split_last_n_tile or split_n_tile is not None), m_tiles, 1)
            if transpose_cta_grid else
            (m_tiles, n_tiles, 1)
        ),
        block_dim=(128, 1, 1),
        cluster_dim=(1, 2, 1) if cluster_multicast_a else None,
        ptx_optimization_level=ptx_optimization_level,
        attrs={"hoist_thread_zero_issue": True}
        if lean_bookkeeping else None,
    )]), target=BLACKWELL)


def blackwell_tcgen05_cta_group2_gemm_module(
    *, m: int = 4096, n: int = 4096, k: int = 4096,
) -> ModuleOp:
    """Build a paired-CTA SM100 GEMM with a shared software pipeline.

    Each neighboring CTA pair computes a 256x64 output region. Both CTAs run
    the same TMA/MMA/store sequence; the single-thread issue mandated by
    ``tcgen05.mma.cta_group::2`` is elected from the pair's even CTA.
    """
    if m % 256 or n % 64 or k % 64:
        raise ValueError("group-2 tcgen05 GEMM dimensions must divide 256x64x64")

    bf16 = BF16Type()
    a = Value(name="a", type=GmemTensorType.build([m, k], bf16))
    bt = Value(name="bt", type=GmemTensorType.build([n, k], bf16))
    c = Value(name="c", type=GmemTensorType.build([m, n], bf16))
    block_ids = BlockIdsOp.build()
    two = IndexConstantOp.build(2)
    peer_rank = IndexModOp.build(block_ids.x, two.result())
    a_block = TileOp.build(
        a, [block_ids.x, 0],
        GmemTensorType.build([128, k], bf16, strides=[k, 1]),
    )
    bt_block = TileOp.build(
        bt, [block_ids.y, 0],
        GmemTensorType.build([64, k], bf16, strides=[k, 1]),
    )
    c_block = TileOp.build(
        c, [block_ids.x, block_ids.y],
        GmemTensorType.build([128, 64], bf16, strides=[n, 1]),
    )
    lhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [128, 64], bf16, swizzle="128B", attrs={"tcgen05_matrix": True},
    ))
    rhs_smem = AllocOp.build_with_fresh_id(SmemTensorType.build(
        [32, 64], bf16, swizzle="128B", attrs={"tcgen05_matrix": True},
    ))
    accumulator = AllocOp.build_with_fresh_id(TmemTensorType.build(
        [128, 64], F32Type(), attrs={"cta_group": 2},
    ), init=0.0)
    tma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="mbar"))
    mma_lock = AllocOp.build_with_fresh_id(LockType.build(kind="tcgen05"))
    loop = ForOp.build(0, k // 64)
    a_tile = TileOp.build(
        a_block.result(), [0, loop.iv],
        GmemTensorType.build([128, 64], bf16, strides=[k, 1]),
    )
    bt_tile = TileOp.build(
        bt_block.result(), [peer_rank.result(), loop.iv],
        GmemTensorType.build([32, 64], bf16, strides=[k, 1]),
    )
    loop.body.ops.extend([
        a_tile,
        bt_tile,
        AsyncOp.build(tma_lock.result(), body=Block(ops=[
            MoveOp.build(lhs_smem.result(), a_tile.result(), attrs={"tma": True}),
            MoveOp.build(rhs_smem.result(), bt_tile.result(), attrs={"tma": True}),
        ])),
        AwaitOp.build(tma_lock.result()),
    ])
    mmas = []
    for stage in range(4):
        lhs_tile = TileOp.build(
            lhs_smem.result(), [0, stage],
            SmemTensorType.build(
                [128, 16], bf16, strides=[64, 1], swizzle="128B",
                attrs={"tcgen05_tma_layout": True},
            ),
        )
        rhs_tile = TileOp.build(
            rhs_smem.result(), [0, stage],
            SmemTensorType.build(
                [32, 16], bf16, strides=[64, 1], swizzle="128B",
                attrs={"tcgen05_tma_layout": True},
            ),
        )
        loop.body.ops.extend([lhs_tile, rhs_tile])
        mmas.append(MatmulOp.build(
            lhs_tile.result(), rhs_tile.result(), accumulator.result(),
        ))
    loop.body.ops.extend([
        AsyncOp.build(mma_lock.result(), body=Block(ops=mmas)),
        AwaitOp.build(mma_lock.result()),
    ])
    body = Block(arguments=[a, bt, c], ops=[
        block_ids,
        two,
        peer_rank,
        a_block,
        bt_block,
        c_block,
        accumulator,
        lhs_smem,
        rhs_smem,
        tma_lock,
        mma_lock,
        loop,
        MoveOp.build(c_block.result(), accumulator.result()),
        DeallocOp.build(accumulator.result()),
    ])
    return ModuleOp.build(Block(ops=[KernelOp.build(
        "blackwell_tcgen05_cta_group2_4096",
        body,
        grid_dim=(m // 128, n // 64, 1),
        block_dim=(128, 1, 1),
        cluster_dim=(2, 1, 1),
        ptx_optimization_level=1,
    )]), target=BLACKWELL)


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_tiles_per_cta: int = 1,
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
    ptx_optimization_level: int | None = None,
) -> ModuleOp:
    if output_tiles_per_cta not in (1, 2, 4):
        raise ValueError("output_tiles_per_cta must be 1, 2, or 4")
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if output_m_tile not in (128, 192):
        raise ValueError("output_m_tile must be 128 or 192")
    if output_m_tile == 192 and output_tiles_per_cta != 1:
        raise ValueError("M=192 currently requires one output tile per CTA")
    if output_n_tile not in (256, 384):
        raise ValueError("output_n_tile must be 256 or 384")
    if operand_k_tile not in (64, 128):
        raise ValueError("operand_k_tile must be 64 or 128")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile == 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix"):
        raise ValueError("output_store must be 'scalar' or 'stmatrix'")
    if output_store == "scalar":
        output_tma_inner_size = None
    problem_name = (
        str(problem_size)
        if m == n == k == problem_size
        else f"m{m}_n{n}_k{k}"
    )
    name = f"auto_matmul_{problem_name}_wgmma256_wg_tma_store_k{operand_stages}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if warpgroup_n_tile not in (128, 256):
        name += f"_wg{warpgroup_n_tile}"
    if output_tiles_per_cta != 1:
        name += f"_{output_tiles_per_cta}tiles"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"
    m_blocks = (m + output_m_tile - 1) // output_m_tile
    grid_x = (m_blocks + output_tiles_per_cta - 1) // output_tiles_per_cta
    grid_y = (n + output_n_tile - 1) // output_n_tile
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) grid_dim=[{grid_x}, {grid_y}, 1] attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            matmul(%a: gmem<{m}x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<{m}x{n}xbf16>) attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    if output_tiles_per_cta == 1:
        module = tile_matmul(
            module, tile_dims=["M", "N"],
            tile_sizes=[output_m_tile, output_n_tile],
            tile_levels=["block.x", "block.y"], tile_labels=[None, None],
            label_filter="mm",
            allow_partial=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        )
    else:
        module = tile_matmul(
            module, tile_dims=["M", "M", "N"],
            tile_sizes=[grid_x * output_m_tile, output_m_tile, 256],
            tile_levels=["for", "block.x", "block.y"],
            tile_labels=["outm", None, None], label_filter="mm",
            allow_partial=(
                m % output_m_tile != 0
                or n % output_n_tile != 0
            ),
        )
    wg_dim, wg_size = (
        ("M", 64) if output_n_tile == 256
        else ("N", warpgroup_n_tile)
    )
    module = tile_matmul(
        module, tile_dims=[wg_dim], tile_sizes=[wg_size],
        tile_levels=["wg.x"], tile_labels=[None], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"], swizzle=None,
        layout="wgmma_layout_0", warps=4, tma=True, nc=False,
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
        target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[operand_k_tile],
        tile_levels=["for"], tile_labels=["k"], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None, tma=True, nc=True,
        physical_layout=(
            {
                "block_width": 64,
                "order": "block_major",
            }
            if operand_k_tile > 64 else None
        ),
        tma_load_slices=(
            {"rhs": output_n_tile // warpgroup_n_tile}
            if output_n_tile == 384 else None
        ),
        move_attrs=operand_move_attrs,
        label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["kk"], label_filter="mm",
    )
    module = multi_buffer(
        module, depth=operand_stages, label_filter="mm_lhs_smem",
    )
    module = multi_buffer(
        module, depth=operand_stages, label_filter="mm_rhs_smem",
    )
    module = licm(module)
    module = unroll_loop(module, unroll_factor=operand_stages, label_filter="k")
    module = Canonicalize().run(module)
    module = lower_to_async(module)
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = Canonicalize().run(module)

    module = pipeline_loop(
        module, label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    )
    module = unroll_loop(
        module, unroll_factor=operand_k_tile // 16,
        label_filter=r"kk(?:_tail\d+)?(?:_iter\d+)?",
    )
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = Canonicalize().run(module)

    if output_tiles_per_cta != 1:
        module = unroll_loop(
            module, unroll_factor=output_tiles_per_cta,
            label_filter="outm",
        )
    module = schedule_async(module)
    return Canonicalize().run(module)
