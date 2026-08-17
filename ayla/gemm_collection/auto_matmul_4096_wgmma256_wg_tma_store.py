"""Configurable wgmma256 GEMM with pipelined TMA output stores."""
import re
import textwrap

from ..base.ir import find_ops
from ..gpu_dialect import gpu_dialect
from ..gpu_dialect import AsyncOp, KernelOp, MoveOp, TileOp, WarpGroupIdOp
from ..passes.canonicalize import canonicalize
from ..passes.fuse_async import fuse_async
from ..passes.licm import licm
from ..passes.lower_to_async import lower_to_async
from ..passes.pipeline_loop import pipeline_loop
from ..passes.promote_buffers import promote_buffers
from ..passes.schedule_async import schedule_async
from ..passes.tile_matmul import tile_matmul
from ..passes.unroll_loop import unroll_loop
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import PtxJitOptions, make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    output_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    operand_swizzle: str = "128B",
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_tile_group: tuple[int, int] | None = None,
    output_group_outer: str = "n",
    cta_raster: str = "m",
    variant_suffix: str | None = None,
    promote_k_tail: bool = True,
    rhs_output_cache_hints: tuple[str | None, ...] | None = None,
    rhs_output_l2_promotions: tuple[str | None, ...] | None = None,
    k_tail_cache_hints: dict[str, str | None] | None = None,
    reuse_k_tail_mbarrier: bool = False,
    startup_tma_stage_order: tuple[int, ...] | None = None,
    precompute_wgmma_lhs_descriptors: bool = False,
    precompute_wgmma_rhs_descriptors: bool = False,
    direct_warp_group_id: bool = False,
    emit_reqntid: bool = False,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_tma_inner_padding: int = 8,
    output_tma_block_order: str = "row_major",
    regroup_stmatrix_rows: bool = False,
    regroup_stmatrix_block_major: bool = False,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
):
    if output_tiles_per_cta not in (1, 2, 4):
        raise ValueError("output_tiles_per_cta must be 1, 2, or 4")
    if problem_size <= 0:
        raise ValueError("problem_size must be positive")
    if output_tile_group is None:
        output_tile_group = (output_tiles_per_cta, 1)
    if (len(output_tile_group) != 2
            or output_tile_group[0] * output_tile_group[1]
            != output_tiles_per_cta):
        raise ValueError(
            "output_tile_group must be (M tiles, N tiles) whose product "
            "equals output_tiles_per_cta"
        )
    if output_m_tile not in (128, 192):
        raise ValueError("output_m_tile must be 128 or 192")
    if output_m_tile == 192 and output_tiles_per_cta != 1:
        raise ValueError("M=192 currently requires one output tile per CTA")
    if output_n_tile not in (256, 384):
        raise ValueError("output_n_tile must be 256 or 384")
    if operand_k_tile not in (16, 32, 64, 128):
        raise ValueError("operand_k_tile must be 16, 32, 64, or 128")
    if operand_swizzle not in ("32B", "64B", "128B"):
        raise ValueError("unsupported operand swizzle")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile == 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix"):
        raise ValueError("output_store must be 'scalar' or 'stmatrix'")
    if output_tma_block_order not in ("row_major", "block_major"):
        raise ValueError("unsupported output TMA block order")
    if cta_raster not in ("m", "n"):
        raise ValueError("cta_raster must be 'm' or 'n'")
    if output_group_outer not in ("m", "n"):
        raise ValueError("output_group_outer must be 'm' or 'n'")
    if rhs_output_cache_hints is not None:
        if len(rhs_output_cache_hints) != output_tiles_per_cta:
            raise ValueError(
                "rhs_output_cache_hints must have one entry per output tile"
            )
        if any(h not in (None, "evict_first", "evict_last")
               for h in rhs_output_cache_hints):
            raise ValueError("unsupported RHS output-tile cache hint")
    if rhs_output_l2_promotions is not None:
        if len(rhs_output_l2_promotions) != output_tiles_per_cta:
            raise ValueError(
                "rhs_output_l2_promotions must have one entry per output tile"
            )
        if any(p not in (None, "64B", "128B", "256B")
               for p in rhs_output_l2_promotions):
            raise ValueError("unsupported RHS output-tile L2 promotion")
    if k_tail_cache_hints is not None:
        if not set(k_tail_cache_hints).issubset({"lhs", "rhs"}):
            raise ValueError("K-tail cache hints only support lhs and rhs")
        if any(h not in (None, "evict_first", "evict_last")
               for h in k_tail_cache_hints.values()):
            raise ValueError("unsupported K-tail cache hint")
    if reuse_k_tail_mbarrier and (
        operand_stages != 3
        or operand_k_tile != 64
        or (problem_size // operand_k_tile) % operand_stages != 1
    ):
        raise ValueError(
            "K-tail barrier reuse currently requires three K=64 stages and "
            "exactly one static tail stage"
        )
    if (
        startup_tma_stage_order is not None
        and tuple(sorted(startup_tma_stage_order)) != tuple(range(operand_stages))
    ):
        raise ValueError(
            "startup_tma_stage_order must be a permutation of operand stages"
        )
    if output_store == "scalar":
        output_tma_inner_size = None
        output_tma_inner_padding = 0
    if regroup_stmatrix_block_major:
        regroup_stmatrix_rows = True
        output_tma_block_order = "block_major"
    if regroup_stmatrix_rows:
        if output_store != "stmatrix":
            raise ValueError("STMatrix regrouping requires STMatrix output")
        output_tma_inner_size = 64
        output_tma_inner_padding = 0
    name = f"auto_matmul_{problem_size}_wgmma256_wg_tma_store_k{operand_stages}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if operand_swizzle != "128B":
        name += f"_swizzle{operand_swizzle.lower()}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if warpgroup_n_tile not in (128, 256):
        name += f"_wg{warpgroup_n_tile}"
    if output_tiles_per_cta != 1:
        name += f"_{output_tiles_per_cta}tiles"
    if output_tile_group != (output_tiles_per_cta, 1):
        name += f"_group{output_tile_group[0]}x{output_tile_group[1]}"
        if output_group_outer != "n":
            name += f"_{output_group_outer}outer"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"
    if cta_raster != "m":
        name += f"_raster_{cta_raster}"
    if variant_suffix is not None:
        name += f"_{variant_suffix}"
    m_blocks = (problem_size + output_m_tile - 1) // output_m_tile
    n_blocks = (problem_size + output_n_tile - 1) // output_n_tile
    group_m, group_n = output_tile_group
    grid_x = (m_blocks + group_m - 1) // group_m
    grid_y = (n_blocks + group_n - 1) // group_n
    launch_grid = (
        (grid_x, grid_y) if cta_raster == "m" else (grid_y, grid_x)
    )
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem_tensor<{problem_size}x{problem_size}xbf16>, %b: gmem_tensor<{problem_size}x{problem_size}xbf16>, %c: gmem_tensor<{problem_size}x{problem_size}xbf16>) grid_dim=[{launch_grid[0]}, {launch_grid[1]}, 1] block_dim=[128, 1, 1] {{
            matmul(%a: gmem_tensor<{problem_size}x{problem_size}xbf16>, %b: gmem_tensor<{problem_size}x{problem_size}xbf16>, %c: gmem_tensor<{problem_size}x{problem_size}xbf16>) attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    if output_tiles_per_cta == 1:
        block_levels = (
            ["block.x", "block.y"]
            if cta_raster == "m" else ["block.y", "block.x"]
        )
        module = tile_matmul(
            module, tile_dims=["M", "N"],
            tile_sizes=[output_m_tile, output_n_tile],
            tile_levels=block_levels, tile_labels=[None, None],
            label_filter="mm",
            allow_partial=(output_m_tile == 192 or output_n_tile == 384),
        )
    elif group_n == 1:
        block_levels = (
            ["for", "block.x", "block.y"]
            if cta_raster == "m" else ["for", "block.y", "block.x"]
        )
        module = tile_matmul(
            module, tile_dims=["M", "M", "N"],
            tile_sizes=[grid_x * output_m_tile, output_m_tile, 256],
            tile_levels=block_levels,
            tile_labels=["outm", None, None], label_filter="mm",
            allow_partial=output_m_tile == 192,
        )
    else:
        # Keep 128 CTAs, but make each simultaneously executing phase a
        # group_m*grid_x by grid_y rectangle of nearby output tiles.  For the
        # 2x2 four-tile configuration this is the 16x8 grouping commonly used
        # to improve concurrent A/B reuse in L2, rather than the default 8x16.
        m_block, n_block = (
            ("block.x", "block.y")
            if cta_raster == "m" else ("block.y", "block.x")
        )
        if output_group_outer == "n":
            group_dims = ["N", "N", "M", "M"]
            group_sizes = [grid_y * output_n_tile, output_n_tile,
                           grid_x * output_m_tile, output_m_tile]
            group_levels = ["for", n_block, "for", m_block]
            group_labels = ["outn", None, "outm", None]
        else:
            group_dims = ["M", "M", "N", "N"]
            group_sizes = [grid_x * output_m_tile, output_m_tile,
                           grid_y * output_n_tile, output_n_tile]
            group_levels = ["for", m_block, "for", n_block]
            group_labels = ["outm", None, "outn", None]
        module = tile_matmul(
            module, tile_dims=group_dims, tile_sizes=group_sizes,
            tile_levels=group_levels, tile_labels=group_labels,
            label_filter="mm",
            allow_partial=(output_m_tile == 192 or output_n_tile == 384),
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
        module, target_type="wrmem", operands=["dst"],
        swizzle="128B" if regroup_stmatrix_rows else None,
        layout="wgmma_layout_0", warps=4, tma=True, nc=False,
        physical_layout=(
            {
                "kind": "blocked_padding", "block": output_tma_inner_size,
                "padding": output_tma_inner_padding,
                **(
                    {"order": "block_major"}
                    if output_tma_block_order == "block_major" else {}
                ),
            }
            if output_tma_inner_size is not None else None
        ),
        writeback_slices=output_slices,
        writeback_buffers=output_store_buffers,
        # C is written once and never re-read, so an evict-first output store
        # keeps A/B resident in L2 instead of being displaced by the epilogue.
        move_attrs=operand_move_attrs,
        target_dtype="f32", label_filter="mm",
    )
    if regroup_stmatrix_rows:
        for move in find_ops(module, lambda op: isinstance(op, MoveOp)):
            if "_dst_smem" in str(move.attrs.get("label", "")):
                move.attrs["regroup_rows"] = True
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[operand_k_tile],
        tile_levels=["for"], tile_labels=["k"], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle=operand_swizzle, layout=None, warps=None, tma=True, nc=True,
        physical_layout=(
            {"kind": "blocked_padding", "block": 64, "padding": 0,
             "order": "block_major"}
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
    module = unroll_loop(module, unroll_factor=operand_stages, label_filter="k")
    module = lower_to_async(module)
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    module = pipeline_loop(
        module, label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    )
    module = unroll_loop(
        module, unroll_factor=operand_k_tile // 16,
        # The `k` unroll above leaves a `_tail0` copy when the trip count is not
        # divisible by operand_stages; its `kk` loop must be unrolled too, or it
        # keeps a fence/commit/wait_group per single wgmma instead of one group.
        label_filter=r"kk(?:_tail\d+)?(?:_iter\d+)?",
    )
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    if group_n > 1 and output_group_outer == "n":
        # Fully flatten both short grouping loops.  Keeping two nested
        # single-iteration loops after partial unrolling gives the sync pass
        # two artificial back edges for the same reusable TMA barriers.
        module = unroll_loop(
            module, unroll_factor=group_n + 1,
            label_filter="outn",
        )
    if group_m > 1:
        module = unroll_loop(
            module,
            unroll_factor=(group_m + 1 if group_n > 1 else group_m),
            label_filter=(
                r"outm(?:_tail\d+)?"
                if group_n > 1 and output_group_outer == "n" else "outm"
            ),
        )
    if group_n > 1 and output_group_outer == "m":
        module = unroll_loop(
            module, unroll_factor=group_n + 1,
            label_filter=r"outn(?:_tail\d+)?",
        )
    # Once the output loop is static, cache policy can be specialized per
    # issued TMA rather than uniformly per operand.
    for move in find_ops(module, lambda op: isinstance(op, MoveOp)):
        label = str(move.attrs.get("label", ""))
        if rhs_output_l2_promotions is not None and "_rhs_smem_" in label:
            output_iter = re.search(r"_iter(\d+)$", label)
            if output_iter is not None:
                promotion = rhs_output_l2_promotions[int(output_iter.group(1))]
                if promotion is None:
                    move.attrs.pop("l2_promotion", None)
                else:
                    move.attrs["l2_promotion"] = promotion
        # A tail-wide override is more specific than an output-tile policy.
        if not promote_k_tail and "_smem_tail" in label:
            move.attrs.pop("l2_promotion", None)
        if rhs_output_cache_hints is not None and "_rhs_smem_" in label:
            output_iter = re.search(r"_iter(\d+)$", label)
            if output_iter is not None:
                hint = rhs_output_cache_hints[int(output_iter.group(1))]
                if hint is None:
                    move.attrs.pop("l2_cache_hint", None)
                else:
                    move.attrs["l2_cache_hint"] = hint
        # The K-tail override is the most specific policy and therefore wins
        # over an output-tile-wide RHS policy above.
        if k_tail_cache_hints is not None and "_smem_tail" in label:
            operand = re.search(r"_(lhs|rhs)_smem_", label)
            if operand is not None and operand.group(1) in k_tail_cache_hints:
                hint = k_tail_cache_hints[operand.group(1)]
                if hint is None:
                    move.attrs.pop("l2_cache_hint", None)
                else:
                    move.attrs["l2_cache_hint"] = hint
    module = schedule_async(module)
    if reuse_k_tail_mbarrier:
        # K=4096 with a 3-stage, K=64 pipeline leaves one static K tile.  Its
        # TMA load is issued only after stage 0's final consumer has completed,
        # so it can use stage 0's transaction barrier.  This removes the
        # otherwise dedicated tail barrier; the remaining stage barriers share
        # one batched initialization rendezvous in PTX lowering.
        stage0_lock = None
        tail_lock = None
        for async_op in find_ops(module, lambda op: isinstance(op, AsyncOp)):
            label = str(async_op.attrs.get("label", ""))
            if "_smem_iter0_async" in label:
                stage0_lock = async_op.lock
            elif "_smem_tail0_async" in label:
                tail_lock = async_op.lock
        if stage0_lock is None or tail_lock is None:
            raise ValueError("K-tail/stage-0 TMA barriers were not found")
        for user, operand_index in list(tail_lock.uses):
            user.operands[operand_index] = stage0_lock
    module = canonicalize(module)
    if startup_tma_stage_order is not None:
        kernel = next(
            op for op in module.body.entry_block.ops if isinstance(op, KernelOp)
        )
        block = kernel.body.entry_block
        stage_pairs: dict[int, tuple[TileOp, AsyncOp]] = {}
        for index, op in enumerate(block.ops):
            if not isinstance(op, AsyncOp):
                continue
            match = re.fullmatch(
                r"mm_lhs_smem_iter(\d+)_async_iter0",
                str(op.attrs.get("label", "")),
            )
            if match is None:
                continue
            stage = int(match.group(1))
            coordinate = block.ops[index - 1]
            if not isinstance(coordinate, TileOp):
                raise ValueError("startup TMA load is not preceded by its tile")
            stage_pairs[stage] = (coordinate, op)
        if set(stage_pairs) != set(startup_tma_stage_order):
            raise ValueError(
                "could not find every startup TMA stage load: "
                f"found {sorted(stage_pairs)}, expected "
                f"{sorted(startup_tma_stage_order)}"
            )
        members = [member for pair in stage_pairs.values() for member in pair]
        indices = sorted(block.ops.index(member) for member in members)
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("startup TMA stage loads are not contiguous")
        block.ops[indices[0]:indices[-1] + 1] = [
            member
            for stage in startup_tma_stage_order
            for member in stage_pairs[stage]
        ]
    if precompute_wgmma_lhs_descriptors:
        for async_op in find_ops(module, lambda op: isinstance(op, AsyncOp)):
            if async_op.lock.type.kind == "wgmma":
                async_op.attrs["precompute_lhs_descriptors"] = True
    if precompute_wgmma_rhs_descriptors:
        for async_op in find_ops(module, lambda op: isinstance(op, AsyncOp)):
            if async_op.lock.type.kind == "wgmma":
                async_op.attrs["precompute_rhs_descriptors"] = True
    if direct_warp_group_id:
        for warp_group_id in find_ops(
            module, lambda op: isinstance(op, WarpGroupIdOp),
        ):
            warp_group_id.attrs["direct_from_thread_x"] = True
    if emit_reqntid:
        [kernel] = find_ops(module, lambda op: isinstance(op, KernelOp))
        kernel.attrs["emit_reqntid"] = True
    return module


def best_auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, precompute_wgmma_lhs_descriptors: bool = True,
    precompute_wgmma_rhs_descriptors: bool = False,
    direct_warp_group_id: bool = True,
    emit_reqntid: bool = True,
):
    """Build the currently retained best 4096x4096 BF16 GEMM.

    Keep the winning configuration here so runtime registration, standalone
    PTX generation, and analysis tools cannot silently drift apart.
    """
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        output_tiles_per_cta=4,
        operand_stages=3,
        output_slices=1,
        output_tma_inner_padding=0,
        output_store_buffers=1,
        regroup_stmatrix_block_major=True,
        reuse_k_tail_mbarrier=True,
        startup_tma_stage_order=(0, 2, 1),
        precompute_wgmma_lhs_descriptors=precompute_wgmma_lhs_descriptors,
        precompute_wgmma_rhs_descriptors=precompute_wgmma_rhs_descriptors,
        direct_warp_group_id=direct_warp_group_id,
        emit_reqntid=emit_reqntid,
        variant_suffix="full_output_store",
        operand_move_attrs={
            "lhs": {
                "l2_cache_hint": "evict_first",
                "l2_promotion": "256B",
            },
            "dst": {"l2_cache_hint": "evict_first"},
        },
    )


# The CUDA Driver defaults to optimization level 4.  Paired H100 measurements
# of this already-explicit PTX consistently favor level 1 by about 0.06-0.14%
# without changing registers, local memory, or numerical output.
BEST_AUTO_MATMUL_4096_PTX_JIT_OPTIONS = PtxJitOptions(optimization_level=1)


def _register_auto_matmul_4096_wgmma256_wg_tma_store() -> None:
    configs = [
        {},
        {"output_tiles_per_cta": 2},
        {"output_tiles_per_cta": 2, "operand_stages": 3, "output_slices": 2,
         "output_tma_inner_padding": 16, "output_store_buffers": 2},
        {"output_tiles_per_cta": 4, "operand_stages": 3,
         "output_slices": 2, "output_tma_inner_padding": 0,
         "output_store_buffers": 2,
         "regroup_stmatrix_block_major": True,
         "reuse_k_tail_mbarrier": True,
         "operand_move_attrs": {
             "lhs": {"l2_cache_hint": "evict_first"},
             "dst": {"l2_cache_hint": "evict_first"},
         }},
        None,  # The retained best, built by the canonical helper above.
        {"output_tiles_per_cta": 4, "operand_stages": 3,
         "output_slices": 2, "output_tma_inner_padding": 0,
         "output_store_buffers": 2,
         "regroup_stmatrix_block_major": True,
         "reuse_k_tail_mbarrier": True,
         "variant_suffix": "promote_256b",
         "operand_move_attrs": {
             "lhs": {"l2_cache_hint": "evict_first",
                     "l2_promotion": "256B"},
             "rhs": {"l2_promotion": "256B"},
             "dst": {"l2_cache_hint": "evict_first"},
         }},
        {"output_tiles_per_cta": 4, "operand_stages": 3, "output_slices": 2,
         "output_tma_inner_padding": 16, "output_store_buffers": 2,
         "cta_raster": "n",
         "operand_move_attrs": {"lhs": {"l2_cache_hint": "evict_first"},
                                "dst": {"l2_cache_hint": "evict_first"}}},
        {"output_tiles_per_cta": 4, "output_tile_group": (2, 2),
         "operand_stages": 3, "output_slices": 2,
         "output_tma_inner_padding": 16, "output_store_buffers": 2,
         "operand_move_attrs": {"lhs": {"l2_cache_hint": "evict_first"},
                                "dst": {"l2_cache_hint": "evict_first"}}},
        {"output_tiles_per_cta": 4, "output_tile_group": (2, 2),
         "output_group_outer": "m", "operand_stages": 3,
         "output_slices": 2, "output_tma_inner_padding": 16,
         "output_store_buffers": 2,
         "operand_move_attrs": {"lhs": {"l2_cache_hint": "evict_first"},
                                "dst": {"l2_cache_hint": "evict_first"}}},
        {"output_tiles_per_cta": 2, "operand_stages": 2,
         "operand_k_tile": 128, "output_slices": 2},
        {"output_tiles_per_cta": 2, "operand_stages": 4, "output_slices": 2},
        {"output_n_tile": 384, "operand_stages": 2, "output_slices": 3},
        {"output_m_tile": 192, "operand_stages": 3, "output_slices": 2},
        {"output_tiles_per_cta": 2, "output_store": "scalar"},
    ]
    for config in configs:
        module = (
            best_auto_matmul_4096_wgmma256_wg_tma_store_module()
            if config is None
            else auto_matmul_4096_wgmma256_wg_tma_store_module(**config)
        )
        kernel = make_ptx_kernel(module)
        make_torch_custom_call(
            make_ptx_launcher(
                kernel,
                jit_options=(
                    BEST_AUTO_MATMUL_4096_PTX_JIT_OPTIONS
                    if config is None else None
                ),
            ),
            [OperandKind.R, OperandKind.R, OperandKind.RESULT],
        )


_register_auto_matmul_4096_wgmma256_wg_tma_store()
