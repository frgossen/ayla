from __future__ import annotations

from ..gpu_dialect import (
    BlockIdsOp,
    ForOp,
    IndexType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    ThreadIdsOp,
    TileOp,
    WarpGroupIdOp,
)
from ..base.ir import Block, Op, Value, clone_type
from ..base.lowering_pass import LoweringPass, lowers


class TileMatmul(LoweringPass):
    """Tile every ``matmul`` along ``tile_dims`` by matching ``tile_sizes``.

    Three parallel lists describe the tiling:
      - ``tile_dims[i]``: one of ``"M" | "N" | "K"`` (matmul shapes
        are lhs=[M, K], rhs=[N, K], dst=[M, N]).
      - ``tile_sizes[i]``: tile size for that dim.
      - ``tile_levels[i]``: where the iteration lives — ``"for"`` for a
        sequential ``for`` loop, ``"block.{x,y,z}"`` for ``block_ids()``
        (writes the matching axis of the kernel's ``grid_dim``), or
        ``"thread.{x,y,z}"`` for ``thread_ids()`` (writes the matching
        axis of the kernel's ``block_dim``).

    Nesting order is outer→inner along the list.

    When ``collective=True``, the iteration space is derived from
    ``dim_size / tile_size`` as usual and kernel constraints are set,
    but no tile ops, loops, or id ops are emitted — the matmul passes
    through with its original shapes.

    ``allow_partial`` permits a non-divisible block-level tile by rounding the
    grid dimension up. Addressing remains full-tile; callers must use memory
    operations with defined out-of-bounds behavior, such as TMA.
    """

    def __init__(
        self,
        tile_dims: list[str],
        tile_sizes: list[int],
        tile_levels: list[str],
        tile_labels: list[str | None],
        label_filter: str | None = None,
        collective: bool = False,
        allow_partial: bool = False,
    ) -> None:
        super().__init__()
        self.label_filter = label_filter
        self.collective = collective
        self.allow_partial = allow_partial
        if not (len(tile_dims) == len(tile_sizes) == len(tile_levels) == len(tile_labels)):
            raise ValueError(
                f"TileMatmul: tile_dims, tile_sizes, tile_levels, tile_labels "
                f"must have the same length; got {len(tile_dims)}, "
                f"{len(tile_sizes)}, {len(tile_levels)}, {len(tile_labels)}"
            )
        if collective and "for" in tile_levels:
            raise ValueError(
                "TileMatmul: 'for' tile_level is not supported in "
                "collective mode — collective tiling does not emit loops"
            )
        self.tile_dims = list(tile_dims)
        self.tile_sizes = list(tile_sizes)
        self.tile_levels = list(tile_levels)
        self.tile_labels = list(tile_labels)
        # Per-kernel state: two 3-lists of `int | None` (x/y/z).
        # `grid_dim_constraints` holds the demands from `block.*` levels
        # (writes to kernel.grid_dim). `block_dim_constraints` holds the
        # demands from `thread.*` levels (writes to kernel.block_dim).
        self.grid_dim_constraints: list[int | None]
        self.block_dim_constraints: list[int | None]
        self.num_warp_groups_constraints: list[int | None]

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        self.grid_dim_constraints = [None, None, None]
        self.block_dim_constraints = [None, None, None]
        self.num_warp_groups_constraints = [None]
        for block in op.blocks:
            self.lower_block(block)
        op.grid_dim = _resolve_launch_dim_constraints(
            op.grid_dim, self.grid_dim_constraints,
        )
        op.block_dim = _resolve_launch_dim_constraints(
            op.block_dim, self.block_dim_constraints,
        )
        [op.num_warp_groups] = _resolve_launch_dim_constraints(
            [op.num_warp_groups], self.num_warp_groups_constraints,
        )
        return [op], []

    @lowers(MatmulOp)
    def lower_matmul(
        self, op: MatmulOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not self.tile_dims:
            return None
        if self.label_filter is not None and op.attrs.get("label") != self.label_filter:
            return None

        lhs = self.value_map[op.lhs]
        rhs = self.value_map[op.rhs]
        dst = self.value_map[op.dst]

        sizes_by_dim = {
            "M": lhs.type.sizes[0],
            "N": rhs.type.sizes[0],
            "K": lhs.type.sizes[1],
        }

        # Compute iteration bounds and accumulate kernel constraints.
        ubs: list[int] = []
        for tile_dim, tile_size, tile_level in zip(
            self.tile_dims, self.tile_sizes, self.tile_levels,
        ):
            if tile_dim not in sizes_by_dim:
                raise ValueError(
                    f"TileMatmul: tile_dim must be 'M', 'N', or 'K'; "
                    f"got {tile_dim!r}"
                )
            tile_dim_size = sizes_by_dim[tile_dim]
            is_partial = tile_dim_size % tile_size != 0
            if is_partial and not (
                self.allow_partial
                and (tile_level == "for" or tile_level.startswith("block."))
            ):
                raise ValueError(
                    f"TileMatmul: tile_dim {tile_dim} size "
                    f"({tile_dim_size}) is not divisible by tile "
                    f"size ({tile_size})"
                )
            ub = (tile_dim_size + tile_size - 1) // tile_size
            ubs.append(ub)
            self._accumulate_constraint(tile_level, ub)
            sizes_by_dim[tile_dim] = tile_size

        if self.collective:
            return None

        # -- Non-collective: emit id ops, tile ops, and for-loops. --

        new_ops: list[Op] = []
        block_ids_op: BlockIdsOp | None = None
        thread_ids_op: ThreadIdsOp | None = None
        wg_id_op: WarpGroupIdOp | None = None
        if any(level.startswith("block.") for level in self.tile_levels):
            block_ids_op = BlockIdsOp.build()
            new_ops.append(block_ids_op)
        if any(level.startswith("thread.") for level in self.tile_levels):
            thread_ids_op = ThreadIdsOp.build()
            new_ops.append(thread_ids_op)
            self.num_warp_groups_constraints = _resolve_launch_dim_constraints(
                self.num_warp_groups_constraints, [-1],
            )
        if any(level.startswith("wg.") for level in self.tile_levels):
            wg_id_op = WarpGroupIdOp.build()
            new_ops.append(wg_id_op)

        ivs: list[Value] = []
        for tile_level, tile_label in zip(self.tile_levels, self.tile_labels):
            if tile_level == "for":
                ivs.append(Value(name=tile_label, type=IndexType()))
            elif tile_level == "block.x":
                ivs.append(block_ids_op.x)
            elif tile_level == "block.y":
                ivs.append(block_ids_op.y)
            elif tile_level == "block.z":
                ivs.append(block_ids_op.z)
            elif tile_level == "thread.x":
                ivs.append(thread_ids_op.x)
            elif tile_level == "thread.y":
                ivs.append(thread_ids_op.y)
            elif tile_level == "thread.z":
                ivs.append(thread_ids_op.z)
            elif tile_level == "wg.x":
                ivs.append(wg_id_op.x)
            else:
                raise ValueError(
                    f"TileMatmul: unexpected tile_level {tile_level!r}"
                )

        body_ops: list[Op] = []
        ops, lhs = _build_tile_op(
            lhs, ["M", "K"],
            tile_dims=self.tile_dims,
            tile_idxs=ivs,
            tile_sizes=self.tile_sizes,
        )
        body_ops.extend(ops)
        ops, rhs = _build_tile_op(
            rhs, ["N", "K"],
            tile_dims=self.tile_dims,
            tile_idxs=ivs,
            tile_sizes=self.tile_sizes,
        )
        body_ops.extend(ops)
        ops, dst = _build_tile_op(
            dst, ["M", "N"],
            tile_dims=self.tile_dims,
            tile_idxs=ivs,
            tile_sizes=self.tile_sizes,
        )
        body_ops.extend(ops)
        body_ops.append(
            MatmulOp.build(
                lhs=lhs, rhs=rhs, dst=dst, attrs=op.attrs,
                controls=[self.value_map[value] for value in op.controls],
            ),
        )

        for iv, ub, level, label in zip(
            reversed(ivs), reversed(ubs), reversed(self.tile_levels),
            reversed(self.tile_labels),
        ):
            if level != "for":
                continue
            body = Block(arguments=[iv], ops=body_ops)
            attrs = {"label": label} if label is not None else None
            body_ops = [ForOp.build(lb=0, ub=ub, body=body, attrs=attrs)]
        new_ops.extend(body_ops)
        return new_ops, []

    def _accumulate_constraint(self, tile_level: str, ub: int) -> None:
        if tile_level == "for":
            pass
        elif tile_level == "block.x":
            self.grid_dim_constraints = _resolve_launch_dim_constraints(
                self.grid_dim_constraints, [ub, None, None],
            )
        elif tile_level == "block.y":
            self.grid_dim_constraints = _resolve_launch_dim_constraints(
                self.grid_dim_constraints, [None, ub, None],
            )
        elif tile_level == "block.z":
            self.grid_dim_constraints = _resolve_launch_dim_constraints(
                self.grid_dim_constraints, [None, None, ub],
            )
        elif tile_level == "thread.x":
            self.block_dim_constraints = _resolve_launch_dim_constraints(
                self.block_dim_constraints, [ub, None, None],
            )
        elif tile_level == "thread.y":
            self.block_dim_constraints = _resolve_launch_dim_constraints(
                self.block_dim_constraints, [None, ub, None],
            )
        elif tile_level == "thread.z":
            self.block_dim_constraints = _resolve_launch_dim_constraints(
                self.block_dim_constraints, [None, None, ub],
            )
        elif tile_level == "wg.x":
            self.num_warp_groups_constraints = _resolve_launch_dim_constraints(
                self.num_warp_groups_constraints, [ub],
            )
            self.block_dim_constraints = _resolve_launch_dim_constraints(
                self.block_dim_constraints, [ub * 128, 1, 1],
            )
        elif tile_level in ("wg.y", "wg.z"):
            raise ValueError(
                f"TileMatmul: tile_level {tile_level!r} is not "
                f"supported; warp groups are 1D — use "
                f"'thread.{tile_level[-1]}' instead"
            )
        else:
            raise ValueError(
                f"TileMatmul: unexpected tile_level {tile_level!r}"
            )


def _resolve_launch_dim_constraints(
    a_list: list[int | None], b_list: list[int | None],
) -> list[int | None]:
    """Symmetric per-axis merge of two ``int | None`` lists.

    For each axis: if either side is ``None``, take the other; if they
    agree, take it; otherwise raise.
    """
    assert len(a_list) == len(b_list)
    result: list[int | None] = []
    for a, b in zip(a_list, b_list):
        if a is None:
            result.append(b)
        elif b is None:
            result.append(a)
        elif a == b:
            result.append(a)
        else:
            raise ValueError(
                f"launch dim conflict: {a_list} vs {b_list}"
            )
    return result


def _build_tile_op(
    arg: Value,
    dim_names: list[str],
    *,
    tile_dims: list[str],
    tile_idxs: list[Value],
    tile_sizes: list[int],
) -> tuple[list[Op], Value]:
    """Build TileOps that contract the dims of ``arg`` named in
    ``tile_dims`` to the matching ``tile_sizes`` at the matching
    ``tile_idxs``. Each tiling step produces a new value that
    subsequent steps operate on, so the same dim can be tiled
    multiple times (e.g. K=64 then K=16).
    """
    assert len(dim_names) == len(arg.type.sizes), (
        f"_build_tile_op: dim_names ({dim_names}) must have one entry "
        f"per arg dim ({arg.type.sizes})"
    )
    assert len(tile_dims) == len(tile_idxs) == len(tile_sizes), (
        "_build_tile_op: tile_dims, tile_idxs, tile_sizes must have the "
        "same length"
    )
    name_to_idx = {n: i for i, n in enumerate(dim_names)}
    ops: list[Op] = []
    cur = arg
    maybe_partial_dims: set[str] = set()
    for name, iv, ts in zip(tile_dims, tile_idxs, tile_sizes):
        d = name_to_idx.get(name)
        if d is None:
            continue
        maybe_partial = name in maybe_partial_dims or cur.type.sizes[d] % ts != 0
        if maybe_partial:
            maybe_partial_dims.add(name)
        new_type = clone_type(cur.type)
        indices: list[Value | int] = [0] * len(new_type.sizes)
        indices[d] = iv
        new_type.sizes[d] = ts
        tile_op = TileOp.build(
            cur, indices, new_type, maybe_partial=maybe_partial,
        )
        ops.append(tile_op)
        cur = tile_op.result()
    return ops, cur


def tile_matmul(
    module: ModuleOp,
    tile_dims: list[str],
    tile_sizes: list[int],
    tile_levels: list[str],
    tile_labels: list[str | None],
    label_filter: str | None = None,
    collective: bool = False,
    allow_partial: bool = False,
) -> ModuleOp:
    """Convenience entry point."""
    return TileMatmul(
        tile_dims, tile_sizes, tile_levels, tile_labels,
        label_filter, collective, allow_partial,
    ).run(module)
