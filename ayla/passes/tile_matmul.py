from __future__ import annotations

from ..gpu_dialect import (
    ForOp,
    IndexType,
    MatmulOp,
    ModuleOp,
    PLoopOp,
    TileOp,
)
from ..base.ir import Block, Op, Value, clone_type
from ..base.lowering_pass import LoweringPass, lowers


def _build_loop(
    tile_level: str, ub: int, body: Block, attrs: dict[str, object] | None,
) -> Op:
    """The loop a tile level iterates as.

    ``for`` is the sequential one; every other level names a hardware axis
    and becomes a ``ploop``, which is what carries the extent to the kernel
    once it is lowered.
    """
    if tile_level == "for":
        return ForOp.build(lb=0, ub=ub, body=body, attrs=attrs)
    level, _, axis = tile_level.partition(".")
    if axis not in PLoopOp.AXES:
        raise ValueError(
            f"TileMatmul: unexpected tile_level {tile_level!r}; expected "
            f"'for' or <level>.<axis>"
        )
    return PLoopOp.build(
        0, ub, level, PLoopOp.AXES.index(axis), body=body, attrs=attrs,
    )


def _wrap_in_loops(
    body_ops: list[Op],
    ivs: list[Value],
    ubs: list[int],
    tile_levels: list[str],
    tile_labels: list[str | None],
) -> list[Op]:
    """Nest ``body_ops`` in one loop per tile level, outer -> inner."""
    for iv, ub, level, label in zip(
        reversed(ivs), reversed(ubs), reversed(tile_levels),
        reversed(tile_labels),
    ):
        body_ops = [_build_loop(
            level, ub,
            Block(arguments=[iv], ops=body_ops),
            {"label": label} if label is not None else None,
        )]
    return body_ops


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


class TileMatmul(LoweringPass):
    """Tile every ``matmul`` along ``tile_dims`` by matching ``tile_sizes``.

    Three parallel lists describe the tiling:
      - ``tile_dims[i]``: one of ``"M" | "N" | "K"`` (matmul shapes
        are lhs=[M, K], rhs=[N, K], dst=[M, N]).
      - ``tile_sizes[i]``: tile size for that dim.
      - ``tile_levels[i]``: where the iteration lives — ``"for"`` for a
        sequential ``for`` loop, or ``"<level>.<axis>"`` for a ``ploop``
        mapped onto that hardware axis.

    Nesting order is outer→inner along the list.

    The surrounding kernel is left alone. A level's extent reaches the launch
    geometry through the ``ploop`` it becomes, once that is lowered.

    When ``collective=True``, the iteration space is derived from
    ``dim_size / tile_size`` as usual and the loops are emitted, but the
    operands are not tiled — the matmul keeps its original shapes and every
    point of the iteration space runs the whole op cooperatively.

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

        # Compute the iteration bounds.
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
            sizes_by_dim[tile_dim] = tile_size

        ivs = [
            Value(name=tile_label, type=IndexType())
            for tile_label in self.tile_labels
        ]

        # Collective tiling leaves the matmul at full shape: the loops carry
        # the iteration space, but every point of it runs the whole op.
        if self.collective:
            matmul = MatmulOp.build(lhs=lhs, rhs=rhs, dst=dst, attrs=op.attrs)
            return _wrap_in_loops(
                [matmul], ivs, ubs, self.tile_levels, self.tile_labels,
            ), []

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
            MatmulOp.build(lhs=lhs, rhs=rhs, dst=dst, attrs=op.attrs),
        )
        return _wrap_in_loops(
            body_ops, ivs, ubs, self.tile_levels, self.tile_labels,
        ), []


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
