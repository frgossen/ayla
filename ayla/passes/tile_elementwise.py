"""Tile element-wise operations over sequential iteration spaces."""
from __future__ import annotations

from ..base.ir import Block, Op, Value, clone_op, clone_type
from ..base.lowering_pass import LoweringPass, lowers
from ..base.pipeline import op_label_matches
from ..gpu_dialect import (
    ElementwiseOp,
    ForOp,
    IndexType,
    ModuleOp,
    MoveOp,
    TileOp,
)


class TileElementwise(LoweringPass):
    """Tile selected element-wise operations along one or more dimensions.

    The four parallel lists follow ``TileMatmul``'s iteration-space model.
    This pass currently supports only sequential ``"for"`` bindings because
    the motivating epilogue is already mapped to blocks and warp groups by
    ``TileMatmul``. Each dimension becomes a nested ``ForOp``; peeling and
    unrolling those loops remain separate transformations.

    Tiles must divide their dimension exactly. Partial and asymmetric tiles
    are deliberately unsupported.
    """

    def __init__(
        self,
        tile_dims: list[int],
        tile_sizes: list[int],
        tile_levels: list[str],
        tile_labels: list[str | None],
        label_filter: str | None = None,
    ) -> None:
        super().__init__()
        if not (
            len(tile_dims)
            == len(tile_sizes)
            == len(tile_levels)
            == len(tile_labels)
        ):
            raise ValueError(
                "TileElementwise: tile_dims, tile_sizes, tile_levels, and "
                "tile_labels must have the same length"
            )
        if not tile_dims:
            raise ValueError("TileElementwise: at least one tile is required")
        if any(size <= 0 for size in tile_sizes):
            raise ValueError("TileElementwise: tile sizes must be positive")
        if any(level != "for" for level in tile_levels):
            raise ValueError(
                'TileElementwise currently supports only the "for" tile level'
            )
        self.tile_dims = list(tile_dims)
        self.tile_sizes = list(tile_sizes)
        self.tile_labels = list(tile_labels)
        self.label_filter = label_filter

    @lowers(ElementwiseOp, MoveOp)
    def lower_elementwise(
        self, op: ElementwiseOp | MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if isinstance(op, MoveOp) and self.label_filter is None:
            return None
        if not op_label_matches(op, self.label_filter):
            return None
        if isinstance(op, MoveOp) and op.indices:
            raise ValueError(
                "TileElementwise cannot tile an indexed move"
            )
        shapes = [list(value.type.sizes) for value in op.operands]
        rank = len(shapes[0])
        if any(len(shape) != rank for shape in shapes[1:]):
            raise ValueError("TileElementwise requires operands of equal rank")

        bounds: list[int] = []
        for dim, tile_size in zip(self.tile_dims, self.tile_sizes):
            if not 0 <= dim < rank:
                raise ValueError(
                    f"TileElementwise: tile dimension {dim} is invalid "
                    f"for rank {rank}"
                )
            extents = [shape[dim] for shape in shapes]
            if any(not isinstance(extent, int) for extent in extents):
                raise ValueError(
                    "TileElementwise requires statically sized tiled dimensions"
                )
            if any(extent != extents[0] for extent in extents[1:]):
                raise ValueError(
                    "TileElementwise requires matching operand extents along "
                    f"dimension {dim}"
                )
            extent = extents[0]
            assert isinstance(extent, int)
            bound, remainder = divmod(extent, tile_size)
            if remainder:
                raise ValueError(
                    f"TileElementwise: dimension {dim} size {extent} is not "
                    f"divisible by tile size {tile_size}"
                )
            bounds.append(bound)
            for shape in shapes:
                shape[dim] = tile_size

        ivs = [
            Value(name=label, type=IndexType()) for label in self.tile_labels
        ]
        body_ops: list[Op] = []
        tiled_values: dict[Value, Value] = {}
        for operand in op.operands:
            if operand in tiled_values:
                continue
            current = operand
            for dim, tile_size, iv in zip(
                self.tile_dims, self.tile_sizes, ivs,
            ):
                typ = clone_type(current.type)
                typ.sizes[dim] = tile_size
                indices: list[Value | int] = [0] * typ.rank
                indices[dim] = iv
                tile = TileOp.build(current, indices, typ)
                body_ops.append(tile)
                current = tile.result()
            tiled_values[operand] = current
        body_ops.append(clone_op(op, tiled_values))

        for iv, bound, label in zip(
            reversed(ivs), reversed(bounds), reversed(self.tile_labels),
        ):
            attrs = {"label": label} if label is not None else None
            body_ops = [
                ForOp.build(
                    lb=0, ub=bound,
                    body=Block(arguments=[iv], ops=body_ops),
                    attrs=attrs,
                )
            ]
        return body_ops, []


def tile_elementwise(
    module: ModuleOp,
    tile_dims: list[int],
    tile_sizes: list[int],
    tile_levels: list[str],
    tile_labels: list[str | None],
    label_filter: str | None = None,
) -> ModuleOp:
    return TileElementwise(
        tile_dims, tile_sizes, tile_levels, tile_labels, label_filter,
    ).run(module)
