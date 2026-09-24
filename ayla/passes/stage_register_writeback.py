"""Materialize register-to-global writes through shared memory when requested."""
from __future__ import annotations

from ..base.ir import Op, Value, clone_type
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AllocOp,
    CollectiveLevel,
    ElementwiseOp,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    WrmemTensorType,
)
from ._utils import (
    build_slice_tile,
    label_matches,
    tile_introduces_narrow_level,
)


def _rebuild_noncolumn_tile(tile: TileOp, new_arg: Value) -> TileOp | None:
    """Replay the row portion of a destination tile on a sliced SMEM tile."""
    tile_type = tile.result().type
    arg_type = tile.arg.type
    new_type = clone_type(new_arg.type)
    indices = list(tile.indices)
    changed = False
    for dim in range(new_type.rank - 1):
        if tile_type.sizes[dim] != arg_type.sizes[dim]:
            new_type.sizes[dim] = tile_type.sizes[dim]
            changed = True
    indices[-1] = 0
    if not changed:
        return None
    return TileOp.build(
        new_arg, indices, new_type, maybe_partial=tile.maybe_partial,
    )


def _column_owner_index(
    column_tiles: list[TileOp], slice_cols: int,
) -> tuple[list[Op], Value]:
    """Return the output-slice coordinate encoded by nested column tiles."""
    ops: list[Op] = []
    terms: list[Value] = []
    constant = 0
    for tile in column_tiles:
        width = tile.result().type.sizes[-1]
        if width % slice_cols:
            raise ValueError(
                "warpgroup-column tiles must align with writeback slices"
            )
        scale = width // slice_cols
        index = tile.indices[-1]
        if isinstance(index, int):
            constant += index * scale
            continue
        term = index
        if scale != 1:
            scale_op = IndexConstantOp.build(scale)
            scaled = IndexMulOp.build(index, scale_op.result())
            ops.extend([scale_op, scaled])
            term = scaled.result()
        terms.append(term)

    if not terms:
        raise ValueError(
            "warpgroup-column writeback needs a dynamic column owner"
        )
    owner = terms[0]
    for term in terms[1:]:
        added = IndexAddOp.build(owner, term)
        ops.append(added)
        owner = added.result()
    if constant:
        constant_op = IndexConstantOp.build(constant)
        added = IndexAddOp.build(owner, constant_op.result())
        ops.extend([constant_op, added])
        owner = added.result()
    return ops, owner


class StageRegisterWriteback(LoweringPass):
    """Materialize selected 2-D WGMMA-register-to-global moves.

    A move marked ``tma`` is the semantic live-out. If its source and
    destination dtypes differ, conversion remains in registers before the
    move is staged through shared memory because TMA cannot read registers
    directly.

    The TMA path slices only the final tensor dimension and supports either
    one shared-memory staging buffer or one buffer per slice. Indexed moves
    and other register layouts are deliberately unsupported.
    """

    def __init__(
        self,
        *,
        writeback_slices: int = 1,
        writeback_buffers: int = 1,
        physical_layout: dict[str, object] | None = None,
        unsafe_reuse_smem: bool = False,
        label_filter: str | None = None,
    ) -> None:
        super().__init__(
            writeback_slices=writeback_slices,
            writeback_buffers=writeback_buffers,
            physical_layout=physical_layout,
            unsafe_reuse_smem=unsafe_reuse_smem,
            label_filter=label_filter,
        )
        if writeback_slices < 1:
            raise ValueError("writeback_slices must be positive")
        if writeback_buffers not in (1, writeback_slices):
            raise ValueError(
                "writeback_buffers must be 1 or equal writeback_slices"
            )

    @lowers(MoveOp)
    def lower_move(self, op: MoveOp) -> tuple[list[Op], list[object]] | None:
        if (
            not label_matches(op, self.label_filter)
            or not op.attrs.get("tma", False)
        ):
            return None
        dst = self.value_map[op.dst]
        src = self.value_map[op.src]
        assert isinstance(dst, Value) and isinstance(src, Value)
        if not (
            isinstance(dst.type, GmemTensorType)
            and isinstance(src.type, WrmemTensorType)
        ):
            return None
        if op.indices:
            raise NotImplementedError(
                "StageRegisterWriteback does not support indexed moves"
            )

        label = op.attrs.get("label")
        owner = (
            label.removesuffix("_dst_wrmem")
            if isinstance(label, str) else None
        )
        new_ops: list[Op] = []
        if src.type.dtype != dst.type.dtype:
            converted = AllocOp.build_with_fresh_id(WrmemTensorType.build(
                list(src.type.sizes), dst.type.dtype,
                warps=src.type.warps, layout=src.type.layout,
            ))
            convert_attrs = (
                None if owner is None else {"label": f"{owner}_dst_convert"}
            )
            new_ops.extend([
                converted,
                ElementwiseOp.build(
                    converted.result(), [src], "convert", attrs=convert_attrs,
                ),
            ])
            src = converted.result()

        attrs = {
            key: value for key, value in op.attrs.items()
            if key not in ("collective", "tma", "label")
        }

        block_dst = dst
        narrow_tiles: list[TileOp] = []
        while isinstance(block_dst.definition, TileOp):
            tile = block_dst.definition
            if not tile_introduces_narrow_level(
                tile, CollectiveLevel.BLOCK,
            ):
                break
            narrow_tiles.append(tile)
            block_dst = tile.arg
        assert isinstance(block_dst.type, GmemTensorType)

        smem_sizes = list(block_dst.type.sizes)
        if smem_sizes[-1] % self.writeback_slices:
            raise ValueError(
                "output columns must divide evenly into writeback_slices"
            )
        smem_sizes[-1] //= self.writeback_slices
        layout = dict(self.physical_layout or {})
        swizzle = layout.pop("swizzle", None)
        smem_type = SmemTensorType.build(
            smem_sizes, block_dst.type.dtype, swizzle=swizzle,
            attrs={"physical_layout": layout} if layout else None,
        )
        alloc_attrs = (
            {"reuse_group": f"{owner}_writeback"}
            if self.unsafe_reuse_smem and owner is not None else None
        )
        smems = [
            AllocOp.build_with_fresh_id(smem_type, attrs=alloc_attrs)
            for _ in range(self.writeback_buffers)
        ]
        new_ops.extend(smems)

        slice_cols = smem_sizes[-1]
        wg_column_tiles = [
            tile for tile in narrow_tiles
            if tile.result().type.sizes[-1] != tile.arg.type.sizes[-1]
        ]
        expected_slices = block_dst.type.sizes[-1] // src.type.sizes[-1]
        if wg_column_tiles and self.writeback_slices != expected_slices:
            raise ValueError(
                "warpgroup-column writeback needs one slice per warpgroup"
            )
        writer_ops: list[Op] = []
        writer_index: Value | None = None
        if wg_column_tiles:
            writer_ops, writer_index = _column_owner_index(
                list(reversed(wg_column_tiles)), slice_cols,
            )
            new_ops.extend(writer_ops)

        for slice_index in range(self.writeback_slices):
            smem = smems[slice_index if self.writeback_buffers > 1 else 0]
            gather_dst = smem.result()
            for tile in reversed(narrow_tiles):
                rebuilt = _rebuild_noncolumn_tile(tile, gather_dst)
                if rebuilt is None:
                    continue
                new_ops.append(rebuilt)
                gather_dst = rebuilt.result()
            gather_attrs = (
                {} if owner is None else {"label": f"{owner}_dst_smem"}
            )
            gather_indices = None
            if writer_index is not None:
                gather_attrs["writer_wg"] = slice_index
                gather_indices = [writer_index]
            elif self.writeback_slices > 1:
                # TODO: Express this source slice with a tile op instead of the
                # wgmma_col_start attribute.
                gather_attrs["wgmma_col_start"] = slice_index * slice_cols
            new_ops.append(MoveOp.build(
                gather_dst, src, gather_indices,
                collective=op.collective, attrs=gather_attrs,
            ))

            store_tiles = (
                [build_slice_tile(block_dst, -1, slice_index, slice_cols)]
                if self.writeback_slices > 1 else []
            )
            new_ops.extend(store_tiles)
            store_dst = store_tiles[0].result() if store_tiles else block_dst
            store_attrs = dict(attrs)
            store_attrs["tma"] = True
            if owner is not None:
                store_attrs["label"] = f"{owner}_dst_tma"
            store_collective = {
                axis: level.join(CollectiveLevel.BLOCK)
                for axis, level in op.collective.items()
            }
            new_ops.append(MoveOp.build(
                store_dst, smem.result(),
                collective=store_collective, attrs=store_attrs,
            ))
        return new_ops, []


def stage_register_writeback(
    module: ModuleOp,
    *,
    writeback_slices: int = 1,
    writeback_buffers: int = 1,
    physical_layout: dict[str, object] | None = None,
    unsafe_reuse_smem: bool = False,
    label_filter: str | None = None,
) -> ModuleOp:
    return StageRegisterWriteback(
        writeback_slices=writeback_slices,
        writeback_buffers=writeback_buffers,
        physical_layout=physical_layout,
        unsafe_reuse_smem=unsafe_reuse_smem,
        label_filter=label_filter,
    ).run(module)
