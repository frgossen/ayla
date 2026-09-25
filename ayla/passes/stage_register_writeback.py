"""Materialize register-to-global writes through shared memory when requested."""
from __future__ import annotations

from ..base.ir import Block, Op, Value
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AllocOp,
    CollectiveLevel,
    ElementwiseOp,
    GmemTensorType,
    IfOp,
    IndexConstantOp,
    IndexEqOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    WrmemTensorType,
)
from ._utils import (
    build_slice_tile,
    label_matches,
    rebuild_tile,
    tile_introduces_narrow_level,
)


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
        wg_column_tile = next((
            tile for tile in narrow_tiles
            if tile.result().type.sizes[-1] != tile.arg.type.sizes[-1]
        ), None)
        expected_slices = block_dst.type.sizes[-1] // src.type.sizes[-1]
        if wg_column_tile is not None and self.writeback_slices != expected_slices:
            raise ValueError(
                "warpgroup-column writeback needs one slice per warpgroup"
            )
        column_owner = (
            None if wg_column_tile is None else wg_column_tile.indices[-1]
        )
        if column_owner is not None and not isinstance(column_owner, Value):
            raise ValueError(
                "warpgroup-column writeback needs a dynamic owner"
            )

        for slice_index in range(self.writeback_slices):
            smem = smems[slice_index if self.writeback_buffers > 1 else 0]
            gather_dst = smem.result()
            for tile in reversed(narrow_tiles):
                if tile is wg_column_tile:
                    continue
                rebuilt = rebuild_tile(tile, gather_dst)
                new_ops.append(rebuilt)
                gather_dst = rebuilt.result()
            gather_attrs = (
                {} if owner is None else {"label": f"{owner}_dst_smem"}
            )
            if column_owner is None and self.writeback_slices > 1:
                # TODO: Express this source slice with a tile op instead of the
                # wgmma_col_start attribute.
                gather_attrs["wgmma_col_start"] = slice_index * slice_cols
            gather = MoveOp.build(
                gather_dst, src,
                collective=op.collective, attrs=gather_attrs,
            )
            if column_owner is None:
                new_ops.append(gather)
            else:
                slice_id = IndexConstantOp.build(slice_index)
                is_owner = IndexEqOp.build(column_owner, slice_id.result())
                new_ops.extend([
                    slice_id,
                    is_owner,
                    IfOp.build(is_owner.result(), Block(ops=[gather])),
                ])

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
            new_ops.append(MoveOp.build(
                store_dst, smem.result(),
                collective=op.collective, attrs=store_attrs,
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
