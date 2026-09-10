"""Promote matmul operands to a target memory space.

For each ``MatmulOp``, selected operands are promoted by inserting
``alloc`` + ``move`` pairs that stage data into the target memory.

Example — promote lhs/rhs to smem::

    # Before:
    matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>,
           %c: rmem<128x128xf32>)

    # After:
    %a_smem: smem<128x64xbf16 swizzle="128B"> = alloc()
    move(%a_smem, %a) attrs={tma: true}
    %b_smem: smem<128x64xbf16 swizzle="128B"> = alloc()
    move(%b_smem, %b) attrs={tma: true}
    matmul(%a_smem, %b_smem, %c)
"""

from __future__ import annotations

from ..gpu_dialect import (
    AllocOp,
    BF16Type,
    BlockIdsOp,
    ConvertOp,
    F16Type,
    F32Type,
    F64Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    LaneIdOp,
    MatmulOp,
    ModuleOp,
    MoveOp,
    RmemTensorType,
    SmemTensorType,
    ThreadIdsOp,
    TileOp,
    WarpGroupIdOp,
    WrmemTensorType,
    is_elementwise,
)
from ..base.ir import Block, Op, Value, clone_type
from ..base.lowering_pass import LoweringPass, lowers
from .utils import label_matches


def _target_level(target_type: str) -> str:
    if target_type == "smem":
        return "block"
    if target_type == "wrmem":
        return "wg"
    if target_type == "rmem":
        return "thread"
    if target_type == "gmem":
        return "block"
    raise ValueError(f"unsupported target type {target_type!r}")


def _narrowest_level(a: str, b: str) -> str:
    if a == "thread" or b == "thread":
        return "thread"
    if a == "wg" or b == "wg":
        return "wg"
    return "block"


def _collective_level(value: Value) -> str:
    """Determine the hardware parallelism level at which *value* varies.

    Returns "block", "wg", or "thread".
    """
    defn = value.definition

    # Block arguments are kernel args or for-loop IVs — both are uniform
    # across all threads in a block.
    if isinstance(defn, Block):
        return "block"

    if isinstance(defn, BlockIdsOp):
        return "block"
    if isinstance(defn, WarpGroupIdOp):
        return "wg"
    if isinstance(defn, (ThreadIdsOp, LaneIdOp)):
        return "thread"
    if isinstance(defn, IndexConstantOp):
        return "block"

    if isinstance(defn, AllocOp):
        ty = value.type
        if isinstance(ty, SmemTensorType):
            return "block"
        if isinstance(ty, WrmemTensorType):
            return "wg"
        if isinstance(ty, (RmemTensorType, GmemTensorType)):
            return "thread"

    if isinstance(defn, TileOp):
        level = _collective_level(defn.arg)
        for idx in defn.indices:
            if isinstance(idx, Value):
                level = _narrowest_level(level, _collective_level(idx))
        return level

    if isinstance(defn, (IndexAddOp, IndexMulOp, IndexDivOp, IndexModOp)):
        level = "block"
        for operand in defn.operands:
            level = _narrowest_level(level, _collective_level(operand))
        return level

    return "block"


def _introduces_narrow(tile: TileOp, target_level: str) -> bool:
    return any(
        isinstance(idx, Value)
        and _narrowest_level(_collective_level(idx), target_level) != target_level
        for idx in tile.indices
    )


def _resolve_type(name: str):
    if name == "bf16":
        return BF16Type
    if name == "f16":
        return F16Type
    if name == "f32":
        return F32Type
    if name == "f64":
        return F64Type
    if name == "smem":
        return SmemTensorType
    if name == "rmem":
        return RmemTensorType
    if name == "wrmem":
        return WrmemTensorType
    if name == "gmem":
        return GmemTensorType
    raise ValueError(f"unsupported type {name!r}")


def _apply_tile(tile: TileOp, new_arg: Value) -> tuple[list[Op], list[Value]]:
    """Rebuild a tile op on a new argument, adjusting sizes for dims the tile changes."""
    tile_type = tile.result().type
    arg_type = tile.arg.type
    new_tile_type = clone_type(new_arg.type)
    for i in range(len(new_tile_type.sizes)):
        if tile_type.sizes[i] != arg_type.sizes[i]:
            new_tile_type.sizes[i] = tile_type.sizes[i]
    new_tile = TileOp.build(
        new_arg, list(tile.indices), new_tile_type,
        maybe_partial=tile.maybe_partial,
    )
    return [new_tile], [new_tile.result()]


def _slice_tile(value: Value, dim: int, index: int, size: int) -> TileOp:
    typ = clone_type(value.type)
    dim %= typ.rank
    typ.sizes[dim] = size
    indices = [0] * typ.rank
    indices[dim] = index
    return TileOp.build(value, indices, typ)


def _sliced_moves(
    dst: Value, src: Value, count: int, dim: int, attrs: dict[str, object],
) -> list[Op]:
    if count == 1:
        return [MoveOp.build(dst, src, attrs=attrs)]
    size, remainder = divmod(src.type.sizes[dim], count)
    if remainder:
        raise ValueError("move extent must divide evenly into slices")
    ops: list[Op] = []
    for index in range(count):
        src_tile, dst_tile = [_slice_tile(v, dim, index, size) for v in (src, dst)]
        move = MoveOp.build(dst_tile.result(), src_tile.result(), attrs=attrs)
        ops.extend([src_tile, dst_tile, move])
    return ops


class PromoteBuffers(LoweringPass):
    """Promote matmul operands to *target_type* memory space.

    When promoting to smem, operands are loaded block-collectively. If
    an operand's tile chain depends on a narrower level (e.g. warp
    group id), the pass separates the chain into block-collective tiles
    (applied before the alloc+move) and narrow-level tiles (replayed on
    the promoted smem buffer after the move).

    TODO: The writeback path (dst promotion) does not yet handle
    narrow-level-dependent output tiles — it always writes back
    directly without considering the collective level.

    Parameters
    ----------
    target_type : str
        Memory space to promote into: ``"gmem"``, ``"smem"``, ``"rmem"``, ``"wrmem"``.
    operands : list[str]
        Which matmul operands to promote: ``"lhs"``, ``"rhs"``, ``"dst"``.
    swizzle : str or None
        Smem swizzle mode (``"128B"``, ``"64B"``, ``"32B"``).
    layout : str or None
        Wrmem layout (``"mma"``, ``"wgmma_layout_0"``).
    warps : int or None
        Wrmem warp count (``1``, ``4``).
    tma : bool
        Use TMA for the move. A WRMEM destination is written back through a
        block-wide SMEM staging buffer because TMA cannot read registers.
    nc : bool
        Non-coherent cache hint for scalar gmem→rmem loads.
    physical_layout : dict or None
        Physical layout attrs for promoted shared-memory buffers.
    move_attrs : dict[str, dict[str, object]] or None
        Extra move attributes by operand, such as a TMA ``l2_cache_hint``.
    writeback_slices, writeback_buffers : int
        Number of TMA output slices and SMEM staging arenas.
    tma_load_slices : dict[str, int] or None
        Leading-dimension TMA load slices by operand.
    """

    def __init__(
        self,
        target_type: str,
        target_dtype: str | None = None,
        *,
        operands: list[str],
        swizzle: str | None = None,
        layout: str | None = None,
        warps: int | None = None,
        tma: bool = False,
        nc: bool = False,
        physical_layout: dict[str, object] | None = None,
        move_attrs: dict[str, dict[str, object]] | None = None,
        writeback_slices: int = 1,
        writeback_buffers: int = 1,
        tma_load_slices: dict[str, int] | None = None,
        label_filter: str | None = None,
        epilogue_label_filter: str | None = None,
    ) -> None:
        super().__init__()
        self.target_type = target_type
        self.target_dtype = target_dtype
        self.operand_names = set(operands)
        self.label_filter = label_filter
        self.epilogue_label_filter = epilogue_label_filter
        self.swizzle = swizzle
        self.layout = layout
        self.warps = warps
        self.tma = tma
        self.nc = nc
        self.physical_layout = physical_layout
        self.move_attrs = dict(move_attrs or {})
        self.writeback_slices = writeback_slices
        self.writeback_buffers = writeback_buffers
        self.tma_load_slices = dict(tma_load_slices or {})
        assert writeback_slices >= 1, "writeback_slices must be positive"
        assert writeback_buffers in (1, writeback_slices), (
            "writeback_buffers must be 1 or equal writeback_slices"
        )
        assert all(n >= 1 for n in self.tma_load_slices.values()), (
            "TMA load slice counts must be positive"
        )

    def _build_smem_type(self, sizes, dtype):
        layout = dict(self.physical_layout or {})
        swizzle = layout.pop("swizzle", self.swizzle)
        attrs = (
            {"physical_layout": layout} if layout else None
        )
        return SmemTensorType.build(
            sizes, dtype, swizzle=swizzle, attrs=attrs,
        )

    def _build_target_type(self, base_type, dtype=None):
        sizes = list(base_type.sizes)
        if dtype is None and self.target_dtype is not None:
            dtype = _resolve_type(self.target_dtype)()
        if dtype is None:
            dtype = base_type.dtype
        tensor_cls = _resolve_type(self.target_type)
        if tensor_cls is GmemTensorType:
            return GmemTensorType.build(sizes, dtype)
        if tensor_cls is SmemTensorType:
            return self._build_smem_type(sizes, dtype)
        if tensor_cls is RmemTensorType:
            return RmemTensorType.build(sizes, dtype)
        if tensor_cls is WrmemTensorType:
            return WrmemTensorType.build(sizes, dtype, warps=self.warps, layout=self.layout)
        raise ValueError(f"unsupported target type {self.target_type!r}")

    def _build_buffer_attrs(
        self, matmul_label: str | None, operand_name: str
    ) -> dict[str, object]:
        """The label identifying the buffer promoted for one matmul operand."""
        if matmul_label is None:
            return {}
        return {"label": f"{matmul_label}_{operand_name}_{self.target_type}"}

    def _build_move_attrs(
        self, matmul_label: str | None, operand_name: str, allow_nc: bool = True
    ) -> dict[str, object]:
        attrs: dict[str, object] = {}
        if matmul_label is not None:
            attrs["label"] = f"{matmul_label}_{operand_name}_{self.target_type}"
        if self.tma:
            attrs["tma"] = True
        if self.nc and allow_nc and not self.tma:
            attrs["nc"] = True
        attrs.update(self.move_attrs.get(operand_name, {}))
        return attrs

    def _build_operand_promotion(
        self,
        src: Value,
        operand_name: str,
        matmul_label: str | None = None,
        init: object = None,
    ) -> tuple[list[Op], list[Value]]:
        """Promote an operand to the target memory, handling any collective level.

        1. Determine the target collective level from the target memory type.
        2. Walk the tile chain from src upward, collecting tiles narrower
           than the target level. The split point is the first value (from
           the bottom) that is at or above the target level.
        3. Alloc in target memory and move the split point value into it.
        4. Replay the chain tiles on the promoted buffer to produce the
           replacement operand.
        """
        target_level = _target_level(self.target_type)

        # Walk the tile chain bottom-to-top (src → root), collecting
        # "chain" tiles (narrower than target level). The split point is
        # the first tile at or above the target level.
        chain: list[TileOp] = []
        v = src
        while isinstance(v.definition, TileOp):
            tile_op = v.definition
            if _narrowest_level(_collective_level(tile_op.result()), target_level) == target_level:
                break
            chain.append(tile_op)
            v = tile_op.arg

        # Walk chain top-to-bottom. Tiles that don't introduce the
        # narrow level are applied to the chain_root before alloc (e.g.
        # k-tile narrows 128x4096 → 128x64). Tiles whose indices
        # introduce the narrow level are skipped for now — they'll be
        # replayed on the promoted buffer after alloc.
        chain_root = v
        new_ops: list[Op] = []

        for tile in reversed(chain):
            if not _introduces_narrow(tile, target_level):
                ops, [chain_root] = _apply_tile(tile, chain_root)
                new_ops.extend(ops)

        # Alloc + move at the target level.
        target_type = self._build_target_type(chain_root.type)
        # The allocation carries the same label as the move that fills it, so a
        # later pass can name the buffer it wants -- multi_buffer only touches
        # labelled ones.
        buffer_attrs = self._build_buffer_attrs(matmul_label, operand_name)
        if init is not None:
            alloc = AllocOp.build_with_fresh_id(
                target_type, init=init, attrs=buffer_attrs
            )
            new_ops.append(alloc)
        else:
            attrs = self._build_move_attrs(matmul_label, operand_name)
            alloc = AllocOp.build_with_fresh_id(
                self._build_target_type(chain_root.type, dtype=chain_root.type.dtype),
                attrs=buffer_attrs,
            )
            new_ops.append(alloc)
            new_ops.extend(_sliced_moves(
                alloc.result(), chain_root,
                self.tma_load_slices.get(operand_name, 1), 0, attrs,
            ))
            if target_type.dtype != chain_root.type.dtype:
                cvt_alloc = AllocOp.build_with_fresh_id(target_type)
                cvt = ConvertOp.build(dst=cvt_alloc.result(), src=alloc.result())
                new_ops.extend([cvt_alloc, cvt])
                alloc = cvt_alloc

        # Replay narrow-level tiles on the promoted buffer (e.g. tile smem by wg_id).
        result = alloc.result()
        for tile in reversed(chain):
            if _introduces_narrow(tile, target_level):
                ops, [result] = _apply_tile(tile, result)
                new_ops.extend(ops)

        return new_ops, [result]

    def _build_tma_writeback(
        self, dst: Value, src: Value, matmul_label: str | None,
    ) -> list[Op]:
        """Stage a narrow distributed value for one block-wide TMA store.

        The block-wide SMEM tile is allocated whole (the bulk store transfers it
        in one copy), then tiled back down to the writer's collective level so
        the gather move matches ``src``'s shape -- mirroring how
        ``_build_operand_promotion`` replays narrow tiles on a promoted buffer,
        which keeps the warp-group row placement in the IR rather than the
        lowering.
        """
        block_dst = dst
        narrow_tiles: list[TileOp] = []
        while isinstance(block_dst.definition, TileOp):
            tile = block_dst.definition
            if not _introduces_narrow(tile, "block"):
                break
            narrow_tiles.append(tile)
            block_dst = tile.arg
        assert isinstance(block_dst.type, GmemTensorType)
        smem_sizes = list(block_dst.type.sizes)
        if smem_sizes[-1] % self.writeback_slices:
            raise ValueError("output columns must divide evenly into writeback_slices")
        smem_sizes[-1] //= self.writeback_slices
        smem_type = self._build_smem_type(
            smem_sizes, block_dst.type.dtype,
        )
        smems = [
            AllocOp.build_with_fresh_id(smem_type)
            for _ in range(self.writeback_buffers)
        ]
        new_ops: list[Op] = list(smems)
        slice_cols = smem_sizes[-1]
        wg_column_tile = next((
            tile for tile in narrow_tiles
            if tile.result().type.sizes[-1] != tile.arg.type.sizes[-1]
        ), None)
        expected_slices = block_dst.type.sizes[-1] // src.type.sizes[-1]
        if wg_column_tile is not None and self.writeback_slices != expected_slices:
            raise ValueError("warpgroup-column writeback needs one slice per warpgroup")
        for slice_idx in range(self.writeback_slices):
            smem = smems[slice_idx if self.writeback_buffers > 1 else 0]
            gather_dst = smem.result()
            for tile in reversed(narrow_tiles):
                if tile is wg_column_tile:
                    continue
                ops, [gather_dst] = _apply_tile(tile, gather_dst)
                new_ops.extend(ops)
            gather_attrs = {"label": f"{matmul_label}_dst_smem"}
            if wg_column_tile is not None:
                gather_attrs["writer_wg"] = slice_idx
            elif self.writeback_slices > 1:
                # TODO: Express this source slice with a tile op instead of the
                # wgmma_col_start attribute.
                gather_attrs["wgmma_col_start"] = slice_idx * slice_cols
            gather = MoveOp.build(dst=gather_dst, src=src, attrs=gather_attrs)
            store_tiles = (
                [_slice_tile(block_dst, -1, slice_idx, slice_cols)]
                if self.writeback_slices > 1 else []
            )
            store_dst_value = store_tiles[0].result() if store_tiles else block_dst
            store_attrs = {"label": f"{matmul_label}_dst_tma", "tma": True}
            store_attrs.update(self.move_attrs.get("dst", {}))
            store = MoveOp.build(
                dst=store_dst_value, src=smem.result(), attrs=store_attrs,
            )
            new_ops.extend([gather, *store_tiles, store])
        return new_ops

    def _build_operand_writeback(
        self, dst: Value, src: Value, operand_name: str, matmul_label: str | None = None
    ) -> tuple[list[Op], list[Value]]:
        target_level = _target_level(self.target_type)
        dst_level = _collective_level(dst)
        if _narrowest_level(dst_level, target_level) != target_level:
            raise NotImplementedError(
                f"writeback for {dst_level}-level dst not yet supported "
                f"when promoting to {self.target_type}"
            )
        new_ops: list[Op] = []
        if src.type.dtype.name != dst.type.dtype.name:
            tmp_type = self._build_target_type(dst.type, dtype=dst.type.dtype)
            tmp_alloc = AllocOp.build_with_fresh_id(tmp_type)
            cvt = ConvertOp.build(dst=tmp_alloc.result(), src=src)
            new_ops.extend([tmp_alloc, cvt])
            src = tmp_alloc.result()
        if self.tma and self.target_type == "wrmem":
            new_ops.extend(self._build_tma_writeback(dst, src, matmul_label))
            return new_ops, []
        attrs = self._build_move_attrs(matmul_label, operand_name, allow_nc=False)
        move = MoveOp.build(dst=dst, src=src, attrs=attrs)
        new_ops.append(move)
        return new_ops, []

    @lowers(KernelOp)
    def lower_kernel(self, op: KernelOp):
        for block in op.blocks:
            self._rewrite_block(block)
        return [op], []

    def _rewrite_block(self, block: Block) -> None:
        """Replace each matmul, and the epilogue behind it, with its promotion.

        Nested blocks first, so a matmul inside a loop is promoted before the
        loop holding it is rebuilt.
        """
        for op in block.ops:
            for nested in op.blocks:
                self._rewrite_block(nested)

        ops = list(block.ops)
        new_ops: list[Op] = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if not self._promotes(op):
                new_ops.append(op)
                i += 1
                continue
            epilogue = self._epilogue(ops, i)
            new_ops.extend(self._promote(op, epilogue))
            i += 1 + len(epilogue)
        block.ops = new_ops

    def _promotes(self, op: Op) -> bool:
        return isinstance(op, MatmulOp) and label_matches(op, self.label_filter)

    def _epilogue(self, ops: list[Op], hero: int) -> list[Op]:
        """The ops directly behind ``ops[hero]`` that promote with it.

        Contiguous only, and each has to earn its place: element-wise, selected
        by ``epilogue_label_filter``, and in place over the destination.
        Anything else ends the run -- an op left where it is must not be
        stepped over, or moving what follows past it would reorder the two.

        Element-wise alone would be too weak a test, which is why the filter
        has to name them: promotion leaves a `convert` behind every
        accumulator, and a `convert` is element-wise.
        """
        if self.epilogue_label_filter is None:
            return []
        found: list[Op] = []
        dst = ops[hero].dst
        for op in ops[hero + 1:]:
            if not is_elementwise(op):
                break
            if not label_matches(op, self.epilogue_label_filter):
                break
            if len(op.operands) != 2 or not all(v is dst for v in op.operands):
                break
            found.append(op)
        return found

    def _promote(self, op: MatmulOp, epilogue: list[Op]) -> list[Op]:
        lhs, rhs, dst = op.operands

        label = op.attrs.get("label")
        pre_ops = []
        new_lhs, new_rhs, new_dst = lhs, rhs, dst

        if "lhs" in self.operand_names:
            ops, [new_lhs] = self._build_operand_promotion(lhs, "lhs", label)
            pre_ops.extend(ops)
        if "rhs" in self.operand_names:
            ops, [new_rhs] = self._build_operand_promotion(rhs, "rhs", label)
            pre_ops.extend(ops)
        init = op.init
        if "dst" in self.operand_names:
            ops, [new_dst] = self._build_operand_promotion(dst, "dst", label, init=init)
            pre_ops.extend(ops)
            init = None

        matmul = MatmulOp.build(
            lhs=new_lhs, rhs=new_rhs, dst=new_dst, init=init, attrs=op.extra_attrs
        )

        # The epilogue runs on the promoted buffer, before it is written back:
        # that is the whole point of promoting the destination, and it is only
        # reachable here because the ops behind the matmul came along.
        post_ops = [
            type(e)(operands=[new_dst, new_dst], results=[], attrs=e.attrs)
            for e in epilogue
        ]
        if "dst" in self.operand_names:
            ops, _ = self._build_operand_writeback(dst, new_dst, "dst", label)
            post_ops.extend(ops)

        return pre_ops + [matmul] + post_ops


def promote_buffers(
    module: ModuleOp,
    target_type: str,
    target_dtype: str | None = None,
    *,
    operands: list[str],
    swizzle: str | None = None,
    layout: str | None = None,
    warps: int | None = None,
    tma: bool = False,
    nc: bool = False,
    physical_layout: dict[str, object] | None = None,
    move_attrs: dict[str, dict[str, object]] | None = None,
    writeback_slices: int = 1,
    writeback_buffers: int = 1,
    tma_load_slices: dict[str, int] | None = None,
    label_filter: str | None = None,
    epilogue_label_filter: str | None = None,
) -> ModuleOp:
    """Convenience entry point."""
    return PromoteBuffers(
        target_type=target_type,
        target_dtype=target_dtype,
        operands=operands,
        swizzle=swizzle,
        layout=layout,
        warps=warps,
        tma=tma,
        nc=nc,
        physical_layout=physical_layout,
        move_attrs=move_attrs,
        writeback_slices=writeback_slices,
        writeback_buffers=writeback_buffers,
        tma_load_slices=tma_load_slices,
        label_filter=label_filter,
        epilogue_label_filter=epilogue_label_filter,
    ).run(module)
