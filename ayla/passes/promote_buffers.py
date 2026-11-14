"""Promote matmul operands to a target memory space.

For each ``MatmulOp``, selected operands are promoted by inserting
``alloc`` + ``move`` pairs that stage data into the target memory.

Example — promote lhs/rhs to smem::

    # Before:
    matmul(%a: gmem<128x64xbf16>, %b: gmem<128x64xbf16>,
           %c: rmem<128x128xf32>) collective=xyz.grid

    # After:
    %a_smem: smem<128x64xbf16 swizzle="128B"> = alloc()
    move(%a_smem, %a) attrs={tma: true}
    %b_smem: smem<128x64xbf16 swizzle="128B"> = alloc()
    move(%b_smem, %b) attrs={tma: true}
    matmul(%a_smem, %b_smem, %c) collective=xyz.grid
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from ..base.ir import Block, Op, Type, Value, clone_type, clone_op, find_ancestor_of_type
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AddrMemTensorType,
    AllocOp,
    BF16Type,
    BlockIdOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
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
    MatmulOp,
    ReduceOp,
    ReshapeOp,
    RegMemTensorType,
    WrmemTensorType,
    MemTensorType,
    MemoryEffect,
    root_buffers,
    has_ordering_effect,
    ModuleOp,
    MoveOp,
    RmemTensorType,
    SmemTensorType,
    ThreadIdOp,
    TileOp,
    WarpGroupIdOp,
    WgrmemTensorType,
    is_column_major,
    memory_effects,
    root_buffer,
)
from ._utils import label_matches, fresh_alloc, select_region


def is_region_local(value: Value, members: set[Op]) -> bool:
    """Whether a logical allocation and all its aliases are private to a region."""
    root = root_buffer(value)
    if root is None or not isinstance(root.definition, AllocOp):
        return False
    visited: set[Value] = set()
    kernel = find_ancestor_of_type(next(iter(members)), KernelOp)

    def local(alias: Value) -> bool:
        if alias in visited:
            return True
        visited.add(alias)
        for user in alias.users():
            if find_ancestor_of_type(user, KernelOp) is not kernel or user in members:
                continue
            if isinstance(user, (TileOp, ReshapeOp)) and local(user.result()):
                continue
            return False
        return True

    return local(root)


def _target_level(target_type: str) -> str:
    if target_type == "smem":
        return "block"
    if target_type == "wgrmem":
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

    if isinstance(defn, BlockIdOp):
        return "block"
    if isinstance(defn, WarpGroupIdOp):
        return "wg"
    if isinstance(defn, ThreadIdOp):
        return "thread"
    if isinstance(defn, IndexConstantOp):
        return "block"

    if isinstance(defn, AllocOp):
        ty = value.type
        if isinstance(ty, SmemTensorType):
            return "block"
        if isinstance(ty, WgrmemTensorType):
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
    if name == "wrmem":
        return WrmemTensorType
    if name == "rmem":
        return RmemTensorType
    if name == "wgrmem":
        return WgrmemTensorType
    if name == "gmem":
        return GmemTensorType
    raise ValueError(f"unsupported type {name!r}")


def _apply_tile(tile: TileOp, new_arg: Value) -> tuple[list[Op], list[Value]]:
    """Rebuild a tile op on a new argument, adjusting sizes for dims the tile changes."""
    tile_type = tile.result().type
    arg_type = tile.arg.type
    new_tile_type = clone_type(new_arg.type)
    assert isinstance(tile_type, MemTensorType)
    assert isinstance(arg_type, MemTensorType)
    assert isinstance(new_tile_type, MemTensorType)
    new_sizes = list(new_tile_type.sizes)
    for i in range(len(new_sizes)):
        if tile_type.sizes[i] != arg_type.sizes[i]:
            new_sizes[i] = tile_type.sizes[i]
    new_tile_type.sizes = new_sizes
    new_tile = TileOp.build(
        new_arg,
        list(tile.indices),
        new_tile_type,
        maybe_partial=tile.maybe_partial,
    )
    return [new_tile], [new_tile.result()]


def _slice_tile(value: Value, dim: int, index: int, size: int) -> TileOp:
    typ = clone_type(value.type)
    assert isinstance(typ, MemTensorType)
    dim %= typ.rank
    sizes = list(typ.sizes)
    sizes[dim] = size
    typ.sizes = sizes
    indices: list[Value | int] = [0] * typ.rank
    indices[dim] = index
    return TileOp.build(value, indices, typ)


def _sliced_moves(
    dst: Value,
    src: Value,
    count: int,
    dim: int,
    collective: dict[CollectiveAxis, CollectiveLevel],
    attrs: dict[str, object],
) -> list[Op]:
    attrs = dict(attrs)
    multicast = cast(bool, attrs.pop("multicast", False))
    if multicast is True:
        collective = {axis: CollectiveLevel.CLUSTER for axis in CollectiveAxis}
    if count == 1:
        return [
            MoveOp.build(
                dst,
                src,
                collective=collective,
                multicast=multicast,
                attrs=attrs,
            )
        ]
    src_type = src.type
    assert isinstance(src_type, MemTensorType)
    extent = src_type.sizes[dim]
    assert extent is not None
    size, remainder = divmod(extent, count)
    if remainder:
        raise ValueError("move extent must divide evenly into slices")
    ops: list[Op] = []
    for index in range(count):
        src_tile, dst_tile = [_slice_tile(v, dim, index, size) for v in (src, dst)]
        move = MoveOp.build(
            dst_tile.result(),
            src_tile.result(),
            collective=collective,
            multicast=multicast,
            attrs=attrs,
        )
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
        Memory space to promote into: ``"gmem"``, ``"smem"``, ``"rmem"``, ``"wgrmem"``.
    operands : list[str]
        Which matmul operands to promote: ``"lhs"``, ``"rhs"``, ``"dst"``.
    swizzle : str or None
        Smem swizzle mode (``"128B"``, ``"64B"``, ``"32B"``).
    layout : str or None
        Wgrmem layout (``"mma"``, ``"wgmma_layout_0"``).
    warps : int or None
        Wgrmem warp count (``1``, ``4``).
    tma : bool
        Use TMA for the move. A WGRMEM destination is written back through a
        block-wide SMEM staging buffer because TMA cannot read registers.
        A selected WGRMEM-to-GMEM move is staged the same way.
    nc : bool
        Non-coherent cache hint for scalar gmem→rmem loads.
    permutation : list[int] or None
        Dense SMEM dimension permutation: logical dim i maps to physical
        dim permutation[i]. Defaults to identity; [1, 0] stages rank-2
        tensors column-major. Also applies to TMA writeback staging buffers.
    physical_layout : dict or None
        Physical layout attrs for promoted shared-memory buffers.
    move_attrs : dict[str, dict[str, object]] or None
        Extra move attributes by operand, such as a TMA ``l2_cache_hint``.
    writeback_slices, writeback_buffers : int
        Number of TMA output slices and SMEM staging arenas.
    unsafe_reuse_smem : bool
        Mark shared allocations by their asserted compute/writeback lifetime
        so a later arena-packing pass may overlay their storage. This is unsafe:
        no analysis verifies that differently named groups are non-overlapping.
        TODO: Replace this policy flag with independent allocations followed by
        lifetime analysis and a pass that selects safe storage reuse.
    reuse_operands : bool
        Retain identical read-only operand promotions within a block. Writes
        to the source or promoted storage, fences, and unknown effects invalidate
        the cache. Intended for projections sharing an input K tile.
    tma_load_slices : dict[str, int] or None
        Leading-dimension TMA load slices by operand.
    """

    target_type: str
    target_dtype: str | None
    swizzle: str | None
    layout: str | None
    warps: int | None
    tma: bool
    nc: bool
    permutation: list[int] | None
    physical_layout: dict[str, object] | None
    move_attrs: dict[str, dict[str, object]]
    writeback_slices: int
    writeback_buffers: int
    unsafe_reuse_smem: bool
    reuse_operands: bool
    tma_load_slices: dict[str, int]
    label_filter: str | None
    epilogue_label_filter: str | None
    operand_names: set[str]

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
        permutation: list[int] | None = None,
        physical_layout: dict[str, object] | None = None,
        move_attrs: dict[str, dict[str, object]] | None = None,
        writeback_slices: int = 1,
        writeback_buffers: int = 1,
        unsafe_reuse_smem: bool = False,
        reuse_operands: bool = False,
        tma_load_slices: dict[str, int] | None = None,
        label_filter: str | None = None,
        epilogue_label_filter: str | None = None,
    ) -> None:
        super().__init__(
            target_type=target_type,
            target_dtype=target_dtype,
            operands=operands,
            swizzle=swizzle,
            layout=layout,
            warps=warps,
            tma=tma,
            nc=nc,
            permutation=list(permutation) if permutation is not None else None,
            physical_layout=physical_layout,
            move_attrs=dict(move_attrs or {}),
            writeback_slices=writeback_slices,
            writeback_buffers=writeback_buffers,
            unsafe_reuse_smem=unsafe_reuse_smem,
            reuse_operands=reuse_operands,
            tma_load_slices=dict(tma_load_slices or {}),
            label_filter=label_filter,
            epilogue_label_filter=epilogue_label_filter,
        )
        assert writeback_slices >= 1, "writeback_slices must be positive"
        assert writeback_buffers in (1, writeback_slices), (
            "writeback_buffers must be 1 or equal writeback_slices"
        )
        assert all(n >= 1 for n in self.tma_load_slices.values()), (
            "TMA load slice counts must be positive"
        )
        self.operand_names = set(operands)

    def _build_smem_type(
        self,
        sizes: Sequence[int],
        dtype: Type,
        column_major: bool = False,
        strides: Sequence[int] | None = None,
    ) -> SmemTensorType:
        layout = dict(self.physical_layout or {})
        swizzle = layout.pop("swizzle", self.swizzle)
        assert swizzle is None or isinstance(swizzle, str)
        if "block_width" in layout:
            # Blocks tile the contiguous dimension; a narrower one gets the
            # widest block that divides it, swizzled across one block row.
            requested = cast(int, layout["block_width"])
            block_width = math.gcd(sizes[-2] if column_major else sizes[-1], requested)
            if block_width != requested:
                layout["block_width"] = block_width
                row_bytes = (
                    block_width
                    * cast(
                        BF16Type | F16Type | F32Type | F64Type,
                        dtype,
                    ).byte_size
                )
                swizzle = f"{row_bytes}B" if row_bytes >= 32 else None
        attrs: dict[str, object] | None = (
            {"physical_layout": layout} if layout else None
        )
        return SmemTensorType.build(
            sizes,
            dtype,
            swizzle=swizzle,
            attrs=attrs,
            strides=strides
            if strides is not None
            else ([1, sizes[0]] if column_major else None),
        )

    def _build_target_type(
        self,
        base_type: MemTensorType,
        dtype: Type | None = None,
    ) -> MemTensorType:
        assert not base_type.is_dynamic
        sizes = [size for size in base_type.sizes if size is not None]
        if dtype is None and self.target_dtype is not None:
            dtype = _resolve_type(self.target_dtype)()
        if dtype is None:
            dtype = base_type.dtype
        tensor_cls = _resolve_type(self.target_type)
        if tensor_cls is GmemTensorType:
            return GmemTensorType.build(sizes, dtype)
        if tensor_cls is SmemTensorType:
            strides = SmemTensorType.canonical_strides(sizes, self.permutation)
            return self._build_smem_type(
                sizes,
                dtype,
                column_major=len(strides) == 2 and strides[-2] == 1,
                strides=cast(list[int], strides),
            )
        if tensor_cls is WrmemTensorType:
            return WrmemTensorType.build(sizes, dtype, layout=self.layout or "default")
        if tensor_cls is RmemTensorType:
            return RmemTensorType.build(sizes, dtype)
        if tensor_cls is WgrmemTensorType:
            assert self.warps is not None and self.layout is not None
            return WgrmemTensorType.build(
                sizes, dtype, warps=self.warps, layout=self.layout
            )
        raise ValueError(f"unsupported target type {self.target_type!r}")

    def _build_buffer_attrs(
        self, matmul_label: str | None, operand_name: str
    ) -> dict[str, object]:
        """The label identifying the buffer promoted for one matmul operand."""
        if matmul_label is None:
            return {}
        attrs: dict[str, object] = {
            "label": f"{matmul_label}_{operand_name}_{self.target_type}",
        }
        if self.unsafe_reuse_smem and self.target_type == "smem":
            attrs["reuse_group"] = f"{matmul_label}_compute"
        return attrs

    def _build_move_attrs(
        self, matmul_label: str | None, operand_name: str, allow_nc: bool = True
    ) -> dict[str, object]:
        attrs: dict[str, object] = {}
        if matmul_label is not None:
            attrs["label"] = f"{matmul_label}_{operand_name}_{self.target_type}"
        # TMA cannot fill registers; a register target only stages writeback.
        if self.tma and self.target_type != "wgrmem":
            attrs["tma"] = True
        if self.nc and allow_nc and not self.tma:
            attrs["nc"] = True
        attrs.update(self.move_attrs.get(operand_name, {}))
        return attrs

    def _build_operand_promotion(
        self,
        src: Value,
        operand_name: str,
        collective: dict[CollectiveAxis, CollectiveLevel],
        matmul_label: str | None = None,
        init: Value | None = None,
        overwrite: bool = False,
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
        assert isinstance(src.type, MemTensorType)
        target_level = _target_level(self.target_type)

        # Walk the tile chain bottom-to-top (src → root), collecting
        # "chain" tiles (narrower than target level). The split point is
        # the first tile at or above the target level.
        chain: list[TileOp] = []
        v = src
        while isinstance(v.definition, TileOp):
            tile_op = v.definition
            if (
                _narrowest_level(_collective_level(tile_op.result()), target_level)
                == target_level
            ):
                break
            chain.append(tile_op)
            v = tile_op.arg

        # Walk chain top-to-bottom. Tiles that don't introduce the
        # narrow level are applied to the chain_root before alloc (e.g.
        # k-tile narrows 128x4096 → 128x64). Tiles whose indices
        # introduce the narrow level are skipped for now — they'll be
        # replayed on the promoted buffer after alloc.
        chain_root = v
        assert isinstance(chain_root.type, MemTensorType)
        new_ops: list[Op] = []

        for tile in reversed(chain):
            if not _introduces_narrow(tile, target_level):
                ops, [chain_root] = _apply_tile(tile, chain_root)
                new_ops.extend(ops)

        # Alloc + move at the target level.
        chain_root_type = chain_root.type
        assert isinstance(chain_root_type, MemTensorType)
        target_type = self._build_target_type(chain_root_type)
        # The allocation carries the same label as the move that fills it, so a
        # later pass can name the buffer it wants -- multi_buffer only touches
        # labelled ones.
        buffer_attrs = self._build_buffer_attrs(matmul_label, operand_name)
        if overwrite:
            alloc = AllocOp.build_with_fresh_id(target_type, attrs=buffer_attrs)
            new_ops.append(alloc)
        elif init is not None:
            # SMEM fill has no lowering. Keep the initialization on the matmul
            # so a subsequent register promotion can materialize it.
            alloc = AllocOp.build_with_fresh_id(target_type, attrs=buffer_attrs)
            new_ops.append(alloc)
            if self.target_type != "smem":
                new_ops.append(
                    ElementwiseOp.build(
                        alloc.result(),
                        srcs=[init],
                        fn="fill",
                    )
                )
        else:
            attrs = self._build_move_attrs(matmul_label, operand_name)
            chain_root_type = chain_root.type
            assert isinstance(chain_root_type, MemTensorType)
            alloc = AllocOp.build_with_fresh_id(
                self._build_target_type(
                    chain_root_type,
                    dtype=chain_root_type.dtype,
                ),
                attrs=buffer_attrs,
            )
            new_ops.append(alloc)
            new_ops.extend(
                _sliced_moves(
                    alloc.result(),
                    chain_root,
                    self.tma_load_slices.get(operand_name, 1),
                    0,
                    collective,
                    attrs,
                )
            )
            if target_type.dtype != chain_root_type.dtype:
                cvt_alloc = AllocOp.build_with_fresh_id(target_type)
                cvt = ElementwiseOp.build(
                    dst=cvt_alloc.result(),
                    srcs=[alloc.result()],
                    fn="convert",
                )
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
        self,
        dst: Value,
        src: Value,
        collective: dict[CollectiveAxis, CollectiveLevel],
        matmul_label: str | None,
        writeback_tiling_scheme: int | None,
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
        assert isinstance(src.type, MemTensorType)
        narrow_tiles: list[TileOp] = []
        while isinstance(block_dst.definition, TileOp):
            tile = block_dst.definition
            if not _introduces_narrow(tile, "block"):
                break
            narrow_tiles.append(tile)
            block_dst = tile.arg
        assert isinstance(block_dst.type, GmemTensorType)
        smem_sizes = list(block_dst.type.sizes)
        output_cols = smem_sizes[-1]
        assert output_cols is not None
        if output_cols % self.writeback_slices:
            raise ValueError("output columns must divide evenly into writeback_slices")
        smem_sizes[-1] = output_cols // self.writeback_slices
        static_smem_sizes = [size for size in smem_sizes if size is not None]
        assert len(static_smem_sizes) == len(smem_sizes)
        # Stage in the destination's dimension order so the bulk store copies
        # a dense box; a column-major (transposed) destination is staged
        # column-major.
        smem_type = self._build_smem_type(
            static_smem_sizes,
            block_dst.type.dtype,
            column_major=is_column_major(block_dst.type),
        )
        attrs: dict[str, object] | None = (
            {"reuse_group": f"{matmul_label}_writeback"}
            if self.unsafe_reuse_smem and matmul_label is not None
            else None
        )
        smems = [
            AllocOp.build_with_fresh_id(smem_type, attrs=attrs)
            for _ in range(self.writeback_buffers)
        ]
        new_ops: list[Op] = list(smems)
        slice_cols = static_smem_sizes[-1]
        wg_column_tile = None
        if self.writeback_slices > 1:
            for tile in narrow_tiles:
                result_type = tile.result().type
                arg_type = tile.arg.type
                assert isinstance(result_type, MemTensorType)
                assert isinstance(arg_type, MemTensorType)
                if result_type.sizes[-1] != arg_type.sizes[-1]:
                    wg_column_tile = tile
                    break
        dst_cols = block_dst.type.sizes[-1]
        src_cols = src.type.sizes[-1]
        assert dst_cols is not None and src_cols is not None
        expected_slices = dst_cols // src_cols
        if wg_column_tile is not None and self.writeback_slices != expected_slices:
            raise ValueError("warpgroup-column writeback needs one slice per warpgroup")
        writer_divisor = 1
        if wg_column_tile is not None:
            column = wg_column_tile.indices[-1]
            if isinstance(column, Value) and isinstance(column.definition, IndexModOp):
                column = column.definition.lhs
            if isinstance(column, Value) and isinstance(column.definition, IndexDivOp):
                divisor = column.definition.rhs.definition
                if isinstance(divisor, IndexConstantOp):
                    writer_divisor = divisor.value
        for slice_idx in range(self.writeback_slices):
            smem = smems[slice_idx if self.writeback_buffers > 1 else 0]
            gather_dst = smem.result()
            for tile in reversed(narrow_tiles):
                if tile is wg_column_tile:
                    continue
                ops, [gather_dst] = _apply_tile(tile, gather_dst)
                new_ops.extend(ops)
            gather_attrs: dict[str, object] = {
                "label": f"{matmul_label}_dst_smem",
            }
            if wg_column_tile is not None:
                gather_attrs["writer_wg"] = slice_idx
                # Column groups repeat for every row group. Selecting only
                # the absolute warpgroup ID drops all subsequent row groups.
                gather_attrs["writer_wg_modulus"] = expected_slices
                if writer_divisor != 1:
                    gather_attrs["writer_wg_divisor"] = writer_divisor
            elif self.writeback_slices > 1:
                # TODO: Express this source slice with a tile op instead of the
                # wgmma_col_start attribute.
                gather_attrs["wgmma_col_start"] = slice_idx * slice_cols
            gather = MoveOp.build(
                dst=gather_dst,
                src=src,
                collective=collective,
                attrs=gather_attrs,
            )
            store_tiles = (
                [_slice_tile(block_dst, -1, slice_idx, slice_cols)]
                if self.writeback_slices > 1
                else []
            )
            store_dst_value = store_tiles[0].result() if store_tiles else block_dst
            store_attrs: dict[str, object] = {
                "label": f"{matmul_label}_dst_tma",
                "tma": True,
            }
            store_attrs.update(self.move_attrs.get("dst", {}))
            if writeback_tiling_scheme is not None:
                store_attrs["dst_tiling_scheme"] = writeback_tiling_scheme
            store = MoveOp.build(
                dst=store_dst_value,
                src=smem.result(),
                collective=collective,
                attrs=store_attrs,
            )
            new_ops.extend([gather, *store_tiles, store])
        return new_ops

    def _build_operand_writeback(
        self,
        dst: Value,
        src: Value,
        operand_name: str,
        collective: dict[CollectiveAxis, CollectiveLevel],
        matmul_label: str | None = None,
        epilogue: list[Op] | None = None,
        writeback_tiling_scheme: int | None = None,
    ) -> tuple[list[Op], list[Value]]:
        assert isinstance(dst.type, MemTensorType)
        assert isinstance(src.type, MemTensorType)
        target_level = _target_level(self.target_type)
        dst_level = _collective_level(dst)
        if _narrowest_level(dst_level, target_level) != target_level:
            raise NotImplementedError(
                f"writeback for {dst_level}-level dst not yet supported "
                f"when promoting to {self.target_type}"
            )
        new_ops: list[Op] = []
        epilogue = list(epilogue or [])
        if src.type.dtype.name != dst.type.dtype.name:
            tmp_type = self._build_target_type(dst.type, dtype=dst.type.dtype)
            tmp_alloc = AllocOp.build_with_fresh_id(tmp_type)
            final = epilogue[-1] if epilogue else None
            # A final pointwise write can target the output representation
            # directly. Preserve every source; the operation is not required
            # to be unary or in place. Reductions and multi-output regions
            # retain an explicit conversion after their computation.
            direct_output = (
                isinstance(src.type, WgrmemTensorType)
                and src.type.dtype == F32Type()
                and isinstance(final, ElementwiseOp)
                and final.dst is src
                and all(isinstance(v.type, RegMemTensorType) and v.type.dtype == F32Type()
                        for v in final.operands[1:])
                and any(cast(MemTensorType, v.type).rank > 0 for v in final.operands[1:])
            )
            new_ops.extend(epilogue[:-1] if direct_output else epilogue)
            new_ops.append(tmp_alloc)
            if direct_output:
                assert isinstance(final, ElementwiseOp)
                new_ops.append(ElementwiseOp.build(
                    tmp_alloc.result(), final.operands[1:], final.fn, attrs=final.extra_attrs))
            else:
                new_ops.append(
                    ElementwiseOp.build(
                        dst=tmp_alloc.result(), srcs=[src], fn='convert',
                        attrs=({'label': f'{matmul_label}_{operand_name}_convert'}
                               if matmul_label is not None else None),
                    )
                )
            src = tmp_alloc.result()
        else:
            new_ops.extend(epilogue)
        if self.tma and self.target_type == "wgrmem":
            new_ops.extend(
                self._build_tma_writeback(
                    dst,
                    src,
                    collective,
                    matmul_label,
                    writeback_tiling_scheme,
                )
            )
            return new_ops, []
        attrs = self._build_move_attrs(matmul_label, operand_name, allow_nc=False)
        if writeback_tiling_scheme is not None:
            attrs["dst_tiling_scheme"] = writeback_tiling_scheme
        move = MoveOp.build(
            dst=dst,
            src=src,
            collective=collective,
            attrs=attrs,
        )
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
        reusable: dict[tuple, Value] = {}
        i = 0
        while i < len(ops):
            op = ops[i]
            if self.reuse_operands:
                effects = memory_effects(op)
                if effects is None or has_ordering_effect(op):
                    reusable.clear()
                else:
                    writes = {
                        root for effect in effects if effect.kind == "W"
                        for root in effect.roots
                    }
                    reusable = {
                        key: value for key, value in reusable.items()
                        if not writes & (root_buffers(key[1]) | root_buffers(value))
                    }
            if not self._promotes(op):
                new_ops.append(op)
                i += 1
                continue
            if isinstance(op, MoveOp):
                label = op.attrs.get("label")
                assert label is None or isinstance(label, str)
                writeback_ops, _ = self._build_operand_writeback(
                    op.dst,
                    op.src,
                    "dst",
                    op.collective,
                    label,
                    writeback_tiling_scheme=cast(
                        int | None, op.attrs.get("dst_tiling_scheme")
                    ),
                )
                new_ops.extend(writeback_ops)
                i += 1
                continue
            if isinstance(op, ReduceOp):
                epilogue = self._epilogue(ops, i)
                new_ops.extend(self._promote_reduce(op, epilogue))
                i += 1 + len(epilogue)
                continue
            epilogue = self._epilogue(ops, i)
            assert isinstance(op, MatmulOp)
            new_ops.extend(self._promote(op, epilogue, reusable))
            i += 1 + len(epilogue)
        block.ops = new_ops

    def _promotes(self, op: Op) -> bool:
        if not label_matches(op, self.label_filter):
            return False
        return isinstance(op, (MatmulOp, ReduceOp)) or self._is_register_writeback(op)

    def _promote_region(self, ops: list[Op], replacements: dict[Value, Value],
                         owner: RegMemTensorType,
                         collective: dict[CollectiveAxis, CollectiveLevel],
                         label: str | None = None) -> tuple[list[Op], dict[Value, Value]]:
        """Keep consumer temporaries under the producer's register ownership.

        External compute operands are staged according to memory effects. Writes
        retain their logical endpoints and use the scheduled output transport;
        internal allocations remain registers.
        """
        mapping = dict(replacements)
        result: list[Op] = []
        stores: dict[Value, Value] = {}
        addresses: dict[Value, Value] = {}
        initialized: set[Value] = set(replacements)

        for op in ops:
            if isinstance(op, AllocOp):
                typ = cast(MemTensorType, op.result().type)
                alloc = AllocOp.build_with_fresh_id(owner.with_shape(typ.static_shape, typ.dtype), attrs=op.extra_attrs)
                mapping[op.result()] = alloc.result()
                result.append(alloc)
                continue
            if isinstance(op, (ElementwiseOp, ReduceOp)):
                effects = memory_effects(op)
                assert effects is not None
                for effect in effects:
                    value = effect.value
                    if value in replacements or isinstance(value.definition, AllocOp) and value in mapping:
                        continue
                    if value in mapping and isinstance(mapping[value].type, RegMemTensorType) and value not in addresses:
                        continue
                    if not isinstance(value.type, AddrMemTensorType):
                        continue
                    if value not in addresses:
                        addresses[value] = mapping.get(value, value)
                    address = addresses[value]
                    if value not in mapping or isinstance(mapping[value].type, AddrMemTensorType):
                        alloc = AllocOp.build_with_fresh_id(owner.with_shape(value.type.static_shape, value.type.dtype))
                        result.append(alloc)
                        mapping[value] = alloc.result()
                    if effect.kind == 'R' and value not in initialized:
                        result.append(MoveOp.build(mapping[value], address, collective=collective))
                        initialized.add(value)
                    if effect.kind == 'W':
                        stores[address] = mapping[value]
                # Effects list writes before reads; marking initialization here
                # preserves the old destination when a reduction accumulates.
                initialized.update(e.value for e in effects if e.kind == 'W')
            attrs: dict[str, object] = {'collective': dict(collective)} if isinstance(op, (ReduceOp, MoveOp)) else {}
            cloned = clone_op(op, mapping, attrs)
            if isinstance(cloned, MoveOp) and all(isinstance(v.type, RegMemTensorType) for v in cloned.operands):
                if op.operands[0] in replacements:
                    # A buffer assignment versions the region's logical output.
                    # Keep the assigned registers as the live-out, avoiding an
                    # unnecessary round trip through the accumulator's dtype.
                    mapping[op.operands[0]] = cloned.src
                    continue
                if cast(MemTensorType, cloned.dst.type).dtype != cast(MemTensorType, cloned.src.type).dtype:
                    cloned = ElementwiseOp.build(cloned.dst, [cloned.src], 'convert', attrs=cloned.extra_attrs)
            if isinstance(cloned, (TileOp, ReshapeOp)) and isinstance(cloned.arg.type, RegMemTensorType):
                cloned.result().type = cloned.arg.type.with_shape(cast(MemTensorType, op.result().type).static_shape)
            for original, replacement in zip(op.results, cloned.results, strict=True):
                mapping[original] = replacement
            if (self.tma and isinstance(cloned, MoveOp)
                and isinstance(cloned.dst.type, GmemTensorType)
                and isinstance(cloned.src.type, WgrmemTensorType)
                and cloned.src.type.layout != 'wgmma_row'):
                writeback, _ = self._build_operand_writeback(
                    cloned.dst, cloned.src, 'dst', collective, label,
                    writeback_tiling_scheme=cast(int | None, cloned.attrs.get('dst_tiling_scheme')))
                result.extend(writeback)
                continue
            result.append(cloned)
        for dst, src in stores.items():
            if isinstance(dst.type, GmemTensorType) and isinstance(src.type, WgrmemTensorType) and src.type.layout != 'wgmma_row':
                # Consumer live-outs use the same scheduled writeback as the
                # producer. Row statistics retain their scalar stores: their
                # ownership and narrow destination are not dense matrix tiles.
                writeback, _ = self._build_operand_writeback(dst, src, 'dst', collective, label)
                result.extend(writeback)
            else:
                result.append(MoveOp.build(dst, src, collective=collective))
        return result, mapping

    def _promote_reduce(self, op: ReduceOp, epilogue: list[Op]) -> list[Op]:
        if self.target_type not in ("wrmem", "rmem") or self.operand_names != {"dst"}:
            raise NotImplementedError(
                "reduction promotion currently supports a register destination"
            )
        dst_type = cast(MemTensorType, op.dst.type)
        typ = self._build_target_type(dst_type)
        alloc = fresh_alloc(typ, op.dst)
        ops: list[Op] = [alloc]
        # Load and writeback have identical per-lane ownership.
        scheme = MemoryEffect.fresh_tiling_scheme_id()
        if op.init is None:
            loaded = alloc
            if dst_type.dtype != typ.dtype:
                loaded = fresh_alloc(
                    self._build_target_type(dst_type, dtype=dst_type.dtype), op.dst
                )
                ops.append(loaded)
            ops.append(
                MoveOp.build(
                    loaded.result(), op.dst, collective=op.collective,
                    attrs={"src_tiling_scheme": scheme},
                )
            )
            if loaded is not alloc:
                ops.append(
                    ElementwiseOp.build(alloc.result(), [loaded.result()], "convert")
                )
        ops.append(
            ReduceOp.build(
                alloc.result(), op.src, fn=op.fn,
                collective=op.collective, init=op.init, attrs=op.extra_attrs,
            )
        )
        assert isinstance(typ, RegMemTensorType)
        consumer_ops, mapping = self._promote_region(epilogue, {op.dst: alloc.result()}, typ, op.collective)
        ops.extend(consumer_ops)
        result = mapping[op.dst]
        if is_region_local(op.dst, {op, *epilogue}):
            return ops
        if dst_type.dtype != typ.dtype:
            converted = fresh_alloc(
                self._build_target_type(dst_type, dtype=dst_type.dtype), op.dst
            )
            ops.extend([
                converted,
                ElementwiseOp.build(converted.result(), [result], "convert"),
            ])
            result = converted.result()
        ops.append(
            MoveOp.build(
                op.dst, result, collective=op.collective,
                attrs={"dst_tiling_scheme": scheme},
            )
        )
        return ops

    def _is_register_writeback(self, op: Op) -> bool:
        """Whether ``op`` is the writeback half of a destination promotion.

        A schedule that builds its own register accumulators -- such as two
        matmuls feeding one epilogue -- writes the result back itself. That
        move still needs staging through SMEM when the store is a TMA.
        """
        return (
            isinstance(op, MoveOp)
            and "dst" in self.operand_names
            and self.tma
            and self.target_type == "wgrmem"
            and isinstance(op.src.type, WgrmemTensorType)
            and isinstance(op.dst.type, GmemTensorType)
        )

    def _epilogue(self, ops: list[Op], hero: int) -> list[Op]:
        """Select the explicitly scheduled tensor consumer region."""
        return select_region(ops, hero + 1, self.epilogue_label_filter)

    def _promote(
        self,
        op: MatmulOp,
        epilogue: list[Op],
        reusable: dict[tuple, Value],
    ) -> list[Op]:
        lhs, rhs, dst = op.lhs, op.rhs, op.dst

        label = op.attrs.get("label")
        assert label is None or isinstance(label, str)
        pre_ops: list[Op] = []
        new_lhs, new_rhs, new_dst = lhs, rhs, dst

        promoted_inputs = {"lhs": lhs, "rhs": rhs}
        for operand_name, value in list(promoted_inputs.items()):
            if operand_name not in self.operand_names:
                continue
            key = (operand_name, value, tuple(op.collective.items()))
            if self.reuse_operands and key in reusable:
                promoted_inputs[operand_name] = reusable[key]
            else:
                ops, [promoted] = self._build_operand_promotion(
                    value, operand_name, op.collective, label,
                )
                pre_ops.extend(ops)
                promoted_inputs[operand_name] = promoted
                if self.reuse_operands:
                    reusable[key] = promoted
        new_lhs, new_rhs = promoted_inputs["lhs"], promoted_inputs["rhs"]
        init = op.init
        if "dst" in self.operand_names:
            # A scalar initializer fills the promoted allocation; registers
            # load a tensor initializer in place of the destination. The
            # writeback below still targets the destination.
            scalar_init = (
                init
                if init is not None and cast(MemTensorType, init.type).rank == 0
                else None
            )
            tensor_init = (
                init if scalar_init is None and self.target_type == "wgrmem" else None
            )
            ops, [new_dst] = self._build_operand_promotion(
                dst if tensor_init is None else tensor_init,
                "dst",
                op.collective,
                label,
                init=scalar_init,
                overwrite=not op.accumulate and init is None,
            )
            pre_ops.extend(ops)
            # A staged SMEM destination is intermediate IR; its matmul still
            # owns initialization until the destination reaches registers.
            if tensor_init is not None or (
                scalar_init is not None and self.target_type != "smem"
            ):
                init = None

        if op.attrs.get("peeled") is True:
            for setup in pre_ops:
                if isinstance(setup, AllocOp):
                    setup.attrs["peeled"] = True

        matmul_attrs = op.extra_attrs
        for operand_name in self.operand_names:
            matmul_attrs.pop(f"{operand_name}_tiling_scheme", None)
        matmul = MatmulOp.build(
            lhs=new_lhs,
            rhs=new_rhs,
            dst=new_dst,
            collective=op.collective,
            init=init,
            attrs=matmul_attrs,
        )

        # The epilogue runs on the promoted buffer, before it is written back:
        # that is the whole point of promoting the destination, and it is only
        # reachable here because the ops behind the matmul came along.
        if isinstance(new_dst.type, RegMemTensorType):
            promoted_epilogue, mapping = self._promote_region(epilogue, {dst: new_dst}, new_dst.type, op.collective, label)
            output = mapping[dst]
        else:
            mapping = {dst: new_dst}
            promoted_epilogue = [clone_op(e, mapping) for e in epilogue]
            output = new_dst
        post_ops: list[Op] = list(promoted_epilogue)
        if "dst" in self.operand_names and not is_region_local(dst, {op, *epilogue}):
            writeback_tiling_scheme = cast(
                int | None, op.attrs.get("dst_tiling_scheme")
            )
            post_ops, _ = self._build_operand_writeback(
                dst,
                output,
                "dst",
                op.collective,
                label,
                promoted_epilogue,
                writeback_tiling_scheme,
            )

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
    permutation: list[int] | None = None,
    physical_layout: dict[str, object] | None = None,
    move_attrs: dict[str, dict[str, object]] | None = None,
    writeback_slices: int = 1,
    writeback_buffers: int = 1,
    unsafe_reuse_smem: bool = False,
    reuse_operands: bool = False,
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
        permutation=permutation,
        physical_layout=physical_layout,
        move_attrs=move_attrs,
        writeback_slices=writeback_slices,
        writeback_buffers=writeback_buffers,
        unsafe_reuse_smem=unsafe_reuse_smem,
        reuse_operands=reuse_operands,
        tma_load_slices=tma_load_slices,
        label_filter=label_filter,
        epilogue_label_filter=epilogue_label_filter,
    ).run(module)
