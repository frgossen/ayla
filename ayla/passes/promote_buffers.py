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

from ..base.ir import Block, Op, Value
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AllocOp,
    BF16Type,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    F16Type,
    F32Type,
    F64Type,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
    MoveOp,
    RmemTensorType,
    SmemTensorType,
    TileOp,
    WrmemTensorType,
    is_elementwise,
)
from ._utils import (
    build_slice_tile,
    label_matches,
    rebuild_tile,
    tile_introduces_narrow_level,
    value_collective_level,
)


def _target_level(target_type: str) -> CollectiveLevel:
    if target_type == "smem":
        return CollectiveLevel.BLOCK
    if target_type == "wrmem":
        return CollectiveLevel.WG
    if target_type == "rmem":
        return CollectiveLevel.THREAD
    if target_type == "gmem":
        return CollectiveLevel.BLOCK
    raise ValueError(f"unsupported target type {target_type!r}")


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


def _sliced_moves(
    dst: Value,
    src: Value,
    count: int,
    dim: int,
    collective: dict[CollectiveAxis, CollectiveLevel],
    attrs: dict[str, object],
) -> list[Op]:
    if count == 1:
        return [MoveOp.build(
            dst, src, collective=collective, attrs=attrs,
        )]
    size, remainder = divmod(src.type.sizes[dim], count)
    if remainder:
        raise ValueError("move extent must divide evenly into slices")
    ops: list[Op] = []
    for index in range(count):
        src_tile, dst_tile = [
            build_slice_tile(v, dim, index, size) for v in (src, dst)
        ]
        move = MoveOp.build(
            dst_tile.result(), src_tile.result(),
            collective=collective, attrs=attrs,
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
        Mark moves for TMA lowering. Register writebacks remain semantic moves
        for ``StageRegisterWriteback`` to materialize.
    nc : bool
        Non-coherent cache hint for scalar gmem→rmem loads.
    physical_layout : dict or None
        Physical layout attrs for promoted shared-memory buffers.
    move_attrs : dict[str, dict[str, object]] or None
        Extra move attributes by operand, such as a TMA ``l2_cache_hint``.
    unsafe_reuse_smem : bool
        Mark shared operand allocations by their asserted compute lifetime so
        a later arena-packing pass may overlay their storage. This is unsafe:
        no analysis verifies that differently named groups are non-overlapping.
        TODO: Replace this policy flag with independent allocations followed by
        lifetime analysis and a pass that selects safe storage reuse.
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
        unsafe_reuse_smem: bool = False,
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
            physical_layout=physical_layout,
            move_attrs=dict(move_attrs or {}),
            unsafe_reuse_smem=unsafe_reuse_smem,
            tma_load_slices=dict(tma_load_slices or {}),
            label_filter=label_filter,
            epilogue_label_filter=epilogue_label_filter,
        )
        assert all(n >= 1 for n in self.tma_load_slices.values()), (
            "TMA load slice counts must be positive"
        )
        self.operand_names = set(operands)

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
        collective: dict[CollectiveAxis, CollectiveLevel],
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
            if min(
                value_collective_level(tile_op.result()), target_level,
            ) == target_level:
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
            if not tile_introduces_narrow_level(tile, target_level):
                rebuilt = rebuild_tile(tile, chain_root)
                new_ops.append(rebuilt)
                chain_root = rebuilt.result()

        # Alloc + move at the target level.
        target_type = self._build_target_type(chain_root.type)
        # The allocation carries the same label as the move that fills it, so a
        # later pass can name the buffer it wants -- multi_buffer only touches
        # labelled ones.
        buffer_attrs = self._build_buffer_attrs(matmul_label, operand_name)
        if init is not None:
            # SMEM initialization has no lowering. Keep the initialization on
            # the matmul so a subsequent register promotion can materialize it.
            alloc = AllocOp.build_with_fresh_id(
                target_type,
                init=None if self.target_type == "smem" else init,
                attrs=buffer_attrs,
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
                self.tma_load_slices.get(operand_name, 1), 0,
                collective, attrs,
            ))
            if target_type.dtype != chain_root.type.dtype:
                cvt_alloc = AllocOp.build_with_fresh_id(target_type)
                cvt = ElementwiseOp.build(
                    dst=cvt_alloc.result(), srcs=[alloc.result()], fn="convert",
                )
                new_ops.extend([cvt_alloc, cvt])
                alloc = cvt_alloc

        # Replay narrow-level tiles on the promoted buffer (e.g. tile smem by wg_id).
        result = alloc.result()
        for tile in reversed(chain):
            if tile_introduces_narrow_level(tile, target_level):
                rebuilt = rebuild_tile(tile, result)
                new_ops.append(rebuilt)
                result = rebuilt.result()

        return new_ops, [result]

    def _build_operand_writeback(
        self,
        dst: Value,
        src: Value,
        operand_name: str,
        collective: dict[CollectiveAxis, CollectiveLevel],
        matmul_label: str | None = None,
        epilogue: list[ElementwiseOp] | None = None,
    ) -> tuple[list[Op], list[Value]]:
        target_level = _target_level(self.target_type)
        dst_level = value_collective_level(dst)
        if min(dst_level, target_level) != target_level:
            raise NotImplementedError(
                f"writeback for {dst_level.name.lower()}-level dst not yet "
                f"supported "
                f"when promoting to {self.target_type}"
        )
        new_ops: list[Op] = []
        epilogue = list(epilogue or [])
        if src.type.dtype.name != dst.type.dtype.name:
            tmp_type = self._build_target_type(dst.type, dtype=dst.type.dtype)
            tmp_alloc = AllocOp.build_with_fresh_id(tmp_type)
            new_ops.extend(epilogue[:-1])
            new_ops.append(tmp_alloc)
            if epilogue:
                final = epilogue[-1]
                # TODO: Generalize this rewrite for multi-source element-wise
                # epilogues such as SwiGLU. Retarget the final operation's
                # destination while preserving and remapping all its sources.
                assert final.dst is src and final.src(0) is src
                new_ops.append(ElementwiseOp.build(
                    dst=tmp_alloc.result(), srcs=[src], fn=final.fn,
                    attrs=final.attrs,
                ))
            else:
                new_ops.append(ElementwiseOp.build(
                    dst=tmp_alloc.result(), srcs=[src], fn="convert",
                    attrs=(
                        {"label": f"{matmul_label}_{operand_name}_convert"}
                        if matmul_label is not None else None
                    ),
                ))
            src = tmp_alloc.result()
        else:
            new_ops.extend(epilogue)
        attrs = self._build_move_attrs(matmul_label, operand_name, allow_nc=False)
        move = MoveOp.build(
            dst=dst, src=src, collective=collective, attrs=attrs,
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
        has to name them: promotion may leave an ``elementwise`` conversion
        behind an accumulator.
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
            ops, [new_lhs] = self._build_operand_promotion(
                lhs, "lhs", op.collective, label,
            )
            pre_ops.extend(ops)
        if "rhs" in self.operand_names:
            ops, [new_rhs] = self._build_operand_promotion(
                rhs, "rhs", op.collective, label,
            )
            pre_ops.extend(ops)
        init = op.init
        if "dst" in self.operand_names:
            ops, [new_dst] = self._build_operand_promotion(
                dst, "dst", op.collective, label, init=init,
            )
            pre_ops.extend(ops)
            # A staged SMEM destination is intermediate IR; its matmul still
            # owns initialization until the destination reaches registers.
            if self.target_type != "smem":
                init = None

        matmul = MatmulOp.build(
            lhs=new_lhs,
            rhs=new_rhs,
            dst=new_dst,
            collective=op.collective,
            init=init,
            attrs=op.extra_attrs,
        )

        # The epilogue runs on the promoted buffer, before it is written back:
        # that is the whole point of promoting the destination, and it is only
        # reachable here because the ops behind the matmul came along.
        post_ops: list[ElementwiseOp] = [
            type(e)(operands=[new_dst, new_dst], results=[], attrs=e.attrs)
            for e in epilogue
        ]
        if "dst" in self.operand_names:
            post_ops, _ = self._build_operand_writeback(
                dst, new_dst, "dst", op.collective, label, post_ops,
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
    physical_layout: dict[str, object] | None = None,
    move_attrs: dict[str, dict[str, object]] | None = None,
    unsafe_reuse_smem: bool = False,
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
        unsafe_reuse_smem=unsafe_reuse_smem,
        tma_load_slices=tma_load_slices,
        label_filter=label_filter,
        epilogue_label_filter=epilogue_label_filter,
    ).run(module)
