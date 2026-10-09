"""Lower a TMA ``move`` to a (sync) ``tma_move`` by walking up the tile chain.

Resolves the tiled gmem operand into a TMA descriptor (a new kernel argument)
plus per-dim element indices, and replaces the move with index-arithmetic ops
followed by a sync ``tma_move``::

- ``move`` (tma) → ``index ops…``, ``tma_move``

A move wrapped in ``async { move (tma) }`` is lowered so the coordinate index
arithmetic lands *before* the enclosing ``async`` op and only the ``tma_move``
stays in the body, leaving ``index ops…; async { tma_move }``. The index math is
plain, side-effect-free arithmetic with no reason to sit inside the mbarrier
block, and keeping it out gives ``lower_async`` (PTX lowering) the single-op
``tma_move`` body it expects. Non-tma moves and other ops are left untouched.
"""
from __future__ import annotations

import math
from typing import cast

from ..gpu_dialect import (
    AsyncOp,
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    KernelOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    is_side_effect_free,
)
from ..base.ir import Block, Op, Value, find_ancestor_of_type
from ..base.lowering_pass import LoweringPass, lowers

# A swizzled TMA box must start on a whole swizzle atom: eight rows of the
# swizzle width.
_SWIZZLE_ATOM_BYTES = {"128B": 1024, "64B": 512, "32B": 256}


class LowerToTmaMove(LoweringPass):

    @lowers(MoveOp)
    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not op.attrs.get("tma"):
            return None
        slices = _multicast_slices(op, cast(Value, self.value_map[op.dst]))
        if slices == 1:
            index_ops, tma_op = self._materialize(op)
            return index_ops + [tma_op], []
        # Each cluster CTA can issue one row slice of a multicast tile, so
        # one TMA unit does not carry the whole tile for the cluster.
        smem = cast(Value, self.value_map[op.dst])
        smem_type = cast(SmemTensorType, smem.type)
        rows, columns = cast(tuple[int, int], tuple(smem_type.sizes))
        slice_type = SmemTensorType.build(
            (rows // slices, columns),
            smem_type.dtype,
            swizzle=smem_type.swizzle,
        )
        new_ops: list[Op] = []
        for index in range(slices):
            view = TileOp.build(smem, [index, 0], slice_type)
            index_ops, tma_op = self._materialize(
                op, smem=view.result(), row_offset=index * (rows // slices),
            )
            new_ops += [view, *index_ops, tma_op]
        return new_ops, []

    @lowers(AsyncOp)
    def lower_async(
        self, op: AsyncOp,
    ) -> tuple[list[Op], list[object]]:
        """Lower the async body, hoisting materialized index ops before it.

        Lowering a body tma ``move`` yields coordinate index arithmetic plus a
        ``tma_move``. That arithmetic is side-effect-free and has no reason to
        sit inside the mbarrier block, so it moves ahead of the async op,
        leaving the body as just the ``tma_move``(s) that ``lower_async`` (PTX
        lowering) expects.
        """
        op.operands = [cast(Value, self.value_map[v]) for v in op.operands]
        body = op.body
        self.lower_block(body)
        hoisted = [o for o in body.ops if is_side_effect_free(o)]
        body.ops = [o for o in body.ops if not is_side_effect_free(o)]
        return hoisted + [op], list(op.results)

    def _materialize(
        self,
        move: MoveOp,
        smem: Value | None = None,
        row_offset: int = 0,
    ) -> tuple[list[Op], TmaMoveOp]:
        """Build the descriptor + index ops and a sync ``tma_move`` for ``move``.

        ``smem`` and ``row_offset`` select a row slice of a load's tile.
        """
        is_load = isinstance(move.dst.type, SmemTensorType)
        gmem_operand = move.src if is_load else move.dst
        smem_operand = move.dst if is_load else move.src

        index_ops, root, indices = _build_tile_chain_indices(gmem_operand)
        if row_offset:
            offset = IndexConstantOp.build(row_offset)
            row = IndexAddOp.build(indices[0], offset.result())
            index_ops += [offset, row]
            indices = [row.result(), *indices[1:]]
        if smem is None:
            smem = cast(Value, self.value_map[smem_operand])

        assert isinstance(root.type, GmemTensorType), (
            f"lower_to_tma_move: tiled operand must be gmem; got {root.type}"
        )
        assert isinstance(smem.type, SmemTensorType), (
            f"lower_to_tma_move: plain operand must be smem; got {smem.type}"
        )

        entry_block = root.definition
        assert isinstance(entry_block, Block)

        layout = smem.type.attrs.get("physical_layout")
        desc_attrs: dict[str, object] = {}
        if layout is not None:
            desc_attrs["physical_layout"] = layout
        tile_shape = list(smem.type.sizes)
        assert all(isinstance(size, int) for size in tile_shape)
        tma_type = TmaDescType.build(
            arg_idx=root.result_index,
            gmem_type=root.type,
            tile_shape=cast(list[int], tile_shape),
            swizzle=smem.type.swizzle,
            l2_promotion=cast(str | None, move.attrs.get("l2_promotion")),
            attrs=desc_attrs or None,
        )
        desc_value = next(
            (
                arg for arg in entry_block.arguments
                if isinstance(arg.type, TmaDescType) and arg.type == tma_type
            ),
            None,
        )
        if desc_value is None:
            desc_value = Value(
                name=f"{root.name}_desc" if root.name else None,
                type=tma_type,
            )
            entry_block.append_argument(desc_value)

        attrs = move.extra_attrs
        l2_cache_hint = cast(str | None, attrs.pop("l2_cache_hint", None))
        if is_load:
            tma_op = TmaMoveOp.build(
                dst=smem,
                src=desc_value,
                indices=indices,
                collective=move.collective,
                attrs=attrs,
                l2_cache_hint=l2_cache_hint,
                multicast=move.multicast,
            )
        else:
            tma_op = TmaMoveOp.build(
                dst=desc_value,
                src=smem,
                indices=indices,
                collective=move.collective,
                attrs=attrs,
                l2_cache_hint=l2_cache_hint,
                multicast=move.multicast,
            )
        return index_ops, tma_op


def _multicast_slices(move: MoveOp, smem: Value) -> int:
    """How many row slices the cluster's CTAs issue a multicast load in.

    A slice must cover whole swizzle atoms, and the tile must be a plain
    row-major 2-D buffer; otherwise one CTA issues the whole tile.
    """
    if move.multicast is not True:
        return 1
    kernel = find_ancestor_of_type(move, KernelOp)
    assert kernel is not None
    cluster_size = math.prod(dim or 1 for dim in kernel.cluster_dim)
    smem_type = smem.type
    if (
        cluster_size < 2
        or not isinstance(smem_type, SmemTensorType)
        or smem_type.rank != 2
        or "physical_layout" in smem_type.attrs
        or not smem_type.is_canonical_strides
    ):
        return 1
    rows, columns = smem_type.sizes
    assert isinstance(rows, int) and isinstance(columns, int)
    dtype_bytes = cast(
        BF16Type | F16Type | F32Type | F64Type, smem_type.dtype,
    ).byte_size
    atom = _SWIZZLE_ATOM_BYTES.get(cast(str, smem_type.swizzle), 0)
    slice_rows, remainder = divmod(rows, cluster_size)
    if remainder or not atom or slice_rows * columns * dtype_bytes % atom:
        return 1
    return cluster_size


def _build_tile_chain_indices(value: Value) -> tuple[list[Op], Value, list[Value]]:
    """Walk up nested tile ops to find the root tensor and per-dim indices.

    Returns (new_ops, root, indices) where new_ops are the index
    arithmetic ops, root is the base tensor, and indices are the
    accumulated per-dim element offsets.
    """
    tiles: list[TileOp] = []
    v = value
    while isinstance(v.definition, TileOp):
        tiles.append(v.definition)
        v = v.definition.arg
    root = v

    root_type = root.type
    assert isinstance(root_type, GmemTensorType)
    ndims = root_type.rank
    zero = IndexConstantOp.build(0)
    new_ops: list[Op] = [zero]
    indices = [zero.result()] * ndims

    for tile_op in reversed(tiles):
        result_type = tile_op.result().type
        assert isinstance(result_type, GmemTensorType)
        result_sizes = list(result_type.sizes)
        assert all(isinstance(size, int) for size in result_sizes)
        for dim, idx in enumerate(tile_op.indices):
            if isinstance(idx, int):
                const_op = IndexConstantOp.build(idx)
                new_ops.append(const_op)
                idx = const_op.result()

            tile_size_op = IndexConstantOp.build(
                cast(int, result_sizes[dim]),
            )
            new_ops.append(tile_size_op)
            mul_op = IndexMulOp.build(idx, tile_size_op.result())
            new_ops.append(mul_op)
            add_op = IndexAddOp.build(indices[dim], mul_op.result())
            new_ops.append(add_op)
            indices[dim] = add_op.result()

    return new_ops, root, indices


def lower_to_tma_move(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToTmaMove().run(module)
