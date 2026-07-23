"""Lower a TMA ``move`` to a (sync) ``tma_move`` by walking up the tile chain.

Resolves the tiled gmem operand into a TMA descriptor (a new kernel argument)
plus per-dim element indices, and replaces the move with index-arithmetic ops
followed by a sync ``tma_move``::

- ``move`` (tma) → ``index ops…``, ``tma_move``

A move wrapped in ``async { move (tma) }`` is lowered in place by the generic
region recursion, leaving ``async { index ops…; tma_move }``. A later ``licm``
pass hoists the (side-effect-free) index ops back out of the region so the body
returns to a single op. This keeps the pass order-independent w.r.t.
``lower_to_async``. Non-tma moves and other ops are left untouched.
"""
from __future__ import annotations

from ..gpu_dialect import (
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
)
from ..base.ir import Block, Op, Value
from ..base.lowering_pass import LoweringPass


class LowerToTmaMove(LoweringPass):

    def __init__(self) -> None:
        super().__init__()
        self.register(MoveOp, type(self).lower_move)

    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not op.attrs.get("tma"):
            return None
        index_ops, tma_op = self._materialize(op)
        return index_ops + [tma_op], []

    def _materialize(self, move: MoveOp) -> tuple[list[Op], TmaMoveOp]:
        """Build the descriptor + index ops and a sync ``tma_move`` for ``move``."""
        is_load = isinstance(move.dst.type, SmemTensorType)
        gmem_operand = move.src if is_load else move.dst
        smem_operand = move.dst if is_load else move.src

        index_ops, root, indices = _build_tile_chain_indices(gmem_operand)
        smem = self.value_map[smem_operand]

        assert isinstance(root.type, GmemTensorType), (
            f"lower_to_tma_move: tiled operand must be gmem; got {root.type}"
        )
        assert isinstance(smem.type, SmemTensorType), (
            f"lower_to_tma_move: plain operand must be smem; got {smem.type}"
        )

        entry_block = root.definition
        assert isinstance(entry_block, Block)

        tma_type = TmaDescType.build(
            arg_idx=root.result_index,
            gmem_type=root.type,
            tile_shape=list(smem.type.sizes),
            swizzle=smem.type.swizzle,
        )
        desc_value = Value(
            name=f"{root.name}_desc" if root.name else None,
            type=tma_type,
        )
        entry_block.append_argument(desc_value)

        if is_load:
            tma_op = TmaMoveOp.build(
                dst=smem, src=desc_value, indices=indices, attrs=dict(move.attrs))
        else:
            tma_op = TmaMoveOp.build(
                dst=desc_value, src=smem, indices=indices, attrs=dict(move.attrs))
        return index_ops, tma_op


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

    ndims = root.type.rank
    zero = IndexConstantOp.build(0)
    new_ops: list[Op] = [zero]
    indices = [zero.result()] * ndims

    for tile_op in reversed(tiles):
        result_sizes = list(tile_op.result().type.sizes)
        for dim, idx in enumerate(tile_op.indices):
            if isinstance(idx, int):
                const_op = IndexConstantOp.build(idx)
                new_ops.append(const_op)
                idx = const_op.result()

            tile_size_op = IndexConstantOp.build(result_sizes[dim])
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
