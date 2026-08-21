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

from ..gpu_dialect import (
    AsyncOp,
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
    is_side_effect_free,
)
from ..base.ir import Block, Op, Value
from ..base.lowering_pass import LoweringPass


class LowerToTmaMove(LoweringPass):

    def __init__(self) -> None:
        super().__init__()
        self.register(MoveOp, type(self).lower_move)
        self.register(AsyncOp, type(self).lower_async)

    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not op.attrs.get("tma"):
            return None
        index_ops, tma_op = self._materialize(op)
        return index_ops + [tma_op], []

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
        op.operands = [self.value_map[v] for v in op.operands]
        body = op.body
        self.lower_block(body)
        hoisted = [o for o in body.ops if is_side_effect_free(o)]
        body.ops = [o for o in body.ops if not is_side_effect_free(o)]
        return hoisted + [op], list(op.results)

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

        layout = smem.type.attrs.get("physical_layout")
        tma_type = TmaDescType.build(
            arg_idx=root.result_index,
            gmem_type=root.type,
            tile_shape=list(smem.type.sizes),
            swizzle=smem.type.swizzle,
            attrs=(
                {"physical_layout": layout}
                if layout is not None else None
            ),
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
