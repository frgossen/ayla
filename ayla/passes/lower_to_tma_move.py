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

from typing import cast

from ..gpu_dialect import (
    AsyncOp,
    GmemTensorType,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    ModuleOp,
    MoveOp,
    ReshapeOp,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
    is_side_effect_free,
)
from ..base.ir import Block, Op, Value
from ..base.lowering_pass import LoweringPass, lowers


class LowerToTmaMove(LoweringPass):

    @lowers(MoveOp)
    def lower_move(
        self, op: MoveOp,
    ) -> tuple[list[Op], list[object]] | None:
        if not op.attrs.get("tma"):
            return None
        index_ops, tma_op = self._materialize(op)
        return index_ops + [tma_op], []

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

    def _materialize(self, move: MoveOp) -> tuple[list[Op], TmaMoveOp]:
        """Build the descriptor + index ops and a sync ``tma_move`` for ``move``."""
        is_load = isinstance(move.dst.type, SmemTensorType)
        gmem_operand = move.src if is_load else move.dst
        smem_operand = move.dst if is_load else move.src

        smem = cast(Value, self.value_map[smem_operand])

        assert isinstance(smem.type, SmemTensorType), (
            f"lower_to_tma_move: plain operand must be smem; got {smem.type}"
        )

        tile_shape = list(smem.type.sizes)
        assert all(isinstance(size, int) for size in tile_shape)
        index_ops, root, indices, tile_shape = _build_tile_chain_indices(
            gmem_operand, cast(list[int], tile_shape),
        )
        assert isinstance(root.type, GmemTensorType)

        entry_block = root.definition
        assert isinstance(entry_block, Block)

        layout = smem.type.attrs.get("physical_layout")
        desc_attrs: dict[str, object] = {}
        if layout is not None:
            desc_attrs["physical_layout"] = layout
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


def _build_tile_chain_indices(
    value: Value, tile_shape: list[int],
) -> tuple[list[Op], Value, list[Value], list[int]]:
    """Walk up tiled/singleton-reshaped views to the root tensor.

    TMA descriptors describe the root tensor, so coordinates and the copied
    box must use the root rank. A reshape that only adds or removes singleton
    dimensions preserves every non-singleton coordinate and rectangular tile.
    """
    zero = IndexConstantOp.build(0)
    new_ops: list[Op] = [zero]
    value_type = value.type
    assert isinstance(value_type, GmemTensorType)
    indices = [zero.result()] * value_type.rank
    v = value

    while isinstance(v.definition, (TileOp, ReshapeOp)):
        definition = v.definition
        if isinstance(definition, ReshapeOp):
            result_type = definition.result().type
            arg_type = definition.arg.type
            assert isinstance(result_type, GmemTensorType)
            assert isinstance(arg_type, GmemTensorType)
            result_sizes = list(result_type.sizes)
            arg_sizes = list(arg_type.sizes)
            assert all(isinstance(size, int) for size in result_sizes)
            assert all(isinstance(size, int) for size in arg_sizes)
            result_nonunit = [
                i for i, size in enumerate(result_sizes) if size != 1
            ]
            arg_nonunit = [i for i, size in enumerate(arg_sizes) if size != 1]
            if [result_sizes[i] for i in result_nonunit] != [
                arg_sizes[i] for i in arg_nonunit
            ]:
                raise ValueError(
                    "lower_to_tma_move: reshape must only add or remove "
                    "singleton dimensions"
                )
            mapped_indices = [zero.result()] * len(arg_sizes)
            mapped_shape = [1] * len(arg_sizes)
            for result_dim, arg_dim in zip(
                result_nonunit, arg_nonunit, strict=True,
            ):
                mapped_indices[arg_dim] = indices[result_dim]
                mapped_shape[arg_dim] = tile_shape[result_dim]
            indices = mapped_indices
            tile_shape = mapped_shape
            v = definition.arg
            continue

        result_type = definition.result().type
        assert isinstance(result_type, GmemTensorType)
        result_sizes = list(result_type.sizes)
        assert all(isinstance(size, int) for size in result_sizes)
        for dim, idx in enumerate(definition.indices):
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
        v = definition.arg

    return new_ops, v, indices, tile_shape


def lower_to_tma_move(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToTmaMove().run(module)
