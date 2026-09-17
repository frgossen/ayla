"""Lower a TMA ``move`` to a descriptor-backed synchronous ``tma_move``.

Creates or reuses a descriptor for the tiled gmem operand's root and retains
the original gmem/smem views on ``tma_move``::

- ``move`` (tma) → ``tma_move(gmem view, smem view, descriptor)``

Tile-chain coordinates are deliberately not materialized here; PTX-template
lowering derives them from the retained view. Non-TMA moves are left untouched.
"""
from __future__ import annotations

from typing import cast

from ..gpu_dialect import (
    GmemTensorType,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    TmaDescType,
    TmaMoveOp,
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
        is_load = isinstance(op.dst.type, SmemTensorType)
        gmem_operand = op.src if is_load else op.dst
        smem_operand = op.dst if is_load else op.src
        gmem = self.value_map[gmem_operand]
        smem = self.value_map[smem_operand]
        assert isinstance(gmem, Value)
        assert isinstance(smem, Value)

        root = gmem
        while isinstance(root.definition, TileOp):
            root = root.definition.arg

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
        tma_type = TmaDescType.build(
            arg_idx=root.result_index,
            gmem_type=root.type,
            tile_shape=list(smem.type.sizes),
            swizzle=smem.type.swizzle,
            l2_promotion=cast(str | None, op.attrs.get("l2_promotion")),
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

        attrs = dict(op.attrs)
        l2_cache_hint = attrs.pop("l2_cache_hint", None)
        dst, src = (smem, gmem) if is_load else (gmem, smem)
        return [TmaMoveOp.build(
            dst=dst,
            src=src,
            descriptor=desc_value,
            collective=op.collective,
            attrs=attrs,
            l2_cache_hint=l2_cache_hint,
        )], []


def lower_to_tma_move(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerToTmaMove().run(module)
