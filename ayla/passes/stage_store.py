"""Stage distributed output stores through round-robin SMEM arenas."""
from __future__ import annotations

import re

from ..base.ir import Block, Op, Value
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    GmemTensorType,
    KernelOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    WrmemTensorType,
)


class StageStore(Pass):
    """Rewrite matching WRMEM-to-GMEM moves as SMEM-staged TMA stores."""

    def __init__(self, label_filter: str, num_buffers: int) -> None:
        if num_buffers < 1:
            raise ValueError("stage_store: num_buffers must be positive")
        self.label_filter = re.compile(label_filter)
        self.num_buffers = num_buffers

    def run_op(self, op: Op) -> None:
        if isinstance(op, KernelOp):
            self._run_kernel(op)
            return
        super().run_op(op)

    def _run_kernel(self, kernel: KernelOp) -> None:
        stores: list[tuple[Block, MoveOp]] = []

        def collect(block: Block) -> None:
            for op in block.ops:
                if (
                    isinstance(op, MoveOp)
                    and self.label_filter.fullmatch(
                        str(op.attrs.get("label", ""))
                    )
                ):
                    stores.append((block, op))
                for region in op.regions:
                    for inner in region.blocks:
                        collect(inner)

        entry = kernel.body.entry_block
        collect(entry)
        if not stores:
            return

        full_dsts: list[Value] = []
        dst_tiles: list[TileOp] = []
        for _block, store in stores:
            if not (
                isinstance(store.src.type, WrmemTensorType)
                and isinstance(store.dst.type, GmemTensorType)
            ):
                raise ValueError(
                    "stage_store: expected matching WRMEM-to-GMEM moves"
                )
            dst_tile = store.dst.definition
            if not isinstance(dst_tile, TileOp):
                raise ValueError(
                    "stage_store: expected a warp-group-tiled GMEM destination"
                )
            full_dst = dst_tile.arg
            if not isinstance(full_dst.type, GmemTensorType):
                raise ValueError("stage_store: expected a GMEM tile root")
            dst_tiles.append(dst_tile)
            full_dsts.append(full_dst)

        first_type = full_dsts[0].type
        src_dtype = stores[0][1].src.type.dtype
        if any(
            dst.type.sizes != first_type.sizes
            or dst.type.dtype != first_type.dtype
            or store.src.type.dtype != src_dtype
            for dst, (_block, store) in zip(full_dsts, stores)
        ):
            raise ValueError(
                "stage_store: all stores must have compatible shapes and dtypes"
            )

        arena_count = min(self.num_buffers, len(stores))
        stage_type = SmemTensorType.build(
            sizes=list(first_type.sizes), dtype=src_dtype, swizzle=None,
        )
        arenas = [AllocOp.build(stage_type) for _ in range(arena_count)]
        entry.ops[0:0] = arenas

        for ordinal, ((block, store), dst_tile, full_dst) in enumerate(
            zip(stores, dst_tiles, full_dsts)
        ):
            label = str(store.attrs.get("label", f"store_{ordinal}"))
            arena = arenas[ordinal % arena_count]
            stage_tile = TileOp.build(
                arena.result(), indices=list(dst_tile.indices),
                type=SmemTensorType.build(
                    sizes=list(store.dst.type.sizes),
                    dtype=store.src.type.dtype, swizzle=None,
                ),
            )
            gather = MoveOp.build(
                dst=stage_tile.result(), src=store.src,
                attrs={"label": f"{label}_stage"},
            )
            tma_store = MoveOp.build(
                dst=full_dst, src=arena.result(),
                attrs={"label": f"{label}_tma", "tma": True},
            )
            pos = block.ops.index(store)
            block.ops[pos:pos + 1] = [stage_tile, gather, tma_store]


def stage_store(
    module: ModuleOp, label_filter: str, num_buffers: int,
) -> ModuleOp:
    """Convenience entry point."""
    return StageStore(label_filter, num_buffers).run(module)
