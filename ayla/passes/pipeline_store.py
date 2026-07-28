"""Pipeline WGMMA output stores across sequential output tiles.

Rewrites labeled per-warpgroup ``wrmem -> gmem`` moves into compact shared
memory stripes followed by asynchronous TMA stores. A single set of staging
buffers is reused, and each tile's store waits are delayed across the next
tile's compute but remain before that tile overwrites the buffers.
"""
from __future__ import annotations

import re

from ..base.ir import Block, Region
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    GmemTensorType,
    LockType,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
)


class PipelineStore(Pass):
    def __init__(self, label_filter: str, stripe_width: int = 64) -> None:
        self.label_filter = re.compile(label_filter)
        self.stripe_width = stripe_width

    def run_block(self, block: Block) -> None:
        # Transform inner blocks first; output-loop unrolling usually leaves
        # the epilogues together in one block, but recursion keeps this pass
        # independent of that canonical form.
        super().run_block(block)
        stores = [
            op for op in block.ops
            if isinstance(op, MoveOp)
            and self.label_filter.fullmatch(str(op.attrs.get("label", "")))
        ]
        if len(stores) < 2:
            return

        first_dst_tile = stores[0].dst.definition
        if not isinstance(first_dst_tile, TileOp):
            raise ValueError("pipeline_store: expected a WG-tiled gmem destination")
        full_dst = first_dst_tile.arg
        tile_n = full_dst.type.sizes[1]
        if tile_n % self.stripe_width:
            raise ValueError(
                f"pipeline_store: N={tile_n} is not divisible by "
                f"stripe_width={self.stripe_width}"
            )

        stripe_type = SmemTensorType.build(
            sizes=[full_dst.type.sizes[0], self.stripe_width],
            dtype=stores[0].src.type.dtype,
            swizzle=None,
        )
        stage_allocs = [
            AllocOp.build(stripe_type)
            for _ in range(tile_n // self.stripe_width)
        ]
        first_pos = min(block.ops.index(op) for op in stores)
        block.ops[first_pos:first_pos] = stage_allocs

        records: list[tuple[MoveOp, AwaitOp]] = []
        for ordinal, store in enumerate(stores):
            dst_tile = store.dst.definition
            assert isinstance(dst_tile, TileOp)
            label = str(store.attrs.get("label", f"store_{ordinal}"))
            lock = AllocOp.build(LockType.build(kind="bulk"))
            replacement = [lock]
            first_stage = None
            for stripe, stage_alloc in enumerate(stage_allocs):
                stage_wg_type = SmemTensorType.build(
                    sizes=[store.dst.type.sizes[0], self.stripe_width],
                    dtype=store.src.type.dtype,
                    swizzle=None,
                )
                stage_wg = TileOp.build(
                    stage_alloc.result(),
                    indices=list(dst_tile.indices),
                    type=stage_wg_type,
                )
                stage = MoveOp.build(
                    dst=stage_wg.result(),
                    src=store.src,
                    attrs={
                        "label": f"{label}_stage_s{stripe}",
                        "wgmma_n_offset": stripe * self.stripe_width,
                    },
                )
                if stripe == 0:
                    first_stage = stage
                gmem_stripe_type = GmemTensorType.build(
                    sizes=[dst_tile.arg.type.sizes[0], self.stripe_width],
                    dtype=dst_tile.arg.type.dtype,
                    strides=list(dst_tile.arg.type.strides),
                )
                gmem_stripe = TileOp.build(
                    dst_tile.arg,
                    indices=[0, stripe],
                    type=gmem_stripe_type,
                )
                tma_store = MoveOp.build(
                    dst=gmem_stripe.result(),
                    src=stage_alloc.result(),
                    attrs={"label": f"{label}_tma_s{stripe}", "tma": True},
                )
                async_store = AsyncOp.build(
                    lock.result(),
                    body=Region(blocks=[Block(ops=[tma_store])]),
                    attrs={"label": f"{label}_tma_s{stripe}_async"},
                )
                replacement.extend([
                    stage_wg, stage, gmem_stripe, async_store,
                ])
            wait = AwaitOp.build(lock.result())
            replacement.append(wait)
            assert first_stage is not None
            pos = block.ops.index(store)
            block.ops[pos:pos + 1] = replacement
            records.append((first_stage, wait))

        for (_, wait), (next_stage, _) in zip(records, records[1:]):
            block.ops.remove(wait)
            pos = block.ops.index(next_stage)
            block.ops.insert(pos, wait)


def pipeline_store(
    module: ModuleOp,
    label_filter: str,
    stripe_width: int = 64,
) -> ModuleOp:
    return PipelineStore(label_filter, stripe_width).run(module)
