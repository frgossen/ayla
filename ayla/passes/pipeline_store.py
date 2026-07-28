"""Stage WGMMA output stores through a software-pipelined TMA path.

The store-specific rewrite only introduces a compact shared-memory output tile
and an ordinary TMA ``move``. The generic async passes then form the async op
and schedule its wait immediately before a staging bank is reused.
"""
from __future__ import annotations

import re

from ..base.ir import Block
from ..base.pipeline import Pass, Pipeline
from ..gpu_dialect import (
    AllocOp,
    ModuleOp,
    MoveOp,
    SmemTensorType,
    TileOp,
)
from .fuse_async import FuseAsync
from .lower_to_async import LowerToAsync
from .schedule_async import ScheduleAsync


class StageStore(Pass):
    """Rewrite matching wrmem-to-gmem stores as smem-striped TMA moves."""

    def __init__(
        self, label_filter: str, num_buffers: int,
    ) -> None:
        if num_buffers < 1:
            raise ValueError("pipeline_store: num_buffers must be positive")
        self.label_filter = re.compile(label_filter)
        self.num_buffers = num_buffers

    def run_block(self, block: Block) -> None:
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
        stage_type = SmemTensorType.build(
            sizes=list(full_dst.type.sizes),
            dtype=stores[0].src.type.dtype,
            swizzle=None,
        )
        stage_buffers = [
            AllocOp.build(stage_type)
            for _ in range(self.num_buffers)
        ]
        first_pos = min(block.ops.index(op) for op in stores)
        block.ops[first_pos:first_pos] = stage_buffers

        for ordinal, store in enumerate(stores):
            dst_tile = store.dst.definition
            if not isinstance(dst_tile, TileOp):
                raise ValueError("pipeline_store: expected a WG-tiled gmem destination")
            label = str(store.attrs.get("label", f"store_{ordinal}"))
            stage_alloc = stage_buffers[ordinal % self.num_buffers]
            stage_wg_type = SmemTensorType.build(
                sizes=list(store.dst.type.sizes),
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
                attrs={"label": f"{label}_stage"},
            )
            tma_store = MoveOp.build(
                dst=dst_tile.arg,
                src=stage_alloc.result(),
                attrs={"label": f"{label}_tma", "tma": True},
            )

            pos = block.ops.index(store)
            block.ops[pos:pos + 1] = [stage_wg, stage, tma_store]


class PipelineStore(Pipeline):
    """Compose store staging with the compiler's generic async framework."""

    def __init__(
        self,
        label_filter: str,
        num_buffers: int = 1,
    ) -> None:
        super().__init__()
        self.add_pass(StageStore(label_filter, num_buffers))
        self.add_pass(LowerToAsync())
        self.add_pass(FuseAsync())
        self.add_pass(ScheduleAsync())


def pipeline_store(
    module: ModuleOp,
    label_filter: str,
    num_buffers: int = 1,
) -> ModuleOp:
    return PipelineStore(label_filter, num_buffers).run(module)
