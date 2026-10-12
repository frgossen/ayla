"""Compose reduction tiling and buffer promotion."""

from ..base.pipeline import Pipeline
from ..gpu_dialect import CollectiveAxis, CollectiveLevel
from ..passes.canonicalize import Canonicalize
from ..passes.promote_buffers import PromoteBuffers
from ..passes.tile_reduce import TileReduce


class TileReducePipeline(Pipeline):
    """Tile mean reductions by block and accumulate vectors in registers."""

    def __init__(self) -> None:
        super().__init__()
        self.add_pass(TileReduce(
            tile_sizes={0: 1},
            parallel_iteration_space=(
                CollectiveLevel.BLOCK, CollectiveAxis.X,
            ),
            hero_label_filter="mean",
        ))
        self.add_pass(Canonicalize())
        self.add_pass(PromoteBuffers(
            target_type="wrmem",
            operands=["dst"],
            layout="blocked",
            label_filter="mean",
        ))
        self.add_pass(Canonicalize())
        self.add_pass(TileReduce(
            tile_sizes={1: 128},
            parallel_iteration_space=None,
            hero_label_filter="mean",
            epilogue_label_filter="mean_epilogue",
            vector_accumulation=True,
        ))
        self.add_pass(Canonicalize())
