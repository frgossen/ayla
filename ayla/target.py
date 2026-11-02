"""Launch constraints shared by CUDA kernel builders and IR verification."""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class LaunchLimits:
    grid_dimensions: tuple[int, int, int]
    block_dimensions: tuple[int, int, int]
    threads_per_block: int

    def validate(
        self,
        grid: Sequence[int | None],
        block: Sequence[int | None],
    ) -> None:
        """Validate resolved dimensions; unresolved IR dimensions stay unknown."""
        for name, dimensions, limits in (
            ("grid", grid, self.grid_dimensions),
            ("block", block, self.block_dimensions),
        ):
            if len(dimensions) != len(limits):
                raise ValueError(f"{name} must have three dimensions")
            for axis, extent, limit in zip("xyz", dimensions, limits, strict=True):
                if extent is None:
                    continue
                if type(extent) is not int or not 1 <= extent <= limit:
                    raise ValueError(f"{name}.{axis} must be an integer in [1, {limit}]; got {extent!r}")
        threads = math.prod(extent for extent in block if extent is not None)
        if threads > self.threads_per_block:
            raise ValueError(f"block exceeds {self.threads_per_block} threads")


# CUDA limits for the architectures currently targeted by the library.
# X grid indices use 31 bits; Y/Z use 16. Z thread blocks have a smaller axis
# limit than X/Y, independently of the maximum total threads in a block.
CUDA_LAUNCH_LIMITS = LaunchLimits(
    grid_dimensions=((1 << 31) - 1, (1 << 16) - 1, (1 << 16) - 1),
    block_dimensions=(1024, 1024, 64),
    threads_per_block=1024,
)
