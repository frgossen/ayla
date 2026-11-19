"""Shared bank, frame and writeback policy for both ring emitters.

Static emission resolves bank identities and job tails at compile time. Rolled
emission resolves them with index arithmetic and event phases. Both follow the
same K-stage order and release TMEM after the last read, before its store.
"""

from dataclasses import dataclass
import math

from ..base.ir import find_ops_of_type
from ..gpu_dialect import AllocOp, AwaitOp, BarrierOp, DeallocOp, ForOp, KernelOp, TmemTensorType, is_side_effect_free


def reject_unaccepted_effects(body, accepted, message):
    """Both ring emitters rebuild only the effects they recognize."""
    if any(not is_side_effect_free(op) and not isinstance(op, (AwaitOp, DeallocOp))
           and op not in accepted for op in body):
        raise ValueError(message)


@dataclass(frozen=True)
class RingInput:
    kernel: KernelOp
    loop: ForOp
    allocation: AllocOp
    group: int
    grid: int
    tile_count: int

    @classmethod
    def from_module(cls, module, *, allow_single_job=False):
        kernels = find_ops_of_type(module, KernelOp)
        if module.target != "sm_100a" or len(kernels) != 1:
            raise ValueError("shared ring requires one SM100 kernel")
        kernel = kernels[0]
        allocations = [
            a
            for a in find_ops_of_type(kernel, AllocOp)
            if isinstance(a.result().type, TmemTensorType)
        ]
        loops = find_ops_of_type(kernel, ForOp)
        cluster = tuple(d or 1 for d in kernel.cluster_dim)
        if (
            cluster not in ((1, 1, 1), (2, 1, 1))
            or kernel.block_dim[1:] != (1, 1)
            or kernel.grid_dim[1:] != (1, 1)
            or len(allocations) != 1
            or len(loops) != 1
            or loops[0].parent_block is not kernel.body
            or loops[0].lb != 0
            or loops[0].step != 1
        ):
            raise ValueError(
                "shared ring requires one persistent output loop with a supported CTA group"
            )
        allocation, loop = allocations[0], loops[0]
        group = 2 if kernel.is_clustered else 1
        m_tile, n_tile = allocation.result().type.sizes
        if n_tile > 256:
            raise ValueError("shared ring two-frame arena exceeds 512 columns")
        m, n = kernel.body.arguments[-1].type.sizes
        tiles = math.ceil(m / (m_tile * group)) * group * math.ceil(n / n_tile)
        grid = kernel.grid_dim[0]
        if grid is None or not 0 < grid <= (tiles if allow_single_job else tiles // 2) or grid % group:
            raise ValueError("shared ring requires at least two output jobs per CTA")
        return cls(kernel, loop, allocation, group, grid, tiles)


@dataclass(frozen=True)
class RingSchedule:
    depth: int
    steps: int
    fragments: int
    overlap: bool
    job_regions: bool = False
    output_writeback_window: tuple[int, int] | None = None
    mma_queue_depth: int = 1

    def __post_init__(self):
        if type(self.mma_queue_depth) is not int or not 1 <= self.mma_queue_depth < self.depth:
            raise ValueError("MMA queue depth must fit below the operand bank count")
        window = self.output_writeback_window
        if window is not None and (
            not isinstance(window, (tuple, list)) or len(window) != 2
            or any(type(x) is not int for x in window)
            or not 0 <= window[0] < window[1] <= self.steps
        ):
            raise ValueError('writeback window must bound nonempty K stages within this job')

    @property
    def writeback_bounds(self):
        return self.output_writeback_window or (0, self.steps)

    @property
    def prefetch_lead(self):
        return self.depth - self.mma_queue_depth

    @property
    def period(self):
        return math.lcm(2, self.depth // math.gcd(self.steps, self.depth))

    @property
    def stage_order(self):
        return (
            ("issue", "finish", "refill", "writeback")
            if self.overlap and not self.job_regions
            else ("issue", "refill")
        )

    def fragments_at(self, k):
        start, stop = self.writeback_bounds
        width = stop - start
        before = min(max(k - start, 0), width)
        after = min(max(k + 1 - start, 0), width)
        return range(
            before * self.fragments // width, after * self.fragments // width
        )

    @property
    def fragment_steps(self):
        start, stop = self.writeback_bounds
        return tuple(
            start + math.ceil((i + 1) * (stop - start) / self.fragments) - 1
            for i in range(self.fragments)
        )

    def writeback_fragment(
        self,
        fragment,
        index,
        ready,
        release,
        *,
        finish=False,
        await_ready=True,
        explicit_barriers=False,
        wait_attrs=None,
    ):
        """Release the frame at its last read; carry only register work onward.

        Static emission runs before sync insertion. Rolled expansion runs after
        it and supplies the two CTA barriers protecting shared staging here.
        """
        read, convert, stage, store = fragment
        ops = []
        if not finish:
            if await_ready and index == 0:
                ops.append(AwaitOp.build(ready, {"label": "accumulator_ready"}))
            ops.append(read)
            if index == self.fragments - 1:
                ops.append(release.barrier("arrive", "last_tmem_read"))
                if self.overlap:
                    return ops
        ops.extend([convert, AwaitOp.build(store.lock, wait_attrs)])
        if explicit_barriers:
            ops.append(BarrierOp.build("block"))
        ops.append(stage)
        if explicit_barriers:
            ops.append(BarrierOp.build("block"))
        ops.append(store)
        return ops
