"""Shared bank, frame and writeback policy for both ring emitters.

Static emission resolves bank identities and job tails at compile time. Rolled
emission resolves them with index arithmetic and event phases. Both follow the
same K-stage order and release TMEM after the last read, before its store.
"""

from dataclasses import asdict, dataclass, field
import math

from ..base.ir import find_ops_of_type
from ..gpu_dialect import AllocOp, AwaitOp, BarrierOp, DeallocOp, ForOp, KernelOp, TmemTensorType, is_side_effect_free



@dataclass(frozen=True)
class LoopSchedule:
    """Code form and specialization counts in logical loop iterations."""
    form: str = "fixed"
    unroll: int = 1
    peel_front: int = 0
    peel_back: int = 0

    def __post_init__(self):
        if self.form not in ("fixed", "rolled"):
            raise ValueError("loop form must be fixed or rolled")
        for name in ("unroll", "peel_front", "peel_back"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "unroll" else 0):
                raise ValueError(f"loop {name} must be a {'positive' if name == 'unroll' else 'nonnegative'} integer")

    def apply(self, module, label, *, width=1, boundary=(0, 0)):
        """Apply generic loop passes; width states jobs per emitted iteration."""
        from ..passes.peel_loop import PeelLoop
        from ..passes.unroll_loop import UnrollLoop
        from .canonicalize import Canonicalize
        if self.peel_front % width or self.peel_back % width:
            raise ValueError("peeling must cover complete emitted loop iterations")
        front = self.peel_front // width
        back = self.peel_back // width
        if front or back:
            module = PeelLoop(label, front, back).run(module)
        if self.unroll > 1:
            module = UnrollLoop(self.unroll, label).run(module)
            if any(boundary):
                module = Canonicalize().run(module)
        if any(boundary):
            if any(loop.trip_count == 1 and loop.attrs.get("label") == label
                   for loop in find_ops_of_type(module, ForOp)):
                module = Canonicalize().run(module)
            module = PeelLoop(label, *boundary).run(module)
        return module


@dataclass(frozen=True)
class PipelineSchedule:
    """Resource/participant policy, independent of grid persistence."""
    operand_scope: str = "reduction"
    accumulator_slots: int | None = None
    output_overlap: bool | None = None
    participants: str = "shared"
    reduction: LoopSchedule = field(default_factory=LoopSchedule)
    output_jobs: LoopSchedule = field(default_factory=lambda: LoopSchedule("rolled"))
    mma_pending: int = 0
    writeback_window: tuple[int, int] | None = None

    def __post_init__(self):
        if self.operand_scope not in ("reduction", "output_jobs"):
            raise ValueError("operand scope must be reduction or output_jobs")
        if self.participants not in ("shared", "round_robin"):
            raise ValueError("participants must be shared or round_robin")
        for name, value in (("accumulator_slots", 2 if self.cross_jobs else 1),
                            ("output_overlap", self.cross_jobs)):
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if type(self.accumulator_slots) is not int or self.accumulator_slots not in (1, 2):
            raise ValueError("accumulator slots must be one or two")
        if type(self.output_overlap) is not bool:
            raise ValueError("output overlap must be boolean")
        if type(self.mma_pending) is not int or self.mma_pending < 0:
            raise ValueError("MMA pending depth must be a nonnegative integer")
        if self.writeback_window is not None:
            object.__setattr__(self, "writeback_window", tuple(self.writeback_window))
        if not self.cross_jobs and (self.accumulator_slots != 1 or self.output_overlap
                or self.participants != "shared" or self.writeback_window is not None):
            raise ValueError("output resource overlap requires a cross-job operand ring")
        if self.cross_jobs and self.accumulator_slots == 1 and (
                not self.rolled or self.output_overlap or self.rotating):
            raise ValueError("one frame requires a serial rolled ring without job regions")
        if self.rotating and not (self.cross_jobs and self.rolled and self.output_overlap):
            raise ValueError("job participant regions require an overlapping common rolled ring")
        if self.writeback_window is not None and (not self.output_overlap or self.rotating):
            raise ValueError("writeback windows require overlapping shared participants")
        if self.reduction.unroll > 1 and not self.rolled:
            raise ValueError("reduction unroll requires a rolled operand pipeline")
        if self.output_jobs != LoopSchedule("rolled") and not (self.cross_jobs and self.rolled):
            raise ValueError("output job loop transforms require the rolled shared ring")
        if self.output_jobs.form != "rolled":
            raise ValueError("output job emission currently requires a rolled loop")
        if self.output_jobs.peel_front % 2 or self.output_jobs.peel_back % 2:
            raise ValueError("output peeling must cover pairs of logical jobs")

    @classmethod
    def parse(cls, value=None, *, default_pending=0):
        if isinstance(value, cls):
            return value
        if value is not None and not isinstance(value, dict):
            raise TypeError("pipeline must be a schedule object or dictionary")
        values = dict(value or {})
        for name, default in (("reduction", LoopSchedule()), ("output_jobs", LoopSchedule("rolled"))):
            if name in values and not isinstance(values[name], LoopSchedule):
                values[name] = LoopSchedule(**{**asdict(default), **values[name]})
        values.setdefault("mma_pending", default_pending)
        return cls(**values)

    def configuration(self, *, default_pending=0):
        default = PipelineSchedule(operand_scope=self.operand_scope, mma_pending=default_pending)
        result = {}
        for name, value in asdict(self).items():
            if name in ("reduction", "output_jobs"):
                base = asdict(getattr(default, name))
                value = {key: item for key, item in value.items() if item != base[key]}
                if value:
                    result[name] = value
            elif value != getattr(default, name) or name == "operand_scope" and self.cross_jobs:
                result[name] = value
        return result

    @property
    def cross_jobs(self):
        return self.operand_scope == "output_jobs"

    @property
    def rolled(self):
        return self.reduction.form == "rolled"

    @property
    def rotating(self):
        return self.participants == "round_robin"

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
