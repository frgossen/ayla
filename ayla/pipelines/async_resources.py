"""Resource helpers shared by static and rolled asynchronous schedules."""

from dataclasses import dataclass

from ..base.ir import Op
from ..gpu_dialect import (
    AllocOp, AsyncOp, BarrierOp, ElementwiseOp, LockType, MoveOp,
)


def allocate_event(kind, label, attrs=None):
    return AllocOp.build_with_fresh_id(
        LockType.build(kind=kind, attrs=attrs), attrs={"label": label}
    )


@dataclass(frozen=True)
class SplitBarrier:
    """One last-read release, separate from engine and output-store completion."""

    allocation: AllocOp
    cta_group: int
    warps: int

    @classmethod
    def build(cls, cta_group, warps, label):
        allocation = allocate_event(
            "cluster_mbar" if cta_group == 2 else "mbar", label,
            {"release_warps": list(range(warps))}
            if cta_group == 2 else {"arrivals": warps},
        )
        return cls(allocation, cta_group, warps)

    def barrier(self, phase, label):
        return BarrierOp.build(
            "cluster_to_leader" if self.cta_group == 2 else "block_leader",
            self.allocation.result(),
            {"phase": phase, "label": label,
             **({"participant_warps": list(range(self.warps)) if phase == "arrive" else [0]}
                if self.cta_group == 2 else {})},
        )


def tma_writeback_fragments(ops: list[Op], accumulator_type):
    """Identify the read/convert/stage/store units of a tiled TMA writeback."""
    reads = [op for op in ops if isinstance(op, MoveOp)
             and isinstance(op.src.type, accumulator_type)]
    fragments = []
    for index, read in enumerate(reads):
        end = ops.index(reads[index + 1]) if index + 1 < len(reads) else len(ops)
        fragment = [op for op in ops[ops.index(read):end]
                    if isinstance(op, (MoveOp, ElementwiseOp, AsyncOp))]
        if (len(fragment) != 4 or not isinstance(fragment[1], ElementwiseOp)
                or not isinstance(fragment[-1], AsyncOp)
                or fragment[-1].lock.type.kind != "bulk"):
            raise ValueError("ring requires read/convert/stage/store fragments")
        fragments.append(fragment)
    if not fragments:
        raise ValueError("ring requires collective accumulator writeback")
    return fragments
