"""Reuse kernel output storage for non-overlapping internal buffers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..base.ir import Op, Value, iter_ops
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    AllocOp,
    ElementwiseOp,
    ForOp,
    IfOp,
    KernelOp,
    MemTensorType,
    ModuleOp,
    memory_effects,
)
from ..analysis.buffer_lifetime import BufferLifetimeAnalysis


@dataclass
class _ReuseSet:
    retained_buffer: Value
    buffers: set[Value]
    lifetime: int

    def join(self, buffer: Value, lifetime: int) -> None:
        self.buffers.add(buffer)
        self.lifetime |= lifetime


def _op_local_conflicts(
    kernel: KernelOp,
) -> dict[Value, set[Value]]:
    op_local_conflicts: dict[Value, set[Value]] = {}
    atomic_ops = iter_ops(
        kernel,
        lambda op: not isinstance(op, (KernelOp, ForOp, IfOp)),
    )
    for op in atomic_ops:
        if isinstance(op, ElementwiseOp):
            continue
        effects = memory_effects(op)
        assert effects is not None
        buffers = {buffer for _, buffer, _ in effects}
        # Conservatively assume every pair accessed by an atomic op conflicts.
        for a, b in combinations(buffers, 2):
            op_local_conflicts.setdefault(a, set()).add(b)
            op_local_conflicts.setdefault(b, set()).add(a)
    return op_local_conflicts


def _can_reuse(
    buffer: Value,
    reuse_set: _ReuseSet,
    lifetime: BufferLifetimeAnalysis,
    op_local_conflicts: dict[Value, set[Value]],
) -> bool:
    if buffer.type != reuse_set.retained_buffer.type:
        return False
    if lifetime.lifetime(buffer) & reuse_set.lifetime:
        return False
    return op_local_conflicts.get(buffer, set()).isdisjoint(reuse_set.buffers)


def _writable_buffer_arguments(kernel: KernelOp) -> list[Value]:
    effects = memory_effects(kernel)
    assert effects is not None
    written = {buffer for kind, buffer, _ in effects if kind == "W"}
    return [
        argument
        for argument in kernel.body.arguments
        if argument in written
        and isinstance(argument.type, MemTensorType)
    ]


class ReuseBuffers(LoweringPass):
    """Reuse writable kernel arguments for compatible internal buffers."""

    @lowers(KernelOp)
    def lower_kernel(
        self, kernel: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        lifetime = BufferLifetimeAnalysis()
        lifetime.analyze(kernel)
        # TODO: Investigate modeling lock effects to support safe lock reuse.
        candidate_allocations = [
            op for op in kernel.body.ops
            if isinstance(op, AllocOp)
            and op.init is None
            and isinstance(op.result().type, MemTensorType)
        ]
        op_local_conflicts = _op_local_conflicts(kernel)
        reuse_sets = [
            _ReuseSet(output, {output}, lifetime.lifetime(output))
            for output in _writable_buffer_arguments(kernel)
        ]

        for alloc in candidate_allocations:
            buffer = alloc.result()
            for reuse_set in reuse_sets:
                if not _can_reuse(
                    buffer, reuse_set, lifetime, op_local_conflicts,
                ):
                    continue
                reuse_set.join(buffer, lifetime.lifetime(buffer))
                for user in buffer.users():
                    for index, operand in enumerate(user.operands):
                        if operand is buffer:
                            user.operands[index] = reuse_set.retained_buffer
                kernel.body.ops.remove(alloc)
                break
            else:
                reuse_sets.append(_ReuseSet(
                    buffer, {buffer}, lifetime.lifetime(buffer),
                ))

        return [kernel], []


def reuse_buffers(module: ModuleOp) -> ModuleOp:
    return ReuseBuffers().run(module)
