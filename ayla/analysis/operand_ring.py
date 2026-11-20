"""Structural proofs for bounded explicit-epoch operand rings."""
from dataclasses import dataclass
from ..base.ir import find_ancestor_of_type, find_ops_of_type
from ..gpu_dialect import (AsyncOp, AwaitOp, ForOp, IndexAddOp, IndexConstantOp, IndexDivOp,
    IndexModOp, IndexMulOp, LockViewOp, MatmulOp, TmaMoveOp, execution_issuer_warp)


def _constant(value):
    op = value.definition
    return op.value if isinstance(op, IndexConstantOp) else None


def _affine(value, variable):
    if value is variable:
        return 1, 0
    constant = _constant(value)
    if constant is not None:
        return 0, constant
    op = value.definition
    if not isinstance(op, (IndexAddOp, IndexMulOp)):
        return None
    left, right = _affine(op.lhs, variable), _affine(op.rhs, variable)
    if left is None or right is None:
        return None
    if isinstance(op, IndexAddOp):
        return left[0] + right[0], left[1] + right[1]
    if left[0] == 0:
        return right[0] * left[1], right[1] * left[1]
    if right[0] == 0:
        return left[0] * right[1], left[1] * right[1]
    return None


@dataclass(frozen=True)
class OperandRing:
    depth: int
    issuer: int
    waits: tuple


def operand_ring(loop):
    """Prove one issuer rotates two matching event arenas with affine epochs.

    Job-dependent epochs fail this proof. They require their own recurrence
    protocol rather than a remembered flag on the loop.
    """
    if not isinstance(loop, ForOp) or loop.step not in (1, 2) or loop.trip_count < 1:
        return None
    issues = [op for op in find_ops_of_type(loop, AsyncOp)
              if find_ancestor_of_type(op, ForOp) is loop]
    mmas = [op for op in issues if op.body.ops and all(isinstance(mm, MatmulOp) for mm in op.body.ops)]
    loads = [op for op in issues if op.body.ops and all(isinstance(move, TmaMoveOp) for move in op.body.ops)]
    if not mmas or not loads or len(issues) != len(mmas) + len(loads):
        return None
    issuers = {execution_issuer_warp(op) for op in issues}
    if len(issuers) != 1:
        return None
    waits = [op for op in find_ops_of_type(loop, AwaitOp)
             if find_ancestor_of_type(op, ForOp) is loop]
    arenas = {}
    depths = set()
    for wait in waits:
        view = wait.lock.definition
        if not isinstance(view, LockViewOp) or wait.waiter_warps != (next(iter(issuers)),):
            return None
        arena, bank, phase = view.operands
        depth = arena.type.attrs.get('count')
        if not isinstance(depth, int) or depth < 2:
            return None
        bank_op, phase_op = bank.definition, phase.definition
        if not (isinstance(bank_op, IndexModOp) and _constant(bank_op.rhs) == depth
                and (_affine(bank_op.lhs, loop.iv) or (0,))[0] == 1
                and isinstance(phase_op, IndexModOp) and _constant(phase_op.rhs) == 2):
            return None
        epoch = phase_op.lhs.definition
        if not (isinstance(epoch, IndexDivOp) and _constant(epoch.rhs) == depth
                and _affine(epoch.lhs, loop.iv) == _affine(bank_op.lhs, loop.iv)):
            return None
        depths.add(depth)
        arenas.setdefault(arena.type.kind, set()).add(arena)
    if (len(depths) != 1 or set(arenas) != {'mbar', 'tcgen05'}
            or any(len(values) != 1 for values in arenas.values())):
        return None
    return OperandRing(depths.pop(), issuers.pop(), tuple(waits))
