"""Prove that index guards select complete physical warps."""

from functools import cache

from ..base.ir import Block
from ..gpu_dialect import (
    BlockIdOp,
    ClusterBlockIdsOp,
    ForOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    KernelOp,
    OperandPipelineOp,
    PersistentPipelineOp,
    PredicateOp,
    SelectOp,
    ThreadIdOp,
    WarpGroupIdOp,
    WarpIdsOp,
)
from .index_proofs import uniform_guard_scope


def warp_uniform_guard(condition, kernel):
    """Check one guard without an enclosing selection context."""
    return warp_uniform_guards((condition,), kernel)


def warp_uniform_guards(conditions, kernel):
    """The guard conjunction is equal on each physical warp's launched lanes.

    Uniform scalar leaves stay symbolic, so enumerating thread coordinates
    does not assume a particular grid, parameter, or loop induction value.
    Index arithmetic follows the backend's unsigned 32-bit operations.
    """
    return _analyze_guards(conditions, kernel)


def warp_guard_truth(condition, kernel, warp):
    """Return a guard's constant truth on a complete physical warp, or None.

    Symbolic kernel parameters and induction values are never substituted.
    The guard must also preserve participation in the other launched warps.
    """
    return _analyze_guards((condition,), kernel, warp=warp)


def cta_uniform_guards(conditions, kernel):
    """Prove equal effective participation on every launched CTA thread."""
    return _analyze_guards(conditions, kernel, cta=True)


def participant_uniform_guards(conditions, kernel, warps):
    """Compare effective control across all threads of a counted rendezvous.

    Warp-uniform control alone can omit entire required arriving warps.
    Threads outside the explicitly declared participant set do not arrive.
    """
    if not isinstance(kernel, KernelOp):
        return False
    dims = kernel.block_dim
    if any(type(d) is not int or d <= 0 for d in dims):
        return False
    participants = tuple(warps)
    total = dims[0] * dims[1] * dims[2]
    if (
        not participants
        or len(set(participants)) != len(participants)
        or any(
            type(w) is not int or w < 0 or (w + 1) * 32 > total for w in participants
        )
    ):
        return False
    return _analyze_guards(conditions, kernel, participants=participants)


def _analyze_guards(conditions, kernel, *, warp=None, cta=False, participants=None):
    unknown = False if warp is None else None
    if not isinstance(kernel, KernelOp):
        return unknown
    conditions = tuple(conditions)
    if warp is None and all(
        uniform_guard_scope(condition, kernel) is not None for condition in conditions
    ):
        return True
    dims = kernel.block_dim
    if any(type(d) is not int or d <= 0 for d in dims):
        return unknown
    bx, by, bz = dims
    total = bx * by * bz

    def coords(thread):
        return thread % bx, thread // bx % by, thread // (bx * by)

    @cache
    def uniform(guards):
        for first in range(0, total, 32):
            expressions = [
                conjunction(guards, thread)
                for thread in range(first, min(first + 32, total))
            ]
            if any(e is None or e != expressions[0] for e in expressions):
                return False
        return True

    def conjunction(guards, thread):
        terms = []
        for guard in guards:
            term = expression(guard, thread)
            if term == 0:
                return 0
            if term is None:
                return None
            if not isinstance(term, int):
                terms.append(term)
        return ("and", *terms) if terms else 1

    def affine(term):
        if isinstance(term, int):
            return {}, term
        if term[0] == "uniform":
            return {term[1]: 1}, 0
        if term[0] not in (IndexAddOp, IndexMulOp):
            return None
        a, b = affine(term[1]), affine(term[2])
        if a is None or b is None:
            return None
        x, xc = a
        y, yc = b
        if term[0] is IndexAddOp:
            terms = dict(x)
            for value, coefficient in y.items():
                terms[value] = (terms.get(value, 0) + coefficient) % 2**32
            return {v: c for v, c in terms.items() if c}, (xc + yc) % 2**32
        if not x:
            return {
                v: c * xc % 2**32 for v, c in y.items() if c * xc % 2**32
            }, xc * yc % 2**32
        if not y:
            return {
                v: c * yc % 2**32 for v, c in x.items() if c * yc % 2**32
            }, xc * yc % 2**32
        return None

    @cache
    def expression(value, thread):
        op = value.definition
        if isinstance(op, IndexConstantOp):
            return op.value % 2**32
        if isinstance(op, ThreadIdOp):
            return coords(thread)[op.axis.value]
        if isinstance(op, WarpIdsOp):
            xyz = coords(thread // 32 * 32)
            return (xyz[0] // 32, xyz[1], xyz[2])[value.result_index]
        if isinstance(op, WarpGroupIdOp):
            return (coords(thread)[0] // 128, 0, 0)[value.result_index]
        if isinstance(op, (BlockIdOp, ClusterBlockIdsOp)):
            return ("uniform", value)
        if isinstance(op, Block):
            parent = op.parent_op
            if parent is kernel:
                return ("uniform", value)
            if (
                not isinstance(parent, (ForOp, OperandPipelineOp, PersistentPipelineOp))
                or value is not op.arguments[0]
            ):
                return None
            ancestor = parent.parent_block.parent_op
            guards = []
            while ancestor is not None and ancestor is not kernel:
                if isinstance(ancestor, IfOp):
                    guards.append(ancestor.condition)
                ancestor = (
                    ancestor.parent_block.parent_op if ancestor.parent_block else None
                )
            if not uniform(tuple(reversed(guards))):
                return None
            return ("uniform", value)
        args = (
            [expression(arg, thread) for arg in op.operands]
            if hasattr(op, "operands")
            else []
        )
        if any(arg is None for arg in args):
            return None
        if isinstance(op, (IndexAddOp, IndexMulOp, IndexDivOp, IndexModOp, IndexLtOp)):
            a, b = args
            if isinstance(a, int) and isinstance(b, int):
                if isinstance(op, IndexAddOp):
                    return (a + b) % 2**32
                if isinstance(op, IndexMulOp):
                    return (a * b) % 2**32
                if isinstance(op, IndexLtOp):
                    return int(a < b)
                if b == 0:
                    return None
                return a // b if isinstance(op, IndexDivOp) else a % b
            if isinstance(op, IndexLtOp) and a == b:
                return 0
            if isinstance(op, IndexLtOp):
                x, y = affine(a), affine(b)
                if (
                    x is not None
                    and y is not None
                    and y[1] % 32 == 0
                    and all(c % 32 == 0 for c in (*x[0].values(), *y[0].values()))
                ):
                    # Against an aligned bound, dropping the low five bits
                    # preserves unsigned comparison, including wrapping sums.
                    return (
                        IndexLtOp,
                        frozenset(x[0].items()),
                        x[1] // 32,
                        frozenset(y[0].items()),
                        y[1] // 32,
                    )
            return (type(op), a, b)
        if isinstance(op, PredicateOp):
            kind = op.attrs["kind"]
            if all(isinstance(arg, int) for arg in args):
                return (
                    int(not args[0])
                    if kind == "not"
                    else int(all(args) if kind == "and" else any(args))
                )
            return (kind, *args)
        if isinstance(op, SelectOp):
            index, *cases = args
            if isinstance(index, int):
                return cases[index] if 0 <= index < len(cases) else None
            return ("select", *args)
        return None

    if cta or participants is not None:
        threads = (
            range(total)
            if participants is None
            else [t for w in participants for t in range(w * 32, (w + 1) * 32)]
        )
        terms = [conjunction(conditions, thread) for thread in threads]
        return all(term is not None and term == terms[0] for term in terms)
    if warp is None:
        return uniform(conditions)
    if (
        type(warp) is not int
        or warp < 0
        or (warp + 1) * 32 > total
        or not uniform(conditions)
    ):
        return None
    terms = [
        conjunction(conditions, thread) for thread in range(warp * 32, (warp + 1) * 32)
    ]
    if all(isinstance(term, int) and term == terms[0] for term in terms):
        return bool(terms[0])
    return None
