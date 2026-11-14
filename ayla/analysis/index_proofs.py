"""Conservative integer-index and collective guard proofs."""

from ..base.ir import Block
from ..gpu_dialect import (
    BlockIdOp,
    CollectiveScope,
    ForOp,
    IfOp,
    IndexAddOp,
    IndexConstantOp,
    IndexDivOp,
    IndexLtOp,
    IndexModOp,
    IndexMulOp,
    SelectOp,
)


def exact_affine(value):
    """An exact affine expression, preserving unknown SSA leaves."""
    op = value.definition
    if isinstance(op, IndexConstantOp):
        return {}, op.value
    if isinstance(op, IndexAddOp):
        a, x = exact_affine(op.lhs)
        b, y = exact_affine(op.rhs)
        terms = dict(a)
        for v, coefficient in b.items():
            terms[v] = terms.get(v, 0) + coefficient
        return {v: c for v, c in terms.items() if c}, x + y
    if isinstance(op, IndexMulOp):
        a, x = exact_affine(op.lhs)
        b, y = exact_affine(op.rhs)
        if not a:
            return {v: c * x for v, c in b.items() if c * x}, x * y
        if not b:
            return {v: c * y for v, c in a.items() if c * y}, x * y
    return {value: 1}, 0


def uniform_guard_scope(value, kernel):
    """The widest collective over which a pure index predicate is uniform.

    A two-CTA X pair is aligned at an even block index. Comparing bx plus
    an even cluster-uniform offset with an even bound gives one shared
    predicate. Unknown producers and thread-dependent predicates stay unknown.
    """
    pure = (
        IndexConstantOp,
        IndexAddOp,
        IndexMulOp,
        IndexDivOp,
        IndexModOp,
        IndexLtOp,
        SelectOp,
    )

    def uniform(v, scope):
        op = v.definition
        if isinstance(op, BlockIdOp):
            if scope == CollectiveScope.GRID:
                return False
            return (
                scope == CollectiveScope.BLOCK or kernel.cluster_dim[op.axis.value] == 1
            )
        if isinstance(op, Block):
            parent = op.parent_op
            if not isinstance(parent, ForOp):
                return False
            # A loop under a divergent outer region is not a uniform clock.
            ancestor = parent.parent_block.parent_op
            while ancestor is not None and ancestor is not kernel:
                if isinstance(ancestor, IfOp) and not uniform(
                    ancestor.condition, scope
                ):
                    return False
                ancestor = ancestor.parent_block.parent_op
            return True
        return isinstance(op, pure) and all(uniform(arg, scope) for arg in op.operands)

    if not uniform(value, CollectiveScope.BLOCK):
        return None
    if uniform(value, CollectiveScope.GRID):
        return CollectiveScope.GRID
    if kernel.is_clustered:
        if uniform(value, CollectiveScope.CLUSTER):
            return CollectiveScope.CLUSTER
        pred = value.definition
        if kernel.cluster_dim == (2, 1, 1) and isinstance(pred, IndexLtOp):
            lhs, offset = exact_affine(pred.lhs)
            rhs, bound = exact_affine(pred.rhs)
            varying = [
                v
                for v in lhs
                if isinstance(v.definition, BlockIdOp) and v.definition.axis.value == 0
            ]
            if (
                len(varying) == 1
                and lhs[varying[0]] == 1
                and not rhs
                and (offset - bound) % 2 == 0
                and all(
                    c % 2 == 0 and uniform(v, CollectiveScope.CLUSTER)
                    for v, c in lhs.items()
                    if v is not varying[0]
                )
            ):
                return CollectiveScope.CLUSTER
    return CollectiveScope.BLOCK
