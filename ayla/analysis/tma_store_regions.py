"""Conservative descriptor-box proofs for issuer-local bulk store queues."""

from __future__ import annotations

import functools

from ..base.ir import Op, Value


def disjoint_tma_store_groups(wait: Op, issue: Op, first: Value, second: Value) -> bool:
    """Bounded boxes below one physically nonoverlapping global argument."""
    from ..gpu_dialect import AsyncOp, AwaitOp

    if not isinstance(wait, AwaitOp):
        return False
    peers = [p for p in wait.lock.users() if isinstance(p, AsyncOp)]
    return _disjoint_tma_groups(peers, issue, first, second)


def disjoint_tma_store_iterations(previous: Op, issue: Op, root: Value, loop: Op | None) -> bool:
    """Separate earlier-iteration boxes using bounds or a monotone affine stride."""
    return _disjoint_tma_groups([previous], issue, root, root, cross_iteration=True, loop=loop)


def _disjoint_tma_groups(peers, issue, first, second, *, cross_iteration=False, loop=None):
    from ..base.ir import Block, find_ancestor_of_type
    from ..gpu_dialect import (
        AsyncOp, BlockIdOp, ForOp, IfOp, GmemTensorType, IndexAddOp,
        IndexConstantOp, IndexMulOp, KernelOp, TmaMoveOp,
        execution_issuer_warp,
    )

    if not isinstance(issue, AsyncOp) or any(not isinstance(peer, AsyncOp) for peer in peers):
        return False
    groups = [*peers, issue]
    if not peers or any(not g.body.ops or not all(isinstance(m, TmaMoveOp)
                      for m in g.body.ops) for g in groups):
        return False
    kernel = find_ancestor_of_type(issue, KernelOp)
    if kernel is None or first is not second or not isinstance(first.type, GmemTensorType):
        return False
    from .memory_regions import has_nonoverlapping_gmem_layout

    if not has_nonoverlapping_gmem_layout(first.type):
        return False

    @functools.cache
    def affine(value):
        op = value.definition
        if isinstance(op, IndexConstantOp):
            result = ({}, op.value, op.value, op.value)
        elif isinstance(op, BlockIdOp):
            size = kernel.grid_dim[op.axis.value]
            if not isinstance(size, int) or size <= 0:
                return None
            result = ({value: 1}, 0, 0, size - 1)
        elif isinstance(op, Block) and isinstance(op.parent_op, ForOp):
            loop = op.parent_op
            if value is not loop.iv or loop.step <= 0 or loop.lb >= loop.ub:
                return None
            last = loop.lb + (loop.ub - loop.lb - 1) // loop.step * loop.step
            result = ({value: 1}, 0, loop.lb, last)
        elif isinstance(op, (IndexAddOp, IndexMulOp)):
            a, b = affine(op.lhs), affine(op.rhs)
            if a is None or b is None:
                return None
            x, xc, xl, xh = a
            y, yc, yl, yh = b
            if isinstance(op, IndexAddOp):
                terms = dict(x)
                for v, coefficient in y.items():
                    terms[v] = terms.get(v, 0) + coefficient
                result = (terms, xc + yc, xl + yl, xh + yh)
            elif not x:
                result = ({v: c * xc for v, c in y.items() if c * xc}, xc * yc, xl * yl, xh * yh)
            elif not y:
                result = ({v: c * yc for v, c in x.items() if c * yc}, xc * yc, xl * yl, xh * yh)
            else:
                return None
        else:
            return None
        return result if 0 <= result[2] <= result[3] < 2**31 else None

    def box(move):
        desc = move.dst.type
        from ..gpu_dialect import TmaDescType

        if (not isinstance(desc, TmaDescType)
                or type(desc.attrs.get("arg_idx")) is not int
                or not 0 <= desc.arg_idx < len(kernel.body.arguments)
                or find_ancestor_of_type(move, KernelOp) is not kernel
                or kernel.body.arguments[desc.arg_idx] is not first):
            return None
        axes = [affine(v) for v in move.indices]
        sizes = list(desc.tile_sizes)
        layout = desc.attrs.get("physical_layout")
        if desc.gmem_type != first.type:
            # LowerToTmaMove describes a column-major matrix in transposed
            # descriptor coordinates. Restore the argument's coordinate
            # order before checking bounds or separating logical boxes.
            # Blocked descriptor coordinates need a different mapping.
            if first.type.rank < 2 or layout is not None:
                return None
            shape, strides = list(first.type.sizes), list(first.type.strides)
            shape[-2], shape[-1] = shape[-1], shape[-2]
            strides[-2], strides[-1] = strides[-1], strides[-2]
            if desc.gmem_type != GmemTensorType.build(shape, first.type.dtype, strides=strides):
                return None
            if len(axes) < 2 or len(sizes) < 2:
                return None
            axes[-2], axes[-1] = axes[-1], axes[-2]
            sizes[-2], sizes[-1] = sizes[-1], sizes[-2]
        if (len(axes) != first.type.rank or len(sizes) != first.type.rank
                or any(a is None for a in axes)
                or any(type(size) is not int or size <= 0 for size in sizes)):
            return None
        if any(a[3] + size > extent for a, size, extent in zip(axes, sizes, first.type.sizes)):
            return None
        if layout is not None:
            width = layout.get("block_width")
            col = axes[-1]
            if (type(width) is not int or width <= 0 or first.type.sizes[-1] % width
                    or sizes[-1] % width or col[1] % width
                    or any(c % width for c in col[0].values())
                    or layout.get("order", "row_major") not in ("row_major", "block_major")):
                return None
        return axes, sizes

    for peer in peers:
        for a in peer.body.ops:
            left = box(a)
            if left is None:
                return False
            for b in issue.body.ops:
                right = box(b)
                if right is None:
                    return False
                def separated(x, xs, y, ys):
                    if not cross_iteration:
                        return x[0] == y[0] and (x[1] + xs <= y[1] or y[1] + ys <= x[1])
                    # Independent bounding boxes remain disjoint even when
                    # other loop clocks have advanced or either access escaped.
                    if x[3] + xs <= y[2] or y[3] + ys <= x[2]:
                        return True
                    if (not isinstance(loop, ForOp) or loop.step <= 0
                            or find_ancestor_of_type(peer, ForOp) is not loop
                            or find_ancestor_of_type(issue, ForOp) is not loop
                            or x[0] != y[0]):
                        return False
                    if any(isinstance(value.definition, Block)
                           and isinstance(value.definition.parent_op, ForOp)
                           and value is not loop.iv for value in x[0]):
                        return False
                    coefficient = x[0].get(loop.iv, 0)
                    delta = y[1] - x[1]
                    pitch = abs(coefficient * loop.step)
                    # Independent issuing warps may advance their loop clocks
                    # differently. Disjoint residues work for every IV pair.
                    if pitch and xs <= delta % pitch and delta % pitch + ys <= pitch:
                        return True

                    from .index_proofs import uniform_guard_scope

                    def uniform_issue(group):
                        ancestor = group.parent_block.parent_op
                        while ancestor is not None and ancestor is not kernel:
                            if isinstance(ancestor, IfOp) and uniform_guard_scope(ancestor.condition, kernel) is None:
                                return False
                            ancestor = ancestor.parent_block.parent_op if ancestor.parent_block is not None else None
                        return True

                    if (execution_issuer_warp(peer) != execution_issuer_warp(issue)
                            or not uniform_issue(peer) or not uniform_issue(issue)):
                        return False
                    # The current IV is at least one positive step beyond
                    # the earlier IV; larger distances strengthen this gap.
                    return ((coefficient >= 0 and coefficient * loop.step + delta >= xs)
                            or (coefficient <= 0 and -coefficient * loop.step - delta >= ys))

                if not any(separated(x, xs, y, ys)
                           for x, xs, y, ys in zip(left[0], left[1], right[0], right[1])):
                    return False
    return True
