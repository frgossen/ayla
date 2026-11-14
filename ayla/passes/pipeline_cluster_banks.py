"""Hardware bank release for clustered TCGEN work, with independent result readiness."""

from ..base.ir import find_ops_of_type, iter_ops, find_ancestor_of_type
from ..base.pipeline import Pass
from ..cluster_layout import cluster_size, arrival_counts, multicast_sources, validate_masks
from ..pipelines.async_resources import allocate_event
from ..gpu_dialect import (
    AsyncOp,
    AwaitOp,
    BarrierOp,
    DeallocOp,
    MatmulOp,
    MoveOp,
    TmemTensorType,
    OperandPipelineOp,
    root_buffers,
    execution_issuer_warp,
    memory_effects,
)


def _release_masks(issue, loads, size):
    """CTAs each MMA group's bank release must notify: the next writers of what it read.

    A reader's own group always observes its completion before reusing the
    bank. Beyond that, only the CTA that issues each copy into a read bank
    needs the release; another reader of a shared tile learns that the old
    readers retired through the next data's ready phase, which its writer
    issues only after observing every reader. Writers follow the TMA
    lowering: equal recipient groups rotate their issue through the members.
    """
    from .lower_to_tma_move import _multicast_slices

    group = issue.lock.type.attrs.get("cta_group", 1)
    masks = [
        sum(1 << q for q in range(r // group * group, r // group * group + group))
        for r in range(size)
    ]
    roots = set().union(
        *(root_buffers(mm.lhs) | root_buffers(mm.rhs) for mm in issue.body.ops)
    )
    for load in loads:
        ordinals = {}
        for move in load.body.ops:
            if move.multicast:
                sharing = move.attrs.get("multicast_cta_masks", [(1 << size) - 1] * size)
                validate_masks(sharing, size, groups=True)
                key = tuple(sharing)
                first = ordinals.get(key, 0)
                transfers = _multicast_slices(move, move.dst)
                ordinals[key] = first + transfers
                writers = [0] * size
                for ordinal in range(first, first + transfers):
                    for q, source in enumerate(multicast_sources(sharing, ordinal)):
                        writers[q] |= 1 << source
            else:
                writers = [1 << r for r in range(size)]
            if not root_buffers(move.dst) & roots:
                continue
            for r in range(size):
                for q in range(r // group * group, r // group * group + group):
                    masks[r] |= writers[q]
    return masks


def verify_cluster_bank_pipeline(kernel):
    """Check the reader/issuer contract used by hardware bank release."""
    size = cluster_size(kernel)
    issues = find_ops_of_type(kernel, AsyncOp)
    mmas = [op for op in issues if op.lock.type.kind == "tcgen05"]
    loads = [op for op in issues if op.lock.type.kind == "mbar"]
    if not mmas or any(execution_issuer_warp(op) != 0 for op in [*mmas, *loads]):
        raise ValueError(
            "cluster bank release requires one common elected issue stream"
        )
    if any(
        not op.body.ops or any(not isinstance(mm, MatmulOp) for mm in op.body.ops)
        for op in mmas
    ):
        raise ValueError("cluster bank events must cover MMA operand readers")
    banks = set().union(
        *(
            root_buffers(mm.lhs) | root_buffers(mm.rhs)
            for op in mmas
            for mm in op.body.ops
        )
    )
    for leaf in iter_ops(kernel):
        if leaf.blocks or isinstance(leaf, (AwaitOp, BarrierOp)):
            continue
        effects = memory_effects(leaf)
        if effects is None:
            raise ValueError("cluster bank release requires modeled memory effects")
        if any(e.kind == "R" and e.roots & banks for e in effects):
            if (
                not isinstance(leaf, MatmulOp)
                or find_ancestor_of_type(leaf, AsyncOp) not in mmas
            ):
                raise ValueError(
                    "cluster bank release requires exclusively asynchronous MMA readers"
                )
    for op in mmas:
        masks = _release_masks(op, loads, size)
        attrs = op.lock.type.attrs
        if (
            not attrs.get("cluster_bank_release")
            or attrs.get("completion_cta_masks") != masks
        ):
            raise ValueError(
                "cluster bank completion must cover exactly its operand consumers"
            )
        if attrs.get("arrival_counts") != list(
            arrival_counts(masks, attrs.get("cta_group", 1))
        ):
            raise ValueError("cluster bank counts must match actual MMA contributors")
    for op in [*mmas, *loads]:
        waits = [w for w in op.lock.users() if isinstance(w, AwaitOp)]
        if (
            not waits
            or any(w.waiter_warps != (0,) for w in waits)
            or op.attrs.get("release_prior")
        ):
            raise ValueError(
                "cluster bank phases require issuer observations without software arrivals"
            )


class PipelineClusterBanks(Pass):
    def run(self, module):
        kernel = module.body.ops[0]
        size = cluster_size(kernel)
        issues = find_ops_of_type(kernel, AsyncOp)
        mmas = [op for op in issues if op.lock.type.kind == "tcgen05"]
        loads = [op for op in issues if op.lock.type.kind == "mbar"]
        if (
            not mmas
            or tuple(d or 1 for d in kernel.cluster_dim)
            == (mmas[0].lock.type.attrs.get("cta_group", 1), 1, 1)
            or any(execution_issuer_warp(op) != 0 for op in [*mmas, *loads])
        ):
            # A cluster that is one MMA group keeps that group's own release;
            # hardware cluster release needs one common elected issue stream.
            return module
        if all(_release_masks(op, loads, size) == [1 << r for r in range(size)] for op in mmas):
            # Every MMA group is one CTA that writes everything it reads, so the
            # ordinary CTA-local completion protocol needs no cluster release.
            return module
        bank_roots = set().union(
            *(
                root_buffers(mm.lhs) | root_buffers(mm.rhs)
                for op in mmas
                for mm in op.body.ops
            )
        )
        for leaf in iter_ops(kernel):
            if isinstance(leaf, MoveOp) and root_buffers(leaf.src) & bank_roots:
                raise ValueError(
                    "hardware cluster release requires exclusively asynchronous MMA operand readers"
                )
        for issue in mmas:
            group = issue.lock.type.attrs.get("cta_group", 1)
            masks = _release_masks(issue, loads, size)
            counts = arrival_counts(masks, group)
            issue.lock.type.attrs.update(
                completion_cta_masks=list(masks),
                arrival_counts=list(counts),
                cluster_bank_release=True,
            )
            issue.attrs.pop("release_prior", None)
            for wait in issue.lock.users():
                if isinstance(wait, AwaitOp):
                    wait.attrs["waiter_warps"] = [0]
        for load in loads:
            load.attrs["cluster_bank_pipeline"] = True
            for wait in load.lock.users():
                if isinstance(wait, AwaitOp):
                    wait.attrs["waiter_warps"] = [0]
        # A distinct once-per-output event lets result readers avoid waiting
        # for other MMA groups' final operand-release notifications.
        by_accumulator = {}
        for issue in mmas:
            dst = next(iter(root_buffers(issue.body.ops[0].dst)))
            by_accumulator.setdefault(dst, []).append(issue)
        for accumulator, issues in by_accumulator.items():
            if any(
                find_ancestor_of_type(issue, OperandPipelineOp) is not None
                for issue in issues
            ):
                # The compact region publishes its completed reduction at
                # its boundary. Preserve that proven boundary handshake.
                continue
            last = issues[-1]
            body = last.parent_block
            attrs = {"completion_role": "result"}
            if last.lock.type.attrs.get("cta_group", 1) == 2:
                attrs["cta_group"] = 2
            ready = allocate_event("tcgen05", "cluster_result_ready", attrs)
            body.ops.insert(0, ready)
            last.operands.append(ready.result())
            index = body.ops.index(last) + 1
            drains = [
                op
                for op in body.ops[index:]
                if isinstance(op, AwaitOp) and op.lock in {i.lock for i in issues}
            ]
            for op in drains:
                body.ops.remove(op)
            body.ops.insert(index, AwaitOp.build(ready.result()))
            retire = next(
                (
                    i
                    for i, op in enumerate(body.ops)
                    if isinstance(op, DeallocOp)
                    and isinstance(op.value.type, TmemTensorType)
                ),
                len(body.ops),
            )
            body.ops[retire:retire] = drains
        kernel.attrs["cluster_bank_pipeline"] = True
        kernel.attrs["ordered_mma_completion"] = True
        verify_cluster_bank_pipeline(kernel)
        return module
