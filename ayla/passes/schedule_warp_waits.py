"""Schedule completion observations at proven consumers in a shared CTA pipeline."""
from ..base.ir import find_ancestor_of_type, find_ops_of_type, iter_ops
from ..base.pipeline import Pass
from ..gpu_dialect import (AllocOp, AsyncOp, AwaitOp, BarrierOp, ForOp, KernelOp, LockType, MatmulOp,
    ModuleOp, MoveOp, TmaMoveOp, SmemTensorType, memory_effects, root_buffers,
    execution_issuer_warp, lock_has_tma_multicast)


class ScheduleWarpWaits(Pass):
    """A kernel's ``paired_phase_relay`` enables both schedules; its
    ``paired_completion_handoffs`` hands bank observations to the refill."""

    def __init__(self, *, operand_ready: bool = True, mma_reuse: bool = False):
        super().__init__(operand_ready=operand_ready, mma_reuse=mma_reuse)

    def run(self, module: ModuleOp) -> ModuleOp:
        if module.attrs.get("target") != "sm_100a":
            raise ValueError("warp operand waits require SM100")
        for kernel in find_ops_of_type(module, KernelOp):
            paired_loads = kernel.cluster_dim == (2, 1, 1)
            if (kernel.is_clustered and not paired_loads) or kernel.block_dim[1:] != (1, 1):
                raise ValueError("warp waits require a one-dimensional block; paired CTAs require a shared issuer")
            relay = kernel.attrs.get("paired_phase_relay", False)
            self._handoffs = kernel.attrs.get("paired_completion_handoffs", False)
            self._leaves = [op for op in iter_ops(kernel) if not op.blocks
                            and not isinstance(op, (AwaitOp, BarrierOp))]
            for load in find_ops_of_type(kernel, AsyncOp):
                if not (self.operand_ready or relay):
                    break
                if load.lock.type.kind != "mbar" or lock_has_tma_multicast(load.lock):
                    continue
                if not load.body.ops or not all(isinstance(op, (MoveOp, TmaMoveOp))
                        and isinstance(op.dst.type, SmemTensorType) for op in load.body.ops):
                    continue
                roots = set().union(*(root_buffers(op.dst) for op in load.body.ops))
                readers = self._accessors(roots, "R")
                if not readers or any(not isinstance(op, MatmulOp) or owner is None
                        or owner.lock.type.kind != "tcgen05" for op, _, owner in readers):
                    continue
                consumers = {owner for _, _, owner in readers}
                waiters = sorted({execution_issuer_warp(op) for op in consumers})
                for wait in load.lock.users():
                    if isinstance(wait, AwaitOp):
                        wait.attrs["waiter_warps"] = waiters
                # The release arrival publishes issuer observations, while the
                # ordinary hardware arrival still proves MMA completion.
                for consumer in consumers:
                    for peer in consumer.lock.users():
                        if isinstance(peer, AsyncOp):
                            if self._handoffs:
                                peer.attrs.pop("release_prior", None)
                            else:
                                peer.attrs["release_prior"] = True
            if self.mma_reuse or relay:
                threads = kernel.num_threads
                if threads is None or threads <= 0 or threads % 32:
                    raise ValueError("consumer MMA waits require a resolved whole-warp CTA")
                self._paired = kernel.cluster_dim == (2, 1, 1)
                self._warps = threads // 32
                self._schedule_mma_reuse(kernel.body)
                if self._paired:
                    self._single_use_accumulator_ready(kernel)
            self._drop_self_releases(kernel)
        return module

    @staticmethod
    def _drop_self_releases(kernel):
        """A release arrival publishes issuer observations to the event's
        awaiters. When only the issuing warp awaits, it already holds them."""
        for op in find_ops_of_type(kernel, AsyncOp):
            if not op.attrs.get("release_prior"):
                continue
            users = op.lock.users()
            issuers = {execution_issuer_warp(u) for u in users if isinstance(u, AsyncOp)}
            if all(u.waiter_warps is not None and set(u.waiter_warps) <= issuers
                   for u in users if isinstance(u, AwaitOp)):
                for peer in users:
                    if isinstance(peer, AsyncOp):
                        peer.attrs.pop("release_prior", None)

    def _single_use_accumulator_ready(self, kernel):
        """Replace a terminal phase handoff by an independent result event.

        A fresh event cannot alias an older bank generation. Repeated job
        regions retain their rendezvous until they have an explicit epoch
        protocol; this proof applies only to a once-executed drain.
        """
        for handoff in list(find_ops_of_type(kernel, BarrierOp)):
            if not handoff.attrs.get('mma_wait_handoff'):
                continue
            if find_ancestor_of_type(handoff, ForOp) is not None:
                continue
            block = handoff.parent_block
            position = block.ops.index(handoff)
            batches = [op for root in block.ops for op in iter_ops(root)
                       if isinstance(op, AsyncOp) and op.lock.type.kind == 'tcgen05']
            # A drain can start before the last batches (their banks do not
            # need refilling). Commit at the actual final issue, rather than
            # requiring all issues to precede the first widened bank wait.
            def unconditional(batch):
                owner = batch.parent_block.parent_op
                while owner is not block.parent_op:
                    if not isinstance(owner, ForOp) or owner.trip_count < 1:
                        return False
                    owner = owner.parent_block.parent_op
                return True

            if (not batches or not all(unconditional(batch) for batch in batches)
                    or any(not batch.body.ops
                    or any(not isinstance(mm, MatmulOp) for mm in batch.body.ops)
                    for batch in batches)):
                continue
            last = batches[-1]
            # Earlier batches may execute in bounded K loops. The final
            # direct drain batch executes once and owns the single-use event.
            if last.parent_block is not block:
                continue
            issuers = {execution_issuer_warp(op) for op in batches}
            if len(issuers) != 1:
                continue
            issuer = next(iter(issuers))
            drains = [op for op in block.ops[position + 1:]
                      if isinstance(op, AwaitOp) and op.lock.type.kind == 'tcgen05'
                      and op.waiter_warps is None]
            if not drains or any(any(peer not in batches for peer in drain.lock.users()
                    if isinstance(peer, AsyncOp)) for drain in drains):
                continue
            at = max(block.ops.index(last), *(block.ops.index(wait) for wait in drains)) + 1
            roots = set().union(*(root_buffers(mm.dst) for batch in batches
                                   for mm in batch.body.ops if isinstance(mm, MatmulOp)))
            # Moving the collective observation may not let a synchronous
            # accumulator accessor pass it. Unknown effects keep the handoff.
            accessors = [leaf for root in block.ops[position + 1:at]
                         if not isinstance(root, (AsyncOp, AwaitOp))
                         for leaf in iter_ops(root) if not leaf.blocks
                         and find_ancestor_of_type(leaf, AsyncOp) not in batches]
            if any(memory_effects(leaf) is None or any(effect.roots & roots
                    for effect in memory_effects(leaf)) for leaf in accessors):
                continue
            ready = AllocOp.build_with_fresh_id(LockType.build('tcgen05', {
                'cta_group': 2, 'completion_role': 'result'}),
                attrs={'label': 'accumulator_ready'})
            kernel.body.ops.insert(0, ready)
            last.operands.append(ready.result())
            # Each bank's final generation still needs an observation before
            # retirement. Only the issuing warp consumes those bank events.
            for drain in drains:
                drain.attrs['waiter_warps'] = [issuer]
            at = max(block.ops.index(last), *(block.ops.index(wait) for wait in drains)) + 1
            block.ops.insert(at, AwaitOp.build(ready.result(), {'label': 'accumulator_ready'}))
            block.ops.remove(handoff)

    def _accessors(self, roots, kinds="RW"):
        """``(leaf, effects, owner)`` for leaves accessing ``roots``; None if unmodeled."""
        found = []
        for leaf in self._leaves:
            effects = memory_effects(leaf)
            if effects is None:
                return None
            if any(e.kind in kinds and e.roots & roots for e in effects):
                found.append((leaf, effects, find_ancestor_of_type(leaf, AsyncOp)))
        return found

    def _reuse_consumers(self, wait, following):
        """Prove the next data consumer is an elected operand-bank refill.

        Same-issuer tensor-core updates are ordered by their engine. Any other
        accumulator access or synchronous operand access keeps the original
        wait. Completion-event reuse also needs its MMA issuer to observe the
        phase, independently of the data consumer's issuer.
        """
        peers = [u for u in wait.lock.users() if isinstance(u, AsyncOp)]
        if not peers or any(not p.body.ops or any(not isinstance(m, MatmulOp)
                for m in p.body.ops) for p in peers):
            return None
        issuers = {execution_issuer_warp(p) for p in peers}
        if len(issuers) != 1:
            return None
        operand_roots = set().union(*(root_buffers(m.lhs) | root_buffers(m.rhs)
            for p in peers for m in p.body.ops))
        accumulator_roots = set().union(*(root_buffers(m.dst)
            for p in peers for m in p.body.ops))
        # A fast nonconsumer may pass a readiness wait before an older event
        # generation is reused. This is safe only when every actual SMEM
        # accessor belongs to the elected TMA/MMA operations that observe it.
        operand_issuers = set()
        reader_issuers = set()
        accessors = self._accessors(operand_roots)
        if accessors is None:
            return None
        for leaf, effects, owner in accessors:
            if owner is None:
                return None
            if isinstance(leaf, MatmulOp) and owner.lock.type.kind == 'tcgen05':
                # Every elected operand reader must keep pace with event
                # generations, including MMA consumers using another lock.
                operand_issuers.add(execution_issuer_warp(owner))
                reader_issuers.add(execution_issuer_warp(owner))
                continue
            if (isinstance(leaf, (MoveOp, TmaMoveOp)) and owner.lock.type.kind == 'mbar'
                    and isinstance(leaf.dst.type, SmemTensorType)
                    and (isinstance(leaf, TmaMoveOp) or leaf.attrs.get('tma'))
                    and not lock_has_tma_multicast(owner.lock)):
                if any(e.kind == 'W' and e.roots & operand_roots for e in effects):
                    operand_issuers.add(execution_issuer_warp(owner))
                continue
            return None
        for op in following:
            if isinstance(op, AwaitOp):
                continue
            effects = memory_effects(op)
            if effects is None or isinstance(op, BarrierOp):
                return None
            if not any(e.roots & (operand_roots | accumulator_roots) for e in effects):
                continue
            if isinstance(op, AsyncOp) and op.lock.type.kind == 'tcgen05':
                if execution_issuer_warp(op) in issuers:
                    continue
                return None
            if (isinstance(op, AsyncOp) and op.lock.type.kind == 'mbar'
                    and not lock_has_tma_multicast(op.lock)
                    and all(isinstance(m, (MoveOp, TmaMoveOp))
                        and isinstance(m.dst.type, SmemTensorType) for m in op.body.ops)
                    and any(e.kind == 'W' and e.roots & operand_roots for e in effects)
                    and not any(e.roots & accumulator_roots for e in effects)):
                if self._handoffs and reader_issuers == issuers:
                    # The paired refill completion acknowledges the refill
                    # warp's old bank observation to the MMA issuer. Keep all
                    # other readers when that single-consumer proof fails.
                    return [execution_issuer_warp(op)]
                return sorted(issuers | operand_issuers)
            return None
        return None

    def _schedule_mma_reuse(self, block, incoming=False):
        """Narrow proven bank waits, retaining a collective pipeline drain.

        Nonwaiters still execute the same loops and advance their local parity.
        Before widening a completion wait, a CTA rendezvous ensures its producer
        has reached that generation; otherwise a fast thread could accept an
        older phase with the same parity. Existing full barriers provide it.
        """
        ops = list(block.ops)
        result = []
        narrowed = incoming
        for index, op in enumerate(ops):
            if (isinstance(op, AwaitOp) and op.lock.type.kind == 'tcgen05'
                    and op.waiter_warps is None):
                consumers = self._reuse_consumers(op, ops[index + 1:])
                if consumers is not None and len(consumers) < self._warps:
                    op.attrs['waiter_warps'] = consumers
                    narrowed = True
                elif narrowed:
                    result.append(BarrierOp.build(attrs={
                        'scope': 'cluster' if self._paired else 'block',
                        'mma_wait_handoff': True,
                        **({'publishes': False} if self._paired else {})}))
                    narrowed = False
            elif isinstance(op, BarrierOp) and op.scope == 'block':
                narrowed = False
            elif not isinstance(op, AsyncOp):
                for child in op.blocks:
                    narrowed |= self._schedule_mma_reuse(child, narrowed)
            result.append(op)
        block.ops = result
        return narrowed
