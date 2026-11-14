"""Retain immutable global panels in private SMEM across a loop.

This opt-in motion runs before synchronization insertion. It splits existing
TMA async groups and uses their ordinary completion protocol in the preheader.
No new memory arena or hardware operation is needed. Global tensor arguments
have the dialect's distinct-root alias semantics; a source must be read-only
throughout the kernel. Speculation across a guard only writes private SMEM,
whose accesses are confined to this loop, and TMA retains its bounds handling.
"""
from ..base.ir import Block, Value, iter_ops
from ..base.pipeline import Pass
from ..base.lowering_pass import lowers
from ..gpu_dialect import (
    AllocOp, AsyncOp, AwaitOp, BarrierOp, ForOp, GmemTensorType, KernelOp,
    LockType, MoveOp, SmemTensorType, TileOp, IndexConstantOp,
    IndexAddOp, IndexMulOp, IndexModOp,
    effective_operands, is_side_effect_free, memory_effects, root_buffer,
    execution_issuer_warp,
)
from ..pipelines.canonicalize import Canonicalize
from .folding import Folding


class SimplifyIndexRemainders(Folding):
    """Drop a provable multiple from a nonoverflowing index remainder."""
    @lowers(IndexModOp)
    def lower_mod(self, op):
        folded = super().lower_mod(op)
        if folded is not None:
            return folded
        modulus = self._const_val(op.rhs)
        add = self.value_map[op.lhs].definition
        bounds = self._range(op.lhs)
        if not modulus or modulus < 0 or not isinstance(add, IndexAddOp) or bounds is None or bounds[1] >= 2**32:
            return None
        for term, residual in ((add.lhs,add.rhs),(add.rhs,add.lhs)):
            mul = term.definition
            if not isinstance(mul, IndexMulOp):
                continue
            constants = (self._const_val(mul.lhs),self._const_val(mul.rhs))
            if any(c is not None and c % modulus == 0 for c in constants):
                replacement = IndexModOp.build(residual, self.value_map[op.rhs])
                return [replacement], [replacement.result()]
        return None


def _interval(value):
    """Conservative static element interval within a single allocation."""
    typ = value.type
    if not isinstance(typ, SmemTensorType):
        return None
    offset = 0
    current = value
    while isinstance(current.definition, TileOp):
        view = current.definition
        for index, size, stride in zip(view.indices, current.type.sizes, current.type.strides):
            if isinstance(index, Value):
                if not isinstance(index.definition, IndexConstantOp):
                    return None
                index = index.definition.attrs['value']
            if not isinstance(index, int) or size is None or stride is None:
                return None
            offset += index * size * stride
        current = view.arg
    if not isinstance(current.definition, AllocOp):
        return None
    if any(s is None or t is None or t <= 0 for s,t in zip(typ.sizes, typ.strides)):
        return None
    end = offset + 1 + sum((s-1)*t for s,t in zip(typ.sizes, typ.strides))
    capacity = current.type.num_elements
    if capacity is None or offset < 0 or end > capacity:
        return None
    return current, offset, end


class HoistInvariantCopies(Pass):
    """Hoist fully immutable SMEM roots, with a conservative region proof."""
    def __init__(self):
        super().__init__()
        self.hoisted_copies = 0

    def run(self, module):
        self.hoisted_copies = 0
        # Producer actors in a warp-specialized/cluster kernel need a separate
        # publication proof. This prototype targets ordinary single-CTA IR.
        if module.target != 'sm_100a':
            return module
        module = SimplifyIndexRemainders().run(module)
        module = Canonicalize().run(module)
        for kernel in module.body.ops:
            if not isinstance(kernel, KernelOp) or any(d not in (None,1) for d in kernel.attrs.get('cluster_dim', [])):
                continue
            self._kernel = kernel
            self.run_block(kernel.body)
        return Canonicalize().run(module)

    def run_block(self, block):
        for op in list(block.ops):
            if isinstance(op, ForOp):
                self._hoist(op)
            for body in op.blocks:
                self.run_block(body)

    def _hoist(self, loop):
        if loop.results:
            return
        inside = set(iter_ops(loop)); inside.discard(loop)
        dependencies = {}

        def cone(value, visiting=None):
            if any(value in b.arguments for o in [loop,*inside] for b in o.blocks):
                return None
            definition = value.definition
            if definition is None:
                # Block arguments belonging to this loop or any inner block
                # cannot be available in its preheader.
                return [] if all(value not in b.arguments for o in [loop,*inside] for b in o.blocks) else None
            if definition not in inside:
                return []
            if definition.blocks or not is_side_effect_free(definition):
                return None
            if isinstance(definition, AllocOp):
                return None  # roots must already have loop-wide lifetimes
            visiting = set() if visiting is None else set(visiting)
            if definition in visiting:
                return None
            visiting.add(definition)
            result = []
            for operand in effective_operands(definition):
                child = cone(operand, visiting)
                if child is None:
                    return None
                for item in child:
                    if item not in result: result.append(item)
            result.append(definition)
            return result

        all_leaves = [o for o in iter_ops(self._kernel) if not o.blocks and not isinstance(o, AwaitOp)]
        effects = {o:([] if isinstance(o, BarrierOp) else memory_effects(o)) for o in all_leaves}
        if any(e is None for e in effects.values()):
            return
        candidates = []
        for group in [o for o in inside if isinstance(o, AsyncOp)]:
            if group.lock.type.kind != 'mbar' or any(not isinstance(m, MoveOp) for m in group.body.ops):
                continue
            waits = [o for o in group.lock.users() if isinstance(o, AwaitOp)]
            if len(waits) != 1 or waits[0].parent_block is not group.parent_block:
                continue
            if group.parent_block.ops.index(waits[0]) <= group.parent_block.ops.index(group):
                continue
            for move in group.body.ops:
                if not move.attrs.get('tma') or move.multicast or move.indices:
                    continue
                if not isinstance(move.src.type, GmemTensorType) or not isinstance(move.dst.type, SmemTensorType):
                    continue
                interval = _interval(move.dst)
                source = root_buffer(move.src)
                if interval is None or source is None:
                    continue
                root = interval[0]
                if root.definition in inside or root.definition.attrs.get('reuse_group') is not None:
                    continue
                dep = []
                for value in (move.dst,move.src):
                    child = cone(value)
                    if child is None: break
                    for item in child:
                        if item not in dep: dep.append(item)
                else:
                    dependencies[move] = dep
                    candidates.append((move,group,waits[0],root,source,interval))

        selected = []
        for root in {c[3] for c in candidates}:
            members = [c for c in candidates if c[3] is root]
            moves = {c[0] for c in members}
            # Every write to this arena must be one of these invariant loads;
            # reads must stay inside the loop. Also reject kernel-wide writes
            # to any source tensor. Unknown effects were rejected above.
            sources = {c[4] for c in members}
            if any(
                (e.kind == 'W' and e.roots & sources)
                or (root in e.roots and (o not in inside or (e.kind == 'W' and o not in moves)))
                for o,es in effects.items() for e in es
            ):
                continue
            spans = sorted((c[5][1],c[5][2]) for c in members)
            if any(right > next_left for (_,right),(next_left,_) in zip(spans,spans[1:])):
                continue
            selected.extend(members)
        if not selected:
            return
        # Preserve lexical load order and topologically ordered SSA views.
        order = {o:i for i,o in enumerate(iter_ops(loop))}
        selected.sort(key=lambda c:order[c[0]])
        prefix = []
        for move,*_ in selected:
            for dep in dependencies[move]:
                if dep not in prefix: prefix.append(dep)
        for dep in prefix:
            dep.parent_block.ops.remove(dep)
        moved = {c[0] for c in selected}
        groups = {c[1] for c in selected}
        for group in groups:
            group.body.ops = [m for m in group.body.ops if m not in moved]
            if not group.body.ops:
                wait = next(c[2] for c in selected if c[1] is group)
                group.parent_block.ops.remove(wait)
                group.parent_block.ops.remove(group)
        issuer_groups = {}
        for item in selected:
            issuer_groups.setdefault(execution_issuer_warp(item[0]), []).append(item)
        for members in issuer_groups.values():
            lock = AllocOp.build_with_fresh_id(LockType.build(kind='mbar'))
            prefix.extend([lock,AsyncOp.build(lock.result(), collective=members[0][1].collective,
                body=Block(ops=[c[0] for c in members]),attrs={'invariant_copy_preload':True}),
                AwaitOp.build(lock.result())])
        parent = loop.parent_block
        position = parent.ops.index(loop)
        parent.ops[position:position] = prefix
        self.hoisted_copies += len(selected)
