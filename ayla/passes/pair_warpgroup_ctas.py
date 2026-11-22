"""Run two single-warpgroup persistent CTAs per CTA, sharing operand stages.

A 128-thread task program is synchronized as if it owned a CTA. Code
generation executes one copy per warpgroup with virtual thread, warp and block
IDs (``kernel.attrs["warpgroup_ctas"] = 2``). Allocations stay private to their
copy, except operand stages written only by TMA and read only by MMA. Those
stages are passed between copies at task boundaries: a copy releases a stage
after the MMA completion that retires its last read in a task, and acquires
it before its first TMA write in the next task. Each copy's own mbarriers keep
their phases, so steady refills are unchanged.
When K steps do not divide evenly into the ring depth, stage pointers and
handoff events rotate together across tasks instead of restarting at slot 0.

The handoff orders the copies' reductions. A copy starts a task only once its
peer retires the corresponding final stages, so one warpgroup's conversion
and writeback overlap the other warpgroup's MMA without a producer warp.
Apply after InsertSyncs; the ordinary analysis remains valid within a copy.
"""

from __future__ import annotations

import math
import re

from ..base.ir import Block, Op, Value, clone_op, find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    BF16Type,
    AwaitOp,
    BarrierOp,
    ForOp,
    IndexAddOp,
    IndexConstantOp,
    IndexModOp,
    IndexMulOp,
    IndexType,
    KernelOp,
    LockType,
    MatmulOp,
    ModuleOp,
    PTXTemplateOp,
    ReinterpretOp,
    ReshapeOp,
    SmemTensorType,
    TileOp,
    TmaMoveOp,
    effective_operands,
    is_side_effect_free,
    memory_effects,
    root_buffers,
)
from ..pipelines.canonicalize import Canonicalize

# Private mbarriers are laid out as [copy0, copy1] and code generation defines
# ``%vwg`` as the executing copy, so the peer's instance is at ``8 - 16 * %vwg``.
_RELEASE = """\
mov.u32 ${b32:thread}, %tid.x;
setp.eq.u32 ${pred:leader}, ${b32:thread}, 0;
GUARD
mul.lo.u32 ${b32:peer}, %vwg, PEER_DOUBLE;
sub.u32 ${b32:peer}, ${b32:operand.lock.val}, ${b32:peer};
add.u32 ${b32:peer}, ${b32:peer}, PEER_STRIDE;
@${pred:leader} mbarrier.arrive.release.cta.shared::cta.b64 _, [${b32:peer}];
"""
_SECOND_COPY = """\
setp.eq.u32 ${pred:second}, %vwg, 1;
and.pred ${pred:leader}, ${pred:leader}, ${pred:second};"""


def _top(op: Op, block) -> Op | None:
    while op.parent_block is not block:
        parent = op.parent_block.parent_op if op.parent_block is not None else None
        if parent is None:
            return None
        op = parent
    return op


def shared_operand_stages(kernel: KernelOp) -> list[AllocOp]:
    """Shared-memory allocations accessed only by TMA writes and MMA reads."""
    allocs = {
        op.result(): op
        for op in kernel.body.ops
        if isinstance(op, AllocOp) and isinstance(op.result().type, SmemTensorType)
    }
    writers: dict[Value, int] = {}
    readers: set[Value] = set()
    excluded = set()
    for op in find_ops_of_type(kernel, Op):
        # Awaits complete the accesses of the async operations they retire.
        if isinstance(op, (AsyncOp, AwaitOp, ForOp, KernelOp, AllocOp)):
            continue
        effects = memory_effects(op)
        if effects is None:
            excluded.update(
                root
                for v in effective_operands(op)
                for root in root_buffers(v) & allocs.keys()
            )
            continue
        for effect in effects:
            for root in effect.roots & allocs.keys():
                if (
                    isinstance(op, TmaMoveOp)
                    and effect.kind == "W"
                    and effect.value is op.dst
                ):
                    writers[root] = writers.get(root, 0) + 1
                elif isinstance(op, MatmulOp) and effect.kind == "R":
                    readers.add(root)
                else:
                    excluded.add(root)
    return [
        allocs[root]
        for root in writers
        if root in readers
        and root not in excluded
        and not {"reuse_group", "reuse_byte_offset"} & allocs[root].attrs.keys()
    ]


class PairWarpgroupCtas(Pass):
    """Share operand stages between two resident single-warpgroup task programs."""

    def __init__(self, *, rotate_stages: bool = True):
        super().__init__()
        self.rotate_stages = rotate_stages

    def run(self, module: ModuleOp) -> ModuleOp:
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        copies = kernel.attrs.get("warpgroup_ctas", 1)
        if type(copies) is not int or copies not in (1, 2):
            raise ValueError("warpgroup_ctas must be one or two")
        if copies == 1 or kernel.attrs.get("paired_warpgroup_ctas"):
            return module
        if kernel.block_dim != (128, 1, 1):
            raise ValueError("warpgroup CTA pairs require two 128-thread task programs")
        grid = kernel.grid_dim[0]
        if (
            type(grid) is not int
            or grid < 2
            or grid % 2
            or any(d not in (None, 1) for d in kernel.grid_dim[1:])
        ):
            raise ValueError("warpgroup CTA pairs require an even one-dimensional grid")
        if any(d not in (None, 1) for d in kernel.cluster_dim):
            raise ValueError("warpgroup CTA pairs do not support clusters")
        for op in find_ops_of_type(kernel, BarrierOp):
            # Block rendezvous would join both copies; only the initial
            # publication of each copy's mbarriers may remain CTA-wide.
            if op.scope != "warp" and not (
                op.scope == "block" and op.attrs.get("initialization_publication")
            ):
                raise ValueError(
                    f"warpgroup CTA pairs cannot keep a {op.scope} barrier"
                )
        for op in find_ops_of_type(kernel, PTXTemplateOp):
            if re.search(r"\b(?:bar|barrier)\.(?:sync|arrive|red)\b", op.template):
                raise ValueError("warpgroup CTA pairs cannot keep raw CTA barriers")
        for op in kernel.body.ops:
            if isinstance(op, AllocOp) and isinstance(op.result().type, LockType):
                if op.result().type.kind not in ("mbar", "wgmma", "bulk"):
                    raise ValueError("warpgroup CTA pairs require CTA-local locks")
        loops = [
            op
            for op in kernel.body.ops
            if isinstance(op, ForOp) and op.attrs.get("tile_output_loop")
        ]
        if len(loops) != 1:
            raise ValueError("warpgroup CTA pairs require one persistent output loop")
        [loop] = loops
        if loop.attrs.get("output_head_prefetch"):
            raise ValueError("paired operand stages cannot prefetch future tasks")
        count = loop.attrs.get("tile_task_count")
        if (
            loop.lb != 0
            or loop.step != 1
            or type(count) is not int
            or count != grid * loop.ub
        ):
            # Both copies must execute the same number of tasks, otherwise the
            # last handoffs would never be released.
            raise ValueError("warpgroup CTA pairs require complete persistent waves")

        stages = shared_operand_stages(kernel)
        if not stages:
            raise ValueError("warpgroup CTA pairs found no shared operand stages")
        shared = {alloc.result() for alloc in stages}
        body = loop.body
        # Stage groups are the allocations filled by one TMA issue.
        groups: list[frozenset] = []
        for issue in find_ops_of_type(kernel, AsyncOp):
            written = frozenset(
                root
                for move in find_ops_of_type(issue, TmaMoveOp)
                for root in root_buffers(move.dst) & shared
            )
            if written and written not in groups:
                groups.append(written)
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
            raise ValueError("operand stages are not filled as consistent groups")
        for op in find_ops_of_type(kernel, Op):
            if isinstance(op, (TmaMoveOp, MatmulOp)) and _top(op, body) is None:
                touched = {
                    root for effect in memory_effects(op) or [] for root in effect.roots
                }
                if touched & shared:
                    raise ValueError(
                        "operand stages are accessed outside the task loop"
                    )

        math = find_ops_of_type(loop, MatmulOp)
        groups.sort(
            key=lambda group: next(
                i
                for i, mm in enumerate(math)
                if (root_buffers(mm.lhs) | root_buffers(mm.rhs)) & group
            )
        )
        steps = kernel.attrs.get("warpgroup_operand_steps")
        if type(steps) is not int or steps < len(groups):
            raise ValueError("paired operand rings need at least one K step per stage")
        rotating = self.rotate_stages and steps % len(groups) != 0
        if rotating:
            _validate_rotation(kernel, groups)
        handoffs = []
        for group in groups:
            positions = {top: i for i, top in enumerate(body.ops)}
            first_write = min(
                positions[_top(move, body)]
                for move in find_ops_of_type(loop, TmaMoveOp)
                if root_buffers(move.dst) & group
            )
            last_write = max(
                positions[_top(move, body)]
                for move in find_ops_of_type(loop, TmaMoveOp)
                if root_buffers(move.dst) & group
            )
            last_read = max(
                positions[_top(mm, body)]
                for mm in find_ops_of_type(loop, MatmulOp)
                if (root_buffers(mm.lhs) | root_buffers(mm.rhs)) & group
            )
            issue = body.ops[last_read]
            if not isinstance(issue, AsyncOp) or issue.lock.type.kind != "wgmma":
                raise ValueError("final operand reads must be top-level MMA groups")
            retire = _retirement(body, last_read)
            if retire is None or last_write > last_read:
                raise ValueError("operand stage reads must retire before the next task")
            handoffs.append((group, body.ops[first_write], body.ops[retire]))

        for alloc in stages:
            alloc.attrs["warpgroup_shared"] = True
        if rotating:
            _rotate_ring(kernel, loop, groups, handoffs, steps)
        else:
            events = []
            for group, acquire_before, release_after in handoffs:
                event = AllocOp.build_with_fresh_id(LockType.build("mbar"))
                events.append(event)
                body.ops.insert(
                    body.ops.index(acquire_before), AwaitOp.build(event.result())
                )
                body.ops.insert(
                    body.ops.index(release_after) + 1, _release(event.result())
                )
            _issue_head_by_first_use(body, {event.result() for event in events})
            kernel.body.ops[0:0] = events
            position = kernel.body.ops.index(loop)
            kernel.body.ops[position:position] = [
                _release(event.result(), _SECOND_COPY) for event in events
            ]
        kernel.attrs["paired_warpgroup_ctas"] = {
            "stages": len(groups),
            "k_steps": steps,
            "rotating": rotating,
        }
        return module


def _release(event: Value, guard: str = "", stride: int = 8) -> PTXTemplateOp:
    return PTXTemplateOp.build(
        [event],
        [],
        _RELEASE.replace("GUARD", guard)
        .replace("PEER_DOUBLE", str(2 * stride))
        .replace("PEER_STRIDE", str(stride)),
        ["lock"],
        [],
        description="release operand stage to peer warpgroup",
    )


def _retirement(body, issued: int) -> int | None:
    """First await completing the WGMMA group issued at ``body.ops[issued]``.

    WGMMA groups retire in order: an await leaving ``pending`` newer groups
    completes the issue once at least that many groups were issued after it.
    """
    newer = 0
    for i in range(issued + 1, len(body.ops)):
        op = body.ops[i]
        if isinstance(op, AsyncOp) and op.lock.type.kind == "wgmma":
            newer += 1
        elif (
            isinstance(op, AwaitOp)
            and op.lock.type.kind == "wgmma"
            and op.attrs.get("pending", 0) <= newer
        ):
            return i
    return None


def _issue_head_by_first_use(body: Block, events: set[Value]) -> None:
    """Acquire and issue the task's initial stages in the order MMA consumes them.

    A peer retires stages in consumption order, so an acquire blocked on a
    later stage would also delay the earliest one. Only the leading sequence
    of index computations, acquires and TMA issues is reordered.
    """
    end = 0
    while end < len(body.ops):
        op = body.ops[end]
        if not (
            is_side_effect_free(op)
            or isinstance(op, AwaitOp)
            and op.lock in events
            or isinstance(op, AsyncOp)
            and op.lock.type.kind == "mbar"
            and all(isinstance(inner, TmaMoveOp) for inner in op.body.ops)
        ):
            break
        end += 1
    head = body.ops[:end]
    pure = [op for op in head if is_side_effect_free(op)]
    pairs = []
    for i, op in enumerate(head):
        if isinstance(op, AsyncOp):
            if i == 0 or not isinstance(head[i - 1], AwaitOp):
                return
            use = next(
                (
                    j
                    for j in range(end, len(body.ops))
                    if isinstance(body.ops[j], AwaitOp) and body.ops[j].lock is op.lock
                ),
                len(body.ops),
            )
            pairs.append((use, [head[i - 1], op]))
    if len(pure) + 2 * len(pairs) != len(head):
        return
    body.ops[:end] = pure + [
        op for _, pair in sorted(pairs, key=lambda p: p[0]) for op in pair
    ]


def _ordered_group(kernel: KernelOp, group: frozenset) -> list[Value]:
    return [
        op.result()
        for op in kernel.body.ops
        if isinstance(op, AllocOp) and op.result() in group
    ]


def _validate_rotation(kernel: KernelOp, groups: list[frozenset]) -> None:
    expected = [value.type for value in _ordered_group(kernel, groups[0])]
    for group in groups:
        types = [value.type for value in _ordered_group(kernel, group)]
        if types != expected or any(not typ.is_canonical_strides for typ in types):
            raise ValueError(
                "rotating operand stages require equal canonical stage groups"
            )


def _lock_view(arena: Value, slot: Value, task: Value) -> PTXTemplateOp:
    # Every physical slot is transferred once per task. Complete waves keep
    # their phases equal, so the local task's parity is the acquire phase.
    return PTXTemplateOp.build(
        [arena, slot, task],
        [LockType.build("mbar")],
        "mad.lo.u32 ${b32:result.dst.val}, ${b32:operand.slot.val}, 8, ${b32:operand.arena.val};\n"
        "and.b32 ${b32:result.dst.parity}, ${b32:operand.task.val}, 1;\n",
        ["arena", "slot", "task"],
        ["dst"],
        description="physical operand-slot handoff",
    )


def _rotate_ring(
    kernel: KernelOp, loop: ForOp, groups: list[frozenset], handoffs, steps: int
) -> None:
    """Materialize physical slots after logical synchronization has been proved.

    Task order is copy0(t), copy1(t), copy0(t+1), ... . Its K0 slot is
    ((2*t + copy) * K_steps) % depth; a task's final slot is therefore followed
    by the peer's first slot. Private TMA locks keep their existing phases.
    Handoff locks and operand pointers both select the same physical slot.
    """
    depth = len(groups)
    roles = _ordered_group(kernel, groups[0])
    offsets, span = [], 0
    for role in roles:
        span = (span + 1023) // 1024 * 1024
        offsets.append(span)
        span += math.prod(role.type.static_shape) * role.type.dtype.byte_size
    span = (span + 1023) // 1024 * 1024
    storage = AllocOp.build_with_fresh_id(
        SmemTensorType.build([depth, span // 2], BF16Type()),
        attrs={"warpgroup_shared": True, "allocation_alignment": 1024},
    )
    arena = AllocOp.build_with_fresh_id(
        LockType.build("mbar"), attrs={"array_count": depth}
    )
    prefix: list[Op] = []

    def value(op):
        prefix.append(op)
        return op.result()

    ordinal = value(
        PTXTemplateOp.build(
            [loop.iv],
            [IndexType()],
            "mad.lo.u32 ${b32:result.task.val}, ${b32:operand.iteration.val}, 2, %vwg;\n",
            ["iteration"],
            ["task"],
            description="paired task ordinal",
        )
    )
    advance = value(IndexConstantOp.build(steps % depth))
    count = value(IndexConstantOp.build(depth))
    base = value(IndexModOp.build(value(IndexMulOp.build(ordinal, advance)), count))
    mapping: dict[Value, Value] = {}
    handles = []
    for logical, (group, acquire_before, release_after) in enumerate(handoffs):
        slot = value(
            IndexModOp.build(
                value(IndexAddOp.build(base, value(IndexConstantOp.build(logical)))),
                count,
            )
        )
        selected = value(
            TileOp.build(
                storage.result(),
                [slot, 0],
                SmemTensorType.build([1, span // 2], BF16Type()),
            )
        )
        flat = value(
            ReshapeOp.build(selected, SmemTensorType.build([span // 2], BF16Type()))
        )
        for role, offset in zip(_ordered_group(kernel, group), offsets, strict=True):
            mapping[role] = value(ReinterpretOp.build(flat, role.type, offset))
        handle = value(_lock_view(arena.result(), slot, loop.iv))
        handles.append(handle)
        loop.body.ops.insert(loop.body.ops.index(acquire_before), AwaitOp.build(handle))
        loop.body.ops.insert(
            loop.body.ops.index(release_after) + 1, _release(handle, stride=depth * 8)
        )
    _issue_head_by_first_use(loop.body, set(handles))
    inside = set(find_ops_of_type(loop, Op))

    def hoisted(v: Value):
        if v in mapping or not isinstance(v.type, SmemTensorType):
            return
        if not root_buffers(v) & mapping.keys() or v.definition in inside:
            return
        definition = v.definition
        if not isinstance(definition, Op) or not is_side_effect_free(definition):
            raise ValueError("rotating stages require pure hoisted operand views")
        for operand in definition.operands:
            hoisted(operand)
        prefix.append(clone_op(definition, mapping))

    original = list(loop.body.ops)
    for top in original:
        for op in find_ops_of_type(top, Op):
            for operand in op.operands:
                hoisted(operand)
    loop.body.ops = prefix + [clone_op(op, mapping) for op in original]
    kernel.body.ops[0:0] = [storage, arena]
    boot = []
    zero = IndexConstantOp.build(0)
    boot.append(zero)
    for physical in range(depth):
        index = IndexConstantOp.build(physical)
        handle = _lock_view(arena.result(), index.result(), zero.result())
        boot.extend([index, handle, _release(handle.result(), _SECOND_COPY, depth * 8)])
    position = kernel.body.ops.index(loop)
    kernel.body.ops[position:position] = boot
    Canonicalize().run(kernel.parent_block.parent_op)
