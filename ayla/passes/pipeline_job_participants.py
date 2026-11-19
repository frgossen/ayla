"""Schedule independent jobs on rotating participants in a common loop.

MMA and output operations use the existing lowering. Slot-private staging
and explicit completion generations bound progress without waiting for stores.
Each CTA hands issue from one job's actor to the next through its own event,
so the ring's loads and MMAs are issued in ordinal order.
"""

from ..base.ir import clone_op, iter_ops, find_ancestor_of_type
from ..pipelines.async_resources import SplitBarrier
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    BarrierOp,
    ElementwiseOp,
    ForOp,
    IfOp,
    KernelOp,
    LockType,
    MatmulOp,
    MoveOp,
    SmemTensorType,
    TileOp,
    TmaMoveOp,
    WarpIdsOp,
    IndexConstantOp,
    IndexLtOp,
    Block,
    IndexAddOp,
    IndexModOp,
    LockViewOp,
)


def schedule_job_participants(result, op):
    kernel = find_ancestor_of_type(op, KernelOp)
    assert kernel is not None
    assert kernel.num_threads == 256
    warp = WarpIdsOp.build()
    coordinate = warp.x
    four = IndexConstantOp.build(4)
    three = IndexConstantOp.build(3)
    active = [
        IndexLtOp.build(coordinate, four.result()),
        IndexLtOp.build(three.result(), coordinate),
    ]
    # The actor of job j-1 releases issue to the actor of job j. The event is
    # this CTA's own: its next phase needs the waiter's own release, so the
    # expected parity cannot alias. Waiting instead for the previous job's
    # MMA completion lets a paired peer, which only loads, fall two phases
    # behind once its loads run more than one job ahead.
    handoff = SplitBarrier.build(1, 1, "job_issue_handoff")
    additions = [warp, four, three, *active, handoff.allocation]
    remaps = {}

    def actor(slot):
        return 4 * slot

    def participants(slot):
        return list(range(4 * slot, 4 * slot + 4))

    def mapped(value):
        if value in remaps:
            return remaps[value]
        definition = value.definition
        assert isinstance(definition, (AllocOp, TileOp)), repr(definition)
        if isinstance(definition, AllocOp):
            clone = AllocOp.build_with_fresh_id(
                value.type, attrs={"label": "rotating_output_slot1"}
            )
        else:
            values = {definition.arg: mapped(definition.arg)}
            clone = clone_op(definition, values)
        additions.append(clone)
        remaps[value] = clone.result()
        return clone.result()

    def remap_output(operation):
        for i, value in enumerate(list(operation.operands)):
            if isinstance(value.type, SmemTensorType):
                operation.operands[i] = mapped(value)

    def walk(operation, compute_slot=None, output_slot=None):
        compute_slot = operation.attrs.get("compute_slot", compute_slot)
        if "compute_slot" in operation.attrs:
            job = operation.body.ops[0].rhs
            zero = IndexConstantOp.build(0)
            minus = IndexConstantOp.build(-1)
            two = IndexConstantOp.build(2)
            prev = IndexAddOp.build(job, minus.result())
            phase = IndexModOp.build(prev.result(), two.result())
            # A scalar view gives this consumer its own expected phase (one per
            # job); no extra hardware barrier or implicit parity mutation.
            view = LockViewOp.build(handoff.allocation.result(), zero.result(), phase.result())
            due = IndexLtOp.build(zero.result(), job)
            wait = AwaitOp.build(
                view.result(),
                {
                    "waiter_warps": [actor(compute_slot)],
                    "label": "previous_job_issued",
                },
            )
            operation.body.ops = [
                zero,
                minus,
                two,
                prev,
                phase,
                view,
                due,
                IfOp.build(due.result(), Block(ops=[wait])),
                *operation.body.ops,
            ]
        output_slot = operation.attrs.get("writeback_slot", output_slot)
        if "writeback_slot" in operation.attrs:
            region = IfOp.build(
                active[output_slot].result(), Block(ops=list(operation.body.ops))
            )
            region.attrs["participant_warps"] = participants(output_slot)
            operation.body.ops = [region]
        if isinstance(operation, AllocOp) and isinstance(
            operation.result().type, LockType
        ):
            label = operation.attrs.get("label", "")
            if label.startswith("last_tmem_read_"):
                slot = int(label[-1])
                lock_type = operation.result().type
                if lock_type.kind == "mbar":
                    # One local arrival from each warp that reads this slot.
                    lock_type.attrs["arrivals"] = len(participants(slot))
                else:
                    lock_type.attrs["release_warps"] = participants(slot)
        if isinstance(operation, AsyncOp):
            if operation.lock.type.kind == "bulk":
                assert output_slot is not None
                operation.attrs["issuer_warp"] = actor(output_slot)
            else:
                if compute_slot is not None:
                    operation.attrs["issuer_warp"] = actor(compute_slot)
        if isinstance(operation, (MatmulOp, TmaMoveOp)):
            if output_slot is not None and isinstance(operation, TmaMoveOp):
                operation.attrs["issuer_warp"] = actor(output_slot)
                if output_slot == 1:
                    remap_output(operation)
            elif compute_slot is not None:
                operation.attrs["issuer_warp"] = actor(compute_slot)
        if isinstance(operation, AwaitOp):
            if output_slot is not None:
                operation.attrs["issuer_warp"] = actor(output_slot)
                if operation.lock.type.kind != "bulk":
                    operation.attrs["waiter_warps"] = participants(output_slot)
            elif compute_slot is not None:
                operation.attrs["waiter_warps"] = [actor(compute_slot)]
        if isinstance(operation, (MoveOp, ElementwiseOp)) and output_slot is not None:
            operation.attrs["participant_warps"] = participants(output_slot)
            if output_slot == 1 and isinstance(operation, MoveOp):
                remap_output(operation)
        if isinstance(operation, BarrierOp):
            label = operation.attrs.get("label", "")
            if label == "last_tmem_read":
                assert output_slot is not None
                operation.attrs["participant_warps"] = participants(output_slot)
            elif label == "tmem_slot_free":
                assert compute_slot is not None
                operation.attrs.update(
                    participant_warps=[actor(compute_slot)],
                    issuer_warp=actor(compute_slot),
                )
            elif output_slot is not None:
                assert operation.attrs["scope"] == "block"
                operation.attrs.update(
                    participant_warps=participants(output_slot), id=10 + output_slot
                )
        for block in operation.blocks:
            for child in block.ops:
                walk(child, compute_slot, output_slot)

    for root in result[0]:
        walk(root)
    limit0 = IndexConstantOp.build(1)
    limit1 = IndexConstantOp.build(5)
    issuing = [
        IndexLtOp.build(coordinate, limit0.result()),
        IndexLtOp.build(coordinate, limit1.result()),
    ]
    additions.extend([limit0, limit1, *issuing])
    for root in result[0]:
        for operation in list(iter_ops(root)):
            if "compute_slot" not in operation.attrs:
                continue
            slot = operation.attrs["compute_slot"]
            for i, child in enumerate(list(operation.body.ops)):
                if (
                    isinstance(child, ForOp)
                    and child.attrs.get("label") == "shared_ring_rolled_k"
                ):
                    release = handoff.barrier("arrive", "job_issue_handoff")
                    release.attrs["participant_warps"] = [actor(slot)]
                    region = IfOp.build(issuing[slot].result(), Block(ops=[child, release]))
                    if slot == 0:
                        region.attrs["participant_warps"] = [0]
                    else:
                        region.attrs["participant_warps"] = [4]
                        region = IfOp.build(active[slot].result(), Block(ops=[region]))
                        region.attrs["participant_warps"] = participants(slot)
                    operation.body.ops[i] = region
    # Cluster barriers do not replace the local rendezvous required before
    # warp zero can deallocate TMEM used by the other output participants.
    result[0].insert(len(result[0]) - 1, BarrierOp.build("block"))
    kernel.attrs["name"] += "_rotating"
    return [*additions, *result[0]], result[1]
