"""Reuse one retired operand allocation for a native GEMM's output staging."""

from ..base.ir import find_ops_of_type
from ..base.pipeline import Pass
from ..gpu_dialect import (
    AllocOp,
    AsyncOp,
    AwaitOp,
    ForOp,
    GmemTensorType,
    KernelOp,
    MoveOp,
    SmemTensorType,
    WgrmemTensorType,
    TileOp,
    MatmulOp,
    execution_issuer_warp,
    execution_warps,
    ReshapeOp,
    ReinterpretOp,
)


def allocation_root(value):
    while isinstance(value.definition, (TileOp, ReshapeOp, ReinterpretOp)):
        value = value.definition.operands[0]
    if not isinstance(value.definition, AllocOp):
        raise ValueError("output reuse requires allocation views")
    return value


class ReuseRetiredOperandForOutput(Pass):
    """Prove a load/compute phase ends before a shared output buffer is written.

    Only K loops may precede output. Every operand-load and MMA event must
    have a subsequent, unconditional, whole-CTA drain outside those loops.
    No operand load, compute or shared operand read may follow output staging.
    The existing typed arena pass then overlays the two proven phases.
    """

    def __init__(self, *, reuse_all_operands=False):
        self.reuse_all_operands = reuse_all_operands

    def single_phase_pair_load(self, kernel):
        """Leader load wait precedes MMA; its multicast completion publishes it.

        Restrict this transitive proof to one fully resident paired load/compute
        phase with the native common issuer and an unconditional MMA drain.
        """
        if kernel.cluster_dim != (2, 1, 1):
            raise ValueError("paired arena proof requires one CTA pair")
        ops = kernel.body.ops
        for op in ops:
            if isinstance(op, ForOp) and op.attrs["ub"] <= op.attrs["lb"]:
                continue
            if op.blocks and not isinstance(op, AsyncOp):
                raise ValueError("paired arena proof requires one unrolled phase")
        loads = [
            (i, op)
            for i, op in enumerate(ops)
            if isinstance(op, AsyncOp) and op.lock.type.kind == "mbar"
        ]
        computes = [
            (i, op)
            for i, op in enumerate(ops)
            if isinstance(op, AsyncOp) and op.lock.type.kind == "tcgen05"
        ]
        if len(loads) != 1 or len(computes) != 1:
            raise ValueError("paired arena proof requires one load and one MMA group")
        li, load = loads[0]
        mi, mma = computes[0]
        if (
            li >= mi
            or load.lock.type.attrs != {"kind": "mbar", "cta_group": 2}
            or mma.lock.type.attrs != {"kind": "tcgen05", "cta_group": 2}
            or not mma.attrs.get("release_prior")
        ):
            raise ValueError("paired arena proof requires native paired completion")
        if len(load.completion_locks) != 1 or len(mma.completion_locks) != 1:
            raise ValueError("paired arena proof rejects extra completion phases")
        actors = [load, mma, *load.body.ops, *mma.body.ops]
        if any(
            execution_issuer_warp(op) != 0 or execution_warps(op) is not None
            for op in actors
        ):
            raise ValueError("paired arena proof requires the common issuer")
        if not any(
            isinstance(op, AwaitOp)
            and op.lock is load.lock
            and op.pending_groups == 0
            and (op.waiter_warps is None or 0 in op.waiter_warps)
            for op in ops[li + 1 : mi]
        ):
            raise ValueError("paired operand readiness must reach the MMA issuer")

        def root(value):
            while isinstance(value.definition, TileOp):
                value = value.definition.arg
            if not isinstance(value.definition, AllocOp):
                raise ValueError(
                    "paired arena proof requires direct operand allocation views"
                )
            return value

        if not all(
            isinstance(op, MoveOp)
            and isinstance(op.dst.type, SmemTensorType)
            and isinstance(op.src.type, GmemTensorType)
            for op in load.body.ops
        ):
            raise ValueError("paired arena proof requires native operand loads")
        if not all(isinstance(op, MatmulOp) for op in mma.body.ops):
            raise ValueError("paired arena proof requires native MMA work")
        loaded = {root(op.dst) for op in load.body.ops}
        read = {root(value) for op in mma.body.ops for value in (op.lhs, op.rhs)}
        if len(loaded) != 2 or read != loaded:
            raise ValueError("MMA must consume the paired loaded operands")
        return load.lock

    def run(self, module):
        [kernel] = module.body.ops
        if not isinstance(kernel, KernelOp):
            raise ValueError("operand/output reuse requires one native kernel")
        paired_load = (
            self.single_phase_pair_load(kernel)
            if kernel.cluster_dim == (2, 1, 1) and self.reuse_all_operands
            else None
        )
        block = kernel.body
        allocations = find_ops_of_type(kernel, AllocOp)
        if any(op.attrs.get("reuse_group") is not None for op in allocations):
            raise ValueError(
                "operand/output reuse cannot combine existing reuse groups"
            )
        for op in block.ops:
            if not op.blocks or isinstance(op, AsyncOp):
                continue
            if not isinstance(op, ForOp) or op.attrs.get("label") != "k":
                raise ValueError("operand/output reuse supports only K loops")
            if any(
                child.blocks and not isinstance(child, AsyncOp) for child in op.body.ops
            ):
                raise ValueError("operand/output reuse rejects nested control flow")

        copies = [
            op
            for op in block.ops
            if isinstance(op, MoveOp)
            and isinstance(op.dst.type, SmemTensorType)
            and isinstance(op.src.type, WgrmemTensorType)
        ]
        if not copies:
            raise ValueError("operand/output reuse requires shared output staging")
        output_values = {allocation_root(op.dst) for op in copies}
        if paired_load is None and (
            len(output_values) != 1
            or any(not isinstance(op.dst.definition, AllocOp) for op in copies)
        ):
            raise ValueError("operand/output reuse requires one output allocation")
        output_value = next(iter(output_values))
        operands = [
            op
            for op in block.ops
            if isinstance(op, AllocOp)
            and isinstance(op.result().type, SmemTensorType)
            and (
                op.attrs.get("label", "").startswith(("mm_lhs_smem", "mm_rhs_smem"))
                if self.reuse_all_operands
                else op.attrs.get("label") == "mm_lhs_smem"
            )
        ]
        if not operands:
            raise ValueError("operand/output reuse requires a native LHS allocation")
        if {op.result() for op in operands} & output_values:
            raise ValueError("operand and output allocations must start disjoint")
        operand = operands[0]
        lhs_type = operand.result().type
        output_type = output_value.type
        if (
            any(
                value.type.dtype != lhs_type.dtype or value.type.num_elements is None
                for value in output_values
            )
            or lhs_type.dtype != output_type.dtype
            or lhs_type.num_elements is None
            or output_type.num_elements is None
            or (
                any(op.result().type.dtype != output_type.dtype for op in operands)
                if self.reuse_all_operands
                else lhs_type.num_elements < output_type.num_elements
            )
        ):
            raise ValueError("output staging must fit within the operand allocation")

        first_output = block.ops.index(copies[0])
        last_signals = {}
        saw_mma = False
        for index, op in enumerate(block.ops[:first_output]):
            for move in find_ops_of_type(op, MoveOp):
                if (
                    isinstance(move.src.type, SmemTensorType)
                    and allocation_root(move.src) in output_values
                ):
                    raise ValueError(
                        "output staging cannot be read before its first write"
                    )
                if isinstance(move.dst.type, SmemTensorType) and isinstance(
                    move.src.type, GmemTensorType
                ):
                    owner = move.parent_block.parent_op
                    if not isinstance(owner, AsyncOp) or owner.lock.type.kind != "mbar":
                        raise ValueError(
                            "operand loads require an explicit completion event"
                        )
            if isinstance(op, ForOp):
                lower, upper, step = (
                    op.attrs.get(name) for name in ("lb", "ub", "step")
                )
                if (
                    all(isinstance(value, int) for value in (lower, upper, step))
                    and step > 0
                    and upper <= lower
                ):
                    continue
            for group in find_ops_of_type(op, AsyncOp):
                if group.lock.type.kind not in ("tcgen05", "mbar"):
                    continue
                saw_mma |= group.lock.type.kind == "tcgen05"
                for lock in group.completion_locks:
                    last_signals[lock] = index
        if not saw_mma:
            raise ValueError("operand/output reuse requires native MMA completion")
        for lock, index in last_signals.items():
            if lock is paired_load:
                continue
            if not any(
                isinstance(op, AwaitOp)
                and op.lock is lock
                and op.waiter_warps is None
                and op.pending_groups == 0
                for op in block.ops[index + 1 : first_output]
            ):
                raise ValueError(
                    "operand loads and MMA must be drained before output staging"
                )

        for op in block.ops[first_output:]:
            if isinstance(op, ForOp):
                raise ValueError(
                    "operand/output reuse rejects loops after output staging"
                )
            for group in find_ops_of_type(op, AsyncOp):
                if group.lock.type.kind in ("tcgen05", "mbar"):
                    raise ValueError(
                        "operand/output reuse rejects operand refills or MMA after output"
                    )
            for move in find_ops_of_type(op, MoveOp):
                if (
                    isinstance(move.dst.type, SmemTensorType)
                    and allocation_root(move.dst) not in output_values
                ):
                    raise ValueError(
                        "operand/output reuse rejects shared operand writes after output"
                    )
                if (
                    isinstance(move.src.type, SmemTensorType)
                    and allocation_root(move.src) not in output_values
                ):
                    raise ValueError(
                        "operand/output reuse rejects shared operand reads after output"
                    )

        for operand in operands if self.reuse_all_operands else operands[:1]:
            operand.attrs["reuse_group"] = "retired_operand"
        for value in output_values:
            value.definition.attrs["reuse_group"] = "retired_output"
        return module
