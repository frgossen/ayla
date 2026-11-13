"""Schedule overlapping matrix/refill groups using the library's async IR."""

from .. import gpu_dialect as g
from ..base.ir import find_ops_of_type, iter_ops
from ..base.pipeline import Pass


class PipelineMatrixRefills(Pass):
    """Overlap straight-line matrix/refill pairs after proving storage and SSA safety."""

    def __init__(self, *, max_pending=1):
        super().__init__()
        self.max_pending = max_pending

    @staticmethod
    def _is_input_transfer(op):
        return (
            isinstance(op, (g.MoveOp, g.TmaMoveOp))
            and isinstance(op.dst.type, g.SmemTensorType)
            and (
                isinstance(op, g.TmaMoveOp)
                or (
                    op.attrs.get("tma") is True
                    and isinstance(op.src.type, g.GmemTensorType)
                )
            )
        )

    def run(self, module):
        self.moved = 0
        for _ in range(self.max_pending):
            self._run_round(module)
        return module

    def _run_round(self, module):
        [kernel] = module.body.ops
        if kernel.is_clustered or kernel.resolved_num_warp_groups != 1:
            return
        if any("reuse_group" in a.attrs for a in find_ops_of_type(kernel, g.AllocOp)):
            return
        ops, queue, index = kernel.body.ops, 0, 0
        while index < len(ops):
            op = ops[index]
            if isinstance(op, (g.ForOp, g.IfOp)):
                queue = None
            elif isinstance(op, g.AsyncOp) and op.lock.type.kind == "wgmma":
                queue = None if queue is None else queue + 1
            elif isinstance(op, g.AwaitOp) and op.lock.type.kind == "wgmma":
                candidate = (
                    self._candidate(kernel, index)
                    if queue is not None
                    and op.pending_groups < self.max_pending
                    and op.pending_groups < queue < 8
                    else None
                )
                if candidate is not None:
                    start, end = candidate
                    moved = list(ops[start:end])
                    del ops[start:end]
                    ops[index:index] = moved
                    op.attrs["pending"] = op.pending_groups + 1
                    op.attrs.pop("sass_wait_elided", None)
                    op.attrs["matrix_refill_pipelined"] = True
                    self.moved += 1
                    queue += 1  # The moved matrix group precedes this wait now.
                    index += len(moved)
                queue = (
                    0
                    if op.is_full_wait
                    else (None if queue is None else min(queue, op.pending_groups))
                )
            index += 1

    def _candidate(self, kernel, index):
        """Return a movable input-wait/matrix slice, or decline without mutation."""
        ops = kernel.body.ops
        pure = (g.AllocOp, g.TileOp, g.ReinterpretOp, g.ReshapeOp, g.IndexConstantOp)

        def following(cursor):
            while cursor < len(ops) and isinstance(ops[cursor], pure):
                cursor += 1
            return cursor

        cursor = following(index + 1)
        if cursor >= len(ops):
            return None
        refill = ops[cursor]
        if not (
            isinstance(refill, g.AsyncOp)
            and refill.lock.type.kind == "mbar"
            and refill.body.ops
            and all(map(self._is_input_transfer, refill.body.ops))
        ):
            return None
        start = cursor + 1
        cursor = following(start)
        if cursor + 1 >= len(ops):
            return None
        input_wait, matrix = ops[cursor : cursor + 2]
        if not (
            isinstance(input_wait, g.AwaitOp)
            and input_wait.lock.type.kind == "mbar"
            and input_wait.lock is not refill.lock
            and isinstance(matrix, g.AsyncOp)
            and matrix.lock.type.kind == "wgmma"
        ):
            return None
        mms = matrix.body.ops
        if not mms or not all(
            isinstance(mm, g.MatmulOp)
            and mm.attrs.get("accumulate", True)
            and isinstance(mm.lhs.type, g.SmemTensorType)
            and isinstance(mm.rhs.type, g.SmemTensorType)
            for mm in mms
        ):
            return None
        targets = {g.root_buffer(t.dst) for t in refill.body.ops}
        reads = {g.root_buffer(v) for mm in mms for v in (mm.lhs, mm.rhs)}
        if reads & targets:
            return None
        wait = ops[index]
        old_group = next(
            (
                op
                for op in reversed(ops[:index])
                if isinstance(op, g.AsyncOp) and op.lock is wait.lock
            ),
            None,
        )
        if old_group is None or not old_group.body.ops:
            return None

        def signature(mm):
            return (
                g.root_buffer(mm.dst),
                tuple(mm.dst.type.sizes),
                tuple(mm.lhs.type.sizes),
                tuple(mm.rhs.type.sizes),
                str(mm.lhs.type.dtype),
                str(mm.rhs.type.dtype),
            )

        if not all(isinstance(mm, g.MatmulOp) for mm in old_group.body.ops) or {
            signature(mm) for mm in old_group.body.ops
        } != {signature(mm) for mm in mms}:
            return None
        if not any(
            isinstance(op, g.AsyncOp) and op.lock is input_wait.lock
            for op in ops[:index]
        ):
            return None
        # Every moved operand must already dominate the insertion point or
        # be defined by an earlier operation in the moved slice.
        available = set(kernel.body.arguments)
        available.update(v for op in ops[:index] for v in op.results)
        for op in ops[start : cursor + 2]:
            for nested in iter_ops(op):
                if any(v not in available for v in nested.operands):
                    return None
                available.update(nested.results)
        return start, cursor + 2


def pipeline_matrix_refills(module, *, max_pending=1):
    """Reorder scheduled async IR before descriptor or instruction lowering."""
    return PipelineMatrixRefills(max_pending=max_pending).run(module)
