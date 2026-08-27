"""Lower parallel loops onto GPU indices.

Every ``ploop`` disappears: its body is inlined into the enclosing block, its
induction variable is replaced by the hardware index the loop maps onto, and
its trip count becomes the extent of the matching launch dimension. The
extents land on the enclosing ``kernel`` as ``grid_dim``, ``block_dim`` and
``num_warp_groups``.

    kernel @k(%c: gmem<32xf32>) {
      ploop block.x (%bx: index = 0 to 32) {
        use(%bx)
      }
    }

becomes::

    kernel @k(%c: gmem<32xf32>) grid_dim=[32, None, None] {
      %bx, %by, %bz = block_ids()
      use(%bx)
    }

A range other than ``0 to n step 1`` turns into index arithmetic on the id --
``iv = lb + step * id`` -- so the loop's iteration space stays what it says it
is whatever the hardware hands out.

Each ploop builds its own id op and its own arithmetic, however many end up
saying the same thing. CSE folds the duplicates and LICM lifts what does not
vary, so there is nothing here worth a cache.

Each ploop claims its extent on the kernel as it is lowered, which is what
makes the levels agree: two ``block.x`` ploops of different lengths cannot
both be the grid's x extent, and a dim the kernel already carries -- from
tiling, or from its own source -- has to agree with the claim rather than
being overwritten. A dim nobody claims keeps what it had.
"""
from __future__ import annotations

from ..gpu_dialect import (
    BlockIdsOp,
    IndexAddOp,
    IndexConstantOp,
    IndexMulOp,
    KernelOp,
    ModuleOp,
    PLoopOp,
    ThreadIdsOp,
    WarpGroupIdOp,
)
from ..base.ir import Op, Value
from ..base.lowering_pass import LoweringPass, lowers

def _build_index(op: PLoopOp) -> tuple[list[Op], Value]:
    """``lb + step * id`` for this loop, and the ops that compute it.

    Written out even for ``0 to n step 1``, where it is ``id * 1 + 0``:
    folding erases the identities, and this pass has no business duplicating
    it.
    """
    ids = {
        "block": BlockIdsOp,
        "thread": ThreadIdsOp,
        "wg": WarpGroupIdOp,
    }[op.level].build()
    step = IndexConstantOp.build(value=op.step)
    scaled = IndexMulOp.build(
        lhs=ids.results[op.axis], rhs=step.result(),
    )
    lb = IndexConstantOp.build(value=op.lb)
    shifted = IndexAddOp.build(lhs=scaled.result(), rhs=lb.result())
    return [ids, step, scaled, lb, shifted], shifted.result()


class LowerPloop(LoweringPass):
    """Replace every PLoopOp with the GPU index it maps onto."""

    def __init__(self) -> None:
        super().__init__()
        # The kernel being lowered, whose launch dims the ploops inside it
        # claim as they go.
        self.current_kernel: KernelOp

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        self.current_kernel = op
        self.lower_block(op.body)
        return [op], []

    @lowers(PLoopOp)
    def lower_ploop(
        self, op: PLoopOp,
    ) -> tuple[list[Op], list[object]]:
        self._claim_launch_dim(op)
        new_ops, index = _build_index(op)

        # Inlining is the walk's own substitution: bind the induction variable
        # to the index before lowering the body, and every use of it comes out
        # rewritten. The body block itself is then just dropped.
        self.value_map[op.iv] = index
        self.lower_block(op.body)
        return new_ops + op.body.ops, []

    def _claim_launch_dim(self, op: PLoopOp) -> None:
        """Record this loop's extent on the kernel it runs in.

        Writing straight onto the kernel is what makes the levels agree: a
        second ploop on a level meets the first one's extent there, and so
        does a dim the kernel already carried.
        """
        kernel = self.current_kernel

        if op.level == "wg":
            # Warp groups are 1D and this wide, so they fix the block shape
            # whole, not just one axis of it -- one claim, checked and written
            # together with the count.
            warp_group_size = 128
            carried = [kernel.num_warp_groups, *kernel.block_dim]
            claimed = [
                op.trip_count, op.trip_count * warp_group_size, 1, 1,
            ]
            if any(c is not None and c != n for c, n in zip(carried, claimed)):
                raise ValueError(
                    f"lower_ploop: wg.x claims num_warp_groups and block_dim "
                    f"{claimed}, but the kernel already has {carried}"
                )
            kernel.num_warp_groups = claimed[0]
            kernel.block_dim = claimed[1:]
            return

        dims = {
            "block": kernel.grid_dim,
            "thread": kernel.block_dim,
        }[op.level]
        if dims[op.axis] is not None and dims[op.axis] != op.trip_count:
            raise ValueError(
                f"lower_ploop: {op.level}.{PLoopOp.AXES[op.axis]} claims "
                f"{op.trip_count}, but the kernel already has {dims[op.axis]}"
            )
        dims[op.axis] = op.trip_count


def lower_ploop(module: ModuleOp) -> ModuleOp:
    """Convenience entry point."""
    return LowerPloop().run(module)
