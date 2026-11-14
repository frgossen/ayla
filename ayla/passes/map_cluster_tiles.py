"""Map logical GEMM CTA coordinates onto a physical cluster launch."""

from ..base.ir import find_ops_of_type, iter_ops
from ..base.pipeline import Pass
from ..cluster_layout import index_expr, logical_tile_coordinates
from ..gpu_dialect import (
    BlockIdOp,
    ClusterBlockIdsOp,
    CollectiveAxis,
    IndexAddOp,
    IndexMulOp,
    IndexDivOp,
    IndexModOp,
    IndexConstantOp,
)


class MapClusterTiles(Pass):
    def __init__(self, layout, axes, persistent_grid=False):
        super().__init__(layout=layout, axes=axes, persistent_grid=persistent_grid)

    def run(self, module):
        kernel = module.body.ops[0]
        layout = self.layout
        logical = tuple(d or 1 for d in kernel.cluster_dim)
        original = list(iter_ops(kernel))
        prefix = []

        def expr(cls, a, b):
            return index_expr(prefix, cls, a, b)

        ids = ClusterBlockIdsOp.build()
        prefix.append(ids)
        x, y = (IndexConstantOp.build(d) for d in layout.physical[:2])
        prefix += [x, y]
        rank = expr(IndexAddOp, ids.x, expr(IndexMulOp, x.result(),
                    expr(IndexAddOp, ids.y, expr(IndexMulOp, y.result(), ids.z))))
        replacements = {}
        # The RHS half is selected by partner rank, never by the entire
        # physical X/Y extent. Existing two-X-CTA code remains untouched.
        if layout.mma_group == 2 and (
            logical != (2, 1, 1) or layout.physical != logical
        ):
            half = expr(IndexModOp, rank, 2)
            for old in find_ops_of_type(kernel, ClusterBlockIdsOp):
                replacements[old.results[self.axes[0]]] = half
        if self.persistent_grid:
            count = kernel.grid_dim[0]
            bx = BlockIdOp.build(CollectiveAxis.X)
            prefix.append(bx)
            group = expr(IndexDivOp, bx.result(), layout.physical[0])
            virtual = expr(
                IndexAddOp, expr(IndexMulOp, group, layout.size), rank
            )
            for old in find_ops_of_type(kernel, BlockIdOp):
                if old.axis == CollectiveAxis.X:
                    replacements[old.result()] = virtual
            kernel.attrs["grid_dim"] = [
                count // layout.size * layout.physical[0],
                layout.physical[1],
                layout.physical[2],
            ]
        elif layout.physical != logical:
            gm = kernel.grid_dim[self.axes[0]]
            gn = kernel.grid_dim[self.axes[1]]
            groups_m = gm // layout.m
            groups_n = gn // layout.n
            bx = BlockIdOp.build(CollectiveAxis.X)
            prefix.append(bx)
            group = expr(IndexDivOp, bx.result(), layout.physical[0])
            lm, ln = logical_tile_coordinates(
                rank, group, layout, groups_m, groups_n, prefix
            )
            for old in find_ops_of_type(kernel, BlockIdOp):
                if old.axis == self.axes[0]:
                    replacements[old.result()] = lm
                elif old.axis == self.axes[1]:
                    replacements[old.result()] = ln
            kernel.attrs["grid_dim"] = [
                groups_m * groups_n * layout.physical[0],
                layout.physical[1],
                layout.physical[2],
            ]
        for old in original:
            old.operands = [replacements.get(v, v) for v in old.operands]
        for old in original:
            if old.results and all(v in replacements for v in old.results):
                old.parent_block.ops.remove(old)
        kernel.body.ops[0:0] = prefix
        kernel.attrs["cluster_dim"] = list(layout.physical)
        kernel.attrs["logical_cluster_shape"] = [layout.m, layout.n]
        return module
