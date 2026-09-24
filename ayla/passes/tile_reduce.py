"""Tile the parallel and reduction dimensions of reductions."""
from __future__ import annotations

from ..base.ir import (
    Block,
    Op,
    Value,
    clone_op,
    clone_type,
    find_ancestor_of_type,
)
from ..base.lowering_pass import LoweringPass, lowers
from ..gpu_dialect import (
    BlockIdsOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    ForOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    KernelOp,
    ModuleOp,
    ReduceOp,
    ThreadIdsOp,
    TileOp,
    WarpIdsOp,
    WarpGroupIdOp,
)
from ._utils import label_matches


def _update_launch_dims(
    hero: ReduceOp,
    launch_dims: dict[tuple[CollectiveLevel, CollectiveAxis], int],
) -> dict[tuple[CollectiveLevel, CollectiveAxis], int]:
    kernel = find_ancestor_of_type(hero, KernelOp)
    assert kernel is not None

    for (level, axis), launch_dim in launch_dims.items():

        is_block_dim_based = level in (
            CollectiveLevel.THREAD,
            CollectiveLevel.WARP,
            CollectiveLevel.WG,
        )
        if is_block_dim_based:
            base_launch_dims = list(kernel.block_dim)
        else:
            base_launch_dims = list(kernel.grid_dim)

        base_launch_dim = launch_dim
        if level == CollectiveLevel.WARP:
            assert axis == CollectiveAxis.X
            base_launch_dim *= 32
        elif level == CollectiveLevel.WG:
            assert axis == CollectiveAxis.X
            base_launch_dim *= 128

        if base_launch_dims[axis] not in (None, base_launch_dim):
            raise ValueError(
                f"launch dim conflict: {base_launch_dims[axis]} vs "
                f"{base_launch_dim}"
            )
        base_launch_dims[axis] = base_launch_dim

        if is_block_dim_based:
            kernel.block_dim = base_launch_dims
        else:
            kernel.grid_dim = base_launch_dims

    return launch_dims


def _tile_value(
    arg: Value,
    tile_sizes: dict[int, int],
    tile_indices: dict[int, Value],
) -> tuple[list[Op], Value]:
    typ = clone_type(arg.type)
    indices: list[Value | int] = [0] * typ.rank
    for dim, tile_size in tile_sizes.items():
        typ.sizes[dim] = tile_size
        indices[dim] = tile_indices[dim]
    tile_op = TileOp.build(arg, indices, typ)
    return [tile_op], tile_op.result()


class TileReduce(LoweringPass):
    """Tile reduction iteration spaces and selected epilogues.

    ``tile_sizes`` maps each tiled dimension to its tile size, while
    ``parallel_iteration_spaces`` optionally maps it to hardware.
    Nested tiling is expressed by composing multiple pass invocations.
    A dimension without a parallel iteration space is lowered to a sequential
    loop whose label is derived from the dimension.
    """

    def __init__(
        self,
        tile_sizes: dict[int, int],
        parallel_iteration_spaces: dict[
            int, tuple[CollectiveLevel, CollectiveAxis]
        ],
        hero_label_filter: str | None = None,
        epilogue_label_filter: str | None = None,
    ) -> None:
        super().__init__(
            tile_sizes=dict(tile_sizes),
            parallel_iteration_spaces=dict(parallel_iteration_spaces),
            hero_label_filter=hero_label_filter,
            epilogue_label_filter=epilogue_label_filter,
        )

    @lowers(KernelOp)
    def lower_kernel(
        self, op: KernelOp,
    ) -> tuple[list[Op], list[object]]:
        self._rewrite_block(op.body)
        return [op], []

    def _rewrite_block(self, block: Block) -> None:
        for op in block.ops:
            for nested in op.blocks:
                self._rewrite_block(nested)

        ops = list(block.ops)
        new_ops: list[Op] = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if not label_matches(op, self.hero_label_filter):
                new_ops.append(op)
                i += 1
                continue
            epilogue: list[Op] = []
            for candidate in ops[i + 1:]:
                if not label_matches(candidate, self.epilogue_label_filter):
                    break
                epilogue.append(candidate)
            new_ops.extend(
                self._rewrite_hero_with_epilogue(op, epilogue)
            )
            i += 1 + len(epilogue)
        block.ops = new_ops

    def _rewrite_hero_with_epilogue(
        self, hero: Op, epilogue: list[Op],
    ) -> list[Op]:
        assert isinstance(hero, ReduceOp)
        assert all(isinstance(op, ElementwiseOp) for op in epilogue)
        assert all(
            operand is hero.dst
            for op in epilogue
            for operand in op.operands
        )
        rank = hero.src.type.rank
        if rank - 1 in self.tile_sizes:
            raise NotImplementedError(
                "TileReduce does not yet support tiling the reduction dimension"
            )

        iteration_space_ops, ivs, innermost_loop = (
            self._materialize_iteration_space(hero)
        )

        src_ops, src = _tile_value(
            hero.src, self.tile_sizes, ivs,
        )
        dst_ops, dst = _tile_value(
            hero.dst, self.tile_sizes, ivs,
        )
        tiled_hero = clone_op(hero, {hero.dst: dst, hero.src: src})
        tiled_epilogue = [
            clone_op(op, {hero.dst: dst}) for op in epilogue
        ]
        tiled_ops = [*src_ops, *dst_ops, tiled_hero, *tiled_epilogue]
        if innermost_loop is None:
            new_ops = [*iteration_space_ops, *tiled_ops]
        else:
            innermost_loop.body.ops.extend(tiled_ops)
            new_ops = iteration_space_ops
        return new_ops

    def _materialize_iteration_space(
        self,
        hero: ReduceOp,
    ) -> tuple[list[Op], dict[int, Value], ForOp | None]:
        bounds: dict[int, int] = {}
        launch_dims: dict[
            tuple[CollectiveLevel, CollectiveAxis], int
        ] = {}
        for dim, tile_size in self.tile_sizes.items():
            bounds[dim] = hero.src.type.sizes[dim] // tile_size
            space = self.parallel_iteration_spaces.get(dim)
            if space is not None:
                launch_dims[space] = (
                    launch_dims.get(space, 1) * bounds[dim]
                )
        launch_dims = _update_launch_dims(hero, launch_dims)

        new_ops: list[Op] = []
        ivs: dict[int, Value] = {}
        innermost_loop: ForOp | None = None
        divisors: dict[
            tuple[CollectiveLevel, CollectiveAxis], int
        ] = {}
        for dim in reversed(self.tile_sizes):
            space = self.parallel_iteration_spaces.get(dim)
            if space is None:
                loop = ForOp.build(
                    lb=0,
                    ub=bounds[dim],
                    attrs={"label": f"d{dim}"},
                )
                loop.body.ops.extend(new_ops)
                new_ops = [loop]
                if innermost_loop is None:
                    innermost_loop = loop
                ivs[dim] = loop.iv
                continue
            level, axis = space
            id_op_type = {
                CollectiveLevel.BLOCK: BlockIdsOp,
                CollectiveLevel.THREAD: ThreadIdsOp,
                CollectiveLevel.WARP: WarpIdsOp,
                CollectiveLevel.WG: WarpGroupIdOp,
            }[level]
            ids = id_op_type.build()
            divisor = divisors.get(space, 1)
            divisors[space] = divisor * bounds[dim]
            cdivisor = IndexConstantOp.build(divisor)
            div = IndexDivOp.build(
                ids.results[axis], cdivisor.result(),
            )
            cbound = IndexConstantOp.build(bounds[dim])
            mod = IndexModOp.build(
                div.result(), cbound.result(),
            )
            new_ops = [
                ids,
                cdivisor,
                div,
                cbound,
                mod,
                *new_ops,
            ]
            ivs[dim] = mod.result()

        return new_ops, ivs, innermost_loop

def tile_reduce(
    module: ModuleOp,
    tile_sizes: dict[int, int],
    parallel_iteration_spaces: dict[
        int, tuple[CollectiveLevel, CollectiveAxis]
    ],
    hero_label_filter: str | None = None,
    epilogue_label_filter: str | None = None,
) -> ModuleOp:
    return TileReduce(
        tile_sizes,
        parallel_iteration_spaces,
        hero_label_filter,
        epilogue_label_filter,
    ).run(module)
