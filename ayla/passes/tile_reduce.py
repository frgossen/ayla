"""Tile the parallel and reduction dimensions of reductions."""
from __future__ import annotations

import math
from collections.abc import Set
from typing import cast

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
    AddrMemTensorType,
    AllocOp,
    BlockIdOp,
    CollectiveAxis,
    CollectiveLevel,
    ElementwiseOp,
    ForOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    KernelOp,
    MemoryEffect,
    MemTensorType,
    ModuleOp,
    ReduceOp,
    ThreadIdOp,
    TileOp,
    WarpGroupIdOp,
    is_grid_collective,
)
from ._utils import label_matches


def _materialize_tile(
    arg: Value,
    dims: list[int],
    tile_sizes: dict[int, int],
    ivs: dict[int, Value],
) -> tuple[list[Op], Value]:
    if not dims:
        return [], arg

    typ = clone_type(arg.type)
    assert isinstance(typ, MemTensorType)
    indices: list[Value | int] = [0] * typ.rank
    sizes = list(typ.sizes)
    for dim in dims:
        sizes[dim] = tile_sizes[dim]
        indices[dim] = ivs[dim]
    typ.sizes = sizes
    tile_op = TileOp.build(arg, indices, typ)
    return [tile_op], tile_op.result()


def _materialize_partials(
    value: Value,
    reduction_dim: int,
    num_partials: int,
    neutral_element: float | int,
) -> tuple[list[Op], Value]:
    if isinstance(value.definition, (AllocOp, Block)):
        assert isinstance(value.type, MemTensorType)
        sizes = list(value.type.sizes)
        sizes[reduction_dim] = num_partials
        attrs: dict[str, object] = {"sizes": sizes}
        if isinstance(value.type, AddrMemTensorType):
            attrs["strides"] = AddrMemTensorType.canonical_strides(sizes)
        typ = clone_type(value.type, attrs)
        alloc = AllocOp.build_with_fresh_id(typ, init=neutral_element)
        return [alloc], alloc.result()

    tile = value.definition
    assert isinstance(tile, TileOp), (
        "TileReduce reduction tiling only supports TileOp destination "
        "view chains"
    )
    ops, replayed_arg = _materialize_partials(
        tile.arg, reduction_dim, num_partials, neutral_element,
    )
    tile_type = tile.result().type
    replayed_type = replayed_arg.type
    assert isinstance(tile_type, MemTensorType)
    assert isinstance(replayed_type, MemTensorType)
    sizes = list(tile_type.sizes)
    sizes[reduction_dim] = replayed_type.sizes[reduction_dim]
    typ = clone_type(replayed_type, {"sizes": sizes})
    replayed = TileOp.build(
        replayed_arg,
        list(tile.indices),
        typ,
        maybe_partial=tile.maybe_partial,
    )
    ops.append(replayed)
    return ops, replayed.result()


def _materialize_iteration_space(
    dim: int,
    bounds: dict[int, int],
    parallel_iteration_space: (
        tuple[CollectiveLevel, CollectiveAxis] | None
    ),
    divisor: int,
) -> tuple[list[Op], Value, ForOp | None]:
    if parallel_iteration_space is None:
        loop = ForOp.build(
            lb=0,
            ub=bounds[dim],
            attrs={"label": f"d{dim}"},
        )
        return [loop], loop.iv, loop

    level, axis = parallel_iteration_space
    if level in (CollectiveLevel.BLOCK, CollectiveLevel.THREAD):
        id_op_type = {
            CollectiveLevel.BLOCK: BlockIdOp,
            CollectiveLevel.THREAD: ThreadIdOp,
        }[level]
        ids = id_op_type.build(axis)
        id_ops: list[Op] = [ids]
        parallel_id = ids.value
    elif level == CollectiveLevel.WARP:
        assert axis == CollectiveAxis.X
        ids = ThreadIdOp.build(CollectiveAxis.X)
        warp_size = IndexConstantOp.build(32)
        warp_id = IndexDivOp.build(ids.value, warp_size.result())
        id_ops = [ids, warp_size, warp_id]
        parallel_id = warp_id.result()
    else:
        id_op_type = {
            CollectiveLevel.WG: WarpGroupIdOp,
        }[level]
        ids = id_op_type.build()
        id_ops = [ids]
        parallel_id = ids.results[axis]
    cdivisor = IndexConstantOp.build(divisor)
    div = IndexDivOp.build(
        parallel_id, cdivisor.result(),
    )
    cbound = IndexConstantOp.build(bounds[dim])
    mod = IndexModOp.build(
        div.result(), cbound.result(),
    )
    return [*id_ops, cdivisor, div, cbound, mod], mod.result(), None


def _narrow_collective(
    collective: dict[CollectiveAxis, CollectiveLevel],
    parallel_iteration_space: (
        tuple[CollectiveLevel, CollectiveAxis] | None
    ),
) -> dict[CollectiveAxis, CollectiveLevel]:
    result = dict(collective)
    if parallel_iteration_space is not None:
        level, axis = parallel_iteration_space
        assert level.join(collective[axis]) == collective[axis]
        result[axis] = level
    return result


def refine_operand_tiling_schemes(
    existing_schemes: list[int | None],
    operand_varying_dims: dict[int, Set[int]],
    bounds: dict[int, int],
    *,
    allow_initialize: bool,
) -> list[int | None]:
    """Keep operand schemes only while their tiles remain disjoint."""
    partitioned_dims = {dim for dim, bound in bounds.items() if bound > 1}
    schemes = list(existing_schemes)
    for operand_index, dims in operand_varying_dims.items():
        if not partitioned_dims <= dims:
            schemes[operand_index] = None
        elif schemes[operand_index] is None and allow_initialize:
            schemes[operand_index] = MemoryEffect.fresh_tiling_scheme_id()
    return schemes


def _refine_tiling_schemes(
    hero: ReduceOp,
    bounds: dict[int, int],
) -> list[int | None]:
    """Retain each operand's ID while its tiled accesses stay disjoint."""
    src_type = hero.src.type
    assert isinstance(src_type, MemTensorType)
    operand_varying_dims: dict[int, Set[int]] = {
        1: frozenset(range(src_type.rank)),
        0: frozenset(range(src_type.rank - 1)),
    }
    operand_tiling_scheme = cast(
        list[int | None], hero.attrs.get("operand_tiling_scheme", [None, None]),
    )
    allow_initialize = (
        is_grid_collective(hero.collective)
        and find_ancestor_of_type(hero, ForOp) is None
    )
    return refine_operand_tiling_schemes(
        operand_tiling_scheme,
        operand_varying_dims,
        bounds,
        allow_initialize=allow_initialize,
    )


class TileReduce(LoweringPass):
    """Tile reduction iteration spaces and selected epilogues.

    ``tile_sizes`` maps each tiled dimension to its tile size. The
    ``parallel_iteration_space`` applies to every tiled dimension, or is
    ``None`` to create sequential iteration spaces.
    Nested tiling is expressed by composing multiple pass invocations.
    A dimension without a parallel iteration space is lowered to a sequential
    loop whose label is derived from the dimension.
    The vector accumulation option selects vector accumulation instead of
    materialized scalar partial reductions.
    """

    tile_sizes: dict[int, int]
    parallel_iteration_space: (
        tuple[CollectiveLevel, CollectiveAxis] | None
    )
    hero_label_filter: str | None
    epilogue_label_filter: str | None
    vector_accumulation: bool

    def __init__(
        self,
        tile_sizes: dict[int, int],
        parallel_iteration_space: (
            tuple[CollectiveLevel, CollectiveAxis] | None
        ),
        hero_label_filter: str | None = None,
        epilogue_label_filter: str | None = None,
        vector_accumulation: bool = False,
    ) -> None:
        super().__init__(
            tile_sizes=dict(tile_sizes),
            parallel_iteration_space=parallel_iteration_space,
            hero_label_filter=hero_label_filter,
            epilogue_label_filter=epilogue_label_filter,
            vector_accumulation=vector_accumulation,
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
            if not (
                isinstance(op, ReduceOp)
                and label_matches(op, self.hero_label_filter)
            ):
                new_ops.append(op)
                i += 1
                continue
            epilogue: list[Op] = []
            for candidate in ops[i + 1:]:
                if not label_matches(candidate, self.epilogue_label_filter):
                    break
                epilogue.append(candidate)
            new_ops.extend(
                self._rewrite_hero_with_epilogue(
                    op, epilogue,
                )
            )
            i += 1 + len(epilogue)
        block.ops = new_ops

    def _rewrite_hero_with_epilogue(
        self, hero: ReduceOp, epilogue: list[Op],
    ) -> list[Op]:
        self._assert_preconditions(hero, epilogue)

        # Characterize tiled dimensions.
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        rank = src_type.rank
        reduction_dim = (
            rank - 1
            if rank - 1 in self.tile_sizes
            else None
        )
        parallel_dims = [
            dim for dim in self.tile_sizes if dim != reduction_dim
        ]
        bounds = {
            dim: cast(int, src_type.sizes[dim]) // tile_size
            for dim, tile_size in self.tile_sizes.items()
        }
        tiling_schemes = _refine_tiling_schemes(hero, bounds)

        # Update launch dimensions.
        if self.parallel_iteration_space is not None:
            launch_dim = math.prod(bounds.values())
            kernel = find_ancestor_of_type(hero, KernelOp)
            assert kernel is not None
            level, axis = self.parallel_iteration_space
            if level == CollectiveLevel.WARP:
                assert axis == CollectiveAxis.X
                launch_dim *= 32
            elif level == CollectiveLevel.WG:
                assert axis == CollectiveAxis.X
                launch_dim *= 128
            if level == CollectiveLevel.BLOCK:
                launch_dims = list(kernel.grid_dim)
            else:
                launch_dims = list(kernel.block_dim)
            assert launch_dims[axis] in (None, launch_dim), (
                f"launch dim conflict: {launch_dims[axis]} vs {launch_dim}"
            )
            launch_dims[axis] = launch_dim
            if level == CollectiveLevel.BLOCK:
                kernel.grid_dim = launch_dims
            else:
                kernel.block_dim = launch_dims

        # Create iteration spaces for the parallel dimensions.
        new_ops: list[Op] = []
        parallel_body_insertion_point = new_ops
        ivs: dict[int, Value] = {}
        divisor = 1
        for dim in parallel_dims:
            iteration_ops, iv, inner_loop = _materialize_iteration_space(
                dim,
                bounds,
                self.parallel_iteration_space,
                divisor,
            )
            ivs[dim] = iv
            parallel_body_insertion_point.extend(iteration_ops)
            divisor *= bounds[dim]
            if inner_loop is not None:
                parallel_body_insertion_point = inner_loop.body.ops

        # Declare partials if needed.
        partials: Value | None = None
        if reduction_dim is not None:
            neutral_element = ReduceOp.NEUTRAL_ELEMENTS[hero.fn]
            assert neutral_element is not None
            num_partials = (
                self.tile_sizes[reduction_dim]
                if self.vector_accumulation
                else bounds[reduction_dim]
            )
            partials_ops, partials = _materialize_partials(
                hero.dst, reduction_dim, num_partials, neutral_element,
            )
            parallel_body_insertion_point.extend(partials_ops)

        # Create tiles wrt. parallel dimension iteration spaces.
        src_ops, src = _materialize_tile(
            hero.src, parallel_dims, self.tile_sizes, ivs,
        )
        dst_ops, dst = _materialize_tile(
            hero.dst, parallel_dims, self.tile_sizes, ivs,
        )
        parallel_body_insertion_point.extend([*src_ops, *dst_ops])
        if partials is not None:
            partials_ops, partials = _materialize_tile(
                partials, parallel_dims, self.tile_sizes, ivs,
            )
            parallel_body_insertion_point.extend(partials_ops)

        # Create iteration spaces for the reduction dimensions.
        reduction_body_insertion_point = parallel_body_insertion_point
        if reduction_dim is not None:
            iteration_ops, reduction_iv, inner_loop = (
                _materialize_iteration_space(
                    reduction_dim,
                    bounds,
                    self.parallel_iteration_space,
                    divisor,
                )
            )
            parallel_body_insertion_point.extend(iteration_ops)
            if inner_loop is not None:
                reduction_body_insertion_point = inner_loop.body.ops
            ivs[reduction_dim] = reduction_iv

        # Create tiles wrt. reduction dimension iteration space.
        partial_ops: list[Op] = []
        partial: Value | None = None
        if reduction_dim is not None:
            assert partials is not None
            src_ops, src = _materialize_tile(
                src,
                [reduction_dim],
                self.tile_sizes,
                ivs,
            )
            if self.vector_accumulation:
                partial = partials
            else:
                partial_ops, partial = _materialize_tile(
                    partials,
                    [reduction_dim],
                    {reduction_dim: 1},
                    ivs,
                )

        # Create the partial reduction, if needed.
        tiled_collectiveness = _narrow_collective(
            hero.collective, self.parallel_iteration_space,
        )
        if reduction_dim is not None:
            assert partial is not None
            partial_attrs = hero.extra_attrs
            partial_attrs.pop("operand_tiling_scheme", None)
            partial_tiling_schemes = [None, tiling_schemes[1]]
            if (
                not self.vector_accumulation
                and find_ancestor_of_type(hero, ForOp) is None
            ):
                partial_tiling_schemes[0] = MemoryEffect.fresh_tiling_scheme_id()
            if any(scheme is not None for scheme in partial_tiling_schemes):
                partial_attrs["operand_tiling_scheme"] = partial_tiling_schemes
            label = partial_attrs.get("label")
            if isinstance(label, str):
                partial_attrs["label"] = f"{label}_partial"
            partial_reduce = ReduceOp.build(
                partial,
                src,
                fn=hero.fn,
                collective=tiled_collectiveness,
                attrs=partial_attrs,
            )
            reduction_body_insertion_point.extend([
                *partial_ops,
                *src_ops,
                partial_reduce,
            ])

        # Create final tiled reduction.
        final_reduction_collectiveness = (
            tiled_collectiveness if parallel_dims else dict(hero.collective)
        )
        final_src = partials if partials is not None else src
        final_reduce = clone_op(
            hero,
            {hero.dst: dst, hero.src: final_src},
            {"collective": final_reduction_collectiveness},
        )
        final_reduce.attrs.pop("operand_tiling_scheme", None)
        final_tiling_schemes = list(tiling_schemes)
        if reduction_dim is not None:
            final_tiling_schemes[1] = None
        if any(scheme is not None for scheme in final_tiling_schemes):
            final_reduce.attrs["operand_tiling_scheme"] = final_tiling_schemes
        parallel_body_insertion_point.append(final_reduce)

        # Rewrite the epilogue.
        parallel_body_insertion_point.extend(
            clone_op(op, {hero.dst: dst}) for op in epilogue
        )
        return new_ops

    def _assert_preconditions(
        self, hero: Op, epilogue: list[Op],
    ) -> None:
        assert isinstance(hero, ReduceOp)
        assert (
            hero.fn in ReduceOp.NEUTRAL_ELEMENTS
            and ReduceOp.NEUTRAL_ELEMENTS[hero.fn] is not None
        ), "TileReduce requires a reduction with a neutral element"
        assert all(isinstance(op, ElementwiseOp) for op in epilogue)
        assert all(
            operand is hero.dst
            for op in epilogue
            for operand in op.operands
        )

        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        rank = src_type.rank
        assert all(0 <= dim < rank for dim in self.tile_sizes), (
            f"TileReduce dimensions must be in [0, {rank})"
        )
        reduction_dim = (
            rank - 1
            if rank - 1 in self.tile_sizes
            else None
        )
        parallel_dims = [
            dim for dim in self.tile_sizes if dim != reduction_dim
        ]
        assert not parallel_dims or reduction_dim is None, (
            "TileReduce cannot tile parallel and reduction dimensions in "
            "one invocation"
        )
        if self.vector_accumulation:
            assert reduction_dim is not None, (
                "TileReduce vector accumulation requires reduction-dimension "
                "tiling"
            )
            assert self.parallel_iteration_space is None, (
                "TileReduce does not implement parallel vector accumulation; "
                "it would require atomic accumulator updates and is not "
                "currently considered useful"
            )
        for dim, tile_size in self.tile_sizes.items():
            size = cast(int, src_type.sizes[dim])
            assert 0 < tile_size <= size, (
                f"TileReduce requires dimension {dim} tile size to be in "
                f"[1, {size}]; got {tile_size}"
            )
            assert size % tile_size == 0, (
                f"TileReduce requires dimension {dim} size {size} to be "
                f"divisible by tile size {tile_size}"
            )
        if reduction_dim is not None:
            current = hero.dst
            while not isinstance(current.definition, (AllocOp, Block)):
                tile = current.definition
                assert isinstance(tile, TileOp), (
                    "TileReduce reduction tiling only supports TileOp "
                    "destination view chains"
                )
                current = tile.arg

        if self.parallel_iteration_space is None:
            return

        level, axis = self.parallel_iteration_space
        # TODO: Make flattened tile-to-CTA mappings cluster-aware and validate
        # them against the enclosing kernel's cluster_dim.
        kernel = find_ancestor_of_type(hero, KernelOp)
        assert kernel is not None
        assert (
            level != CollectiveLevel.CLUSTER
            and all(dim is None for dim in kernel.cluster_dim)
        ), "TileReduce does not support cluster-aware iteration spaces"

        # TODO: Support mappings across interdependent intra-block levels by
        # deriving relative IDs and composing their launch constraints.
        current_level = hero.collective[axis]
        assert (
            current_level == CollectiveLevel.GRID
            and level == CollectiveLevel.BLOCK
            or current_level == CollectiveLevel.BLOCK
            and level < CollectiveLevel.BLOCK
        ), (
            "TileReduce only supports grid-to-block and "
            "block-to-intra-block parallel tiling"
        )

def tile_reduce(
    module: ModuleOp,
    tile_sizes: dict[int, int],
    parallel_iteration_space: (
        tuple[CollectiveLevel, CollectiveAxis] | None
    ),
    hero_label_filter: str | None = None,
    epilogue_label_filter: str | None = None,
    vector_accumulation: bool = False,
) -> ModuleOp:
    return TileReduce(
        tile_sizes,
        parallel_iteration_space,
        hero_label_filter,
        epilogue_label_filter,
        vector_accumulation,
    ).run(module)
