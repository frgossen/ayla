"""Tile the parallel and reduction dimensions of reductions."""
from __future__ import annotations

import math
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
    build_hardware_id,
    AddrMemTensorType,
    AllocOp,
    CollectiveAxis,
    CollectiveLevel,
    ConstantOp,
    ElementwiseOp,
    ForOp,
    IndexConstantOp,
    IndexDivOp,
    IndexModOp,
    KernelOp,
    MemTensorType,
    ModuleOp,
    ReduceOp,
    RmemTensorType,
    TileOp,
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
    reduction_fn: str,
    reduction_dst: Value,
    num_partials: int,
) -> tuple[list[Op], Value]:
    # TODO: Resolve gmem allocations through buffer promotion, buffer reuse,
    # or scratch gmem kernel arguments. See the design notes:
    # https://docs.google.com/document/d/1UG1GMse251yAtngItIoH6W46qGGJE3VitYRJ0kw_0BM/edit?tab=t.0#heading=h.fi4redhw9yo6
    assert isinstance(reduction_dst.type, MemTensorType)
    reduction_dim = reduction_dst.type.rank - 1
    if isinstance(reduction_dst.definition, (AllocOp, Block)):
        sizes = list(reduction_dst.type.sizes)
        sizes[reduction_dim] = num_partials
        attrs: dict[str, object] = {"sizes": sizes}
        if isinstance(reduction_dst.type, AddrMemTensorType):
            attrs["strides"] = AddrMemTensorType.canonical_strides(sizes)
        typ = clone_type(reduction_dst.type, attrs)
        neutral_element = ReduceOp.NEUTRAL_ELEMENTS[reduction_fn]
        assert neutral_element is not None
        alloc = AllocOp.build_with_fresh_id(typ)
        neutral = ConstantOp.build(
            neutral_element, RmemTensorType.build([], reduction_dst.type.dtype),
        )
        fill = ElementwiseOp.build(
            alloc.result(), srcs=[neutral.result()], fn="fill",
        )
        return [alloc, neutral, fill], alloc.result()

    tile = reduction_dst.definition
    assert isinstance(tile, TileOp), (
        "TileReduce reduction tiling only supports TileOp destination "
        "view chains"
    )
    ops, replayed_arg = _materialize_partials(
        reduction_fn, tile.arg, num_partials,
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
    id_ops, parallel_id = build_hardware_id(level, axis)
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


def _clone_hero_with_epilogue(
    hero: ReduceOp, epilogue: list[Op], dst: Value, src: Value,
    collective: dict[CollectiveAxis, CollectiveLevel],
) -> list[Op]:
    return [
        clone_op(
            hero, {hero.dst: dst, hero.src: src}, {"collective": collective},
        ),
        *(clone_op(op, {hero.dst: dst}) for op in epilogue),
    ]

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
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        bounds = {
            dim: cast(int, src_type.sizes[dim]) // tile_size
            for dim, tile_size in self.tile_sizes.items()
        }
        self._update_launch_dimensions(hero, bounds)

        if src_type.rank - 1 not in self.tile_sizes:
            return self._rewrite_hero_with_epilogue_parallel(
                hero, epilogue, bounds,
            )
        if self.vector_accumulation:
            return self._rewrite_hero_with_epilogue_reduction_accumulate(
                hero, epilogue, bounds,
            )
        return self._rewrite_hero_with_epilogue_reduction_hierarchical(
            hero, epilogue, bounds,
        )

    def _update_launch_dimensions(
        self, hero: ReduceOp, bounds: dict[int, int],
    ) -> None:
        if self.parallel_iteration_space is None:
            return
        launch_dim = math.prod(bounds.values())
        kernel = find_ancestor_of_type(hero, KernelOp)
        assert kernel is not None
        level, axis = self.parallel_iteration_space
        if level == CollectiveLevel.THREAD_32:
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

    def _rewrite_hero_with_epilogue_parallel(
        self, hero: ReduceOp, epilogue: list[Op], bounds: dict[int, int],
    ) -> list[Op]:
        new_ops: list[Op] = []
        body = new_ops
        ivs: dict[int, Value] = {}
        divisor = 1
        parallel_dims = list(self.tile_sizes)
        for dim in parallel_dims:
            iteration_ops, iv, inner_loop = _materialize_iteration_space(
                dim, bounds, self.parallel_iteration_space, divisor,
            )
            ivs[dim] = iv
            body.extend(iteration_ops)
            divisor *= bounds[dim]
            if inner_loop is not None:
                body = inner_loop.body.ops

        src_ops, src = _materialize_tile(
            hero.src, parallel_dims, self.tile_sizes, ivs,
        )
        dst_ops, dst = _materialize_tile(
            hero.dst, parallel_dims, self.tile_sizes, ivs,
        )
        body.extend([*src_ops, *dst_ops])
        collective = (
            _narrow_collective(hero.collective, self.parallel_iteration_space)
            if parallel_dims else dict(hero.collective)
        )
        body.extend(_clone_hero_with_epilogue(
            hero, epilogue, dst, src, collective,
        ))
        return new_ops

    def _rewrite_hero_with_epilogue_reduction_hierarchical(
        self, hero: ReduceOp, epilogue: list[Op], bounds: dict[int, int],
    ) -> list[Op]:
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        new_ops, partials = _materialize_partials(
            hero.fn, hero.dst, bounds[src_type.rank - 1],
        )
        iteration_ops, iv, inner_loop = _materialize_iteration_space(
            src_type.rank - 1, bounds, self.parallel_iteration_space, divisor=1,
        )
        new_ops.extend(iteration_ops)
        body = inner_loop.body.ops if inner_loop is not None else new_ops
        ivs = {src_type.rank - 1: iv}
        partials_ops, partial = _materialize_tile(
            partials, [src_type.rank - 1], {src_type.rank - 1: 1}, ivs,
        )
        src_ops, src = _materialize_tile(
            hero.src, [src_type.rank - 1], self.tile_sizes, ivs,
        )
        body.extend([
            *partials_ops, *src_ops, self._build_partial_reduce(hero, partial, src),
        ])
        new_ops.extend(_clone_hero_with_epilogue(
            hero, epilogue, hero.dst, partials, dict(hero.collective),
        ))
        return new_ops

    def _rewrite_hero_with_epilogue_reduction_accumulate(
        self, hero: ReduceOp, epilogue: list[Op], bounds: dict[int, int],
    ) -> list[Op]:
        src_type = hero.src.type
        assert isinstance(src_type, MemTensorType)
        new_ops, accumulator = _materialize_partials(
            hero.fn, hero.dst, self.tile_sizes[src_type.rank - 1],
        )
        iteration_ops, iv, loop = _materialize_iteration_space(
            src_type.rank - 1, bounds, None, divisor=1,
        )
        assert loop is not None
        new_ops.extend(iteration_ops)
        src_ops, src = _materialize_tile(
            hero.src, [src_type.rank - 1], self.tile_sizes, {src_type.rank - 1: iv},
        )
        loop.body.ops.extend([
            *src_ops, self._build_partial_reduce(hero, accumulator, src),
        ])
        new_ops.extend(_clone_hero_with_epilogue(
            hero, epilogue, hero.dst, accumulator, dict(hero.collective),
        ))
        return new_ops

    def _build_partial_reduce(
        self, hero: ReduceOp, dst: Value, src: Value,
    ) -> ReduceOp:
        attrs = hero.extra_attrs
        label = attrs.get("label")
        if isinstance(label, str):
            attrs["label"] = f"{label}_partial"
        return ReduceOp.build(
            dst, src, fn=hero.fn,
            collective=_narrow_collective(
                hero.collective, self.parallel_iteration_space,
            ),
            attrs=attrs,
        )

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
