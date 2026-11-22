"""Register layout semantics of the GPU dialect, independent of any backend.

Coordinates are relative to the view and exclude the runtime lane/warp base.
Root register indices survive static WGMMA views. Tensor types expose this
mapping to analyses, transformations, and backends.
"""

from dataclasses import dataclass
import itertools
import math
from typing import cast

from .base.ir import Value
from .gpu_dialect import (
    RegisterElement,
    RegisterLayout,
    ThreadIndex,
    RegisterViewOp,
    BF16Type,
    F16Type,
    F32Type,
    RegMemTensorType,
    RmemTensorType,
    TileOp,
    WgrmemTensorType,
    WrmemTensorType,
    root_buffer,
)



def register_layout(value: Value) -> RegisterLayout:
    typ = value.type
    if not isinstance(typ, (RmemTensorType, WrmemTensorType, WgrmemTensorType)):
        raise NotImplementedError(f"no register layout for {typ}")
    shape = typ.static_shape
    if isinstance(value.definition, RegisterViewOp):
        return RegisterLayout(shape, (RegisterElement(value.definition.index, ()),))
    if isinstance(typ, WgrmemTensorType):
        if typ.layout in ("wgmma_layout_0", "wgmma_warp"):
            elements = tuple(
                RegisterElement(2 * pair + e, (row, col + e))
                for pair, row, col in wgmma_view_pairs(value)
                for e in (0, 1)
            )
            return RegisterLayout(
                shape,
                elements,
                4,
                2,
                distributed_rows=True,
                thread_bases=(
                    ((ThreadIndex("thread_x", 32, 4, 16), ThreadIndex("lane", 4))
                     if typ.warps == 4 else (ThreadIndex("lane", 4),)),
                    (ThreadIndex("lane", modulus=4, scale=2),),
                ),
            )
        if typ.layout == "wgmma_row":
            if (
                len(shape) != 2
                or shape[-1] != 1
                or shape[0] % 64
                or typ.warps != 4
                or typ.dtype != F32Type()
            ):
                raise ValueError(
                    "wgmma_row requires an F32 Mx1 tensor with M divisible by 64 and four warps"
                )
            if isinstance(value.definition, TileOp):
                raise NotImplementedError("WGMMA row register views are unsupported")
            elements = tuple(
                RegisterElement(i, (64 * (i // 2) + 8 * (i % 2), 0))
                for i in range(shape[0] // 32)
            )
            return RegisterLayout(
                shape,
                elements,
                distributed_rows=True,
                replicas=4,
                thread_bases=(
                    (ThreadIndex("thread_x", 32, 4, 16), ThreadIndex("lane", 4)),
                    (),
                ),
            )
        raise NotImplementedError(f"unsupported WGMMA register layout: {typ.layout}")
    lanes = 32 if isinstance(typ, WrmemTensorType) else 1
    if isinstance(typ, WrmemTensorType):
        _warp_register_count(typ)
    coordinates = (
        itertools.product(*(range(s) for s in shape[:-1]), range(0, shape[-1], lanes))
        if shape
        else [()]
    )
    elements = tuple(RegisterElement(i, tuple(c)) for i, c in enumerate(coordinates))
    bases = ((),) * (len(shape) - 1) + ((ThreadIndex("lane"),),) if lanes > 1 else ()
    return RegisterLayout(shape, elements, lanes, thread_bases=bases)


def _warp_register_count(typ: WrmemTensorType) -> int:
    if typ.layout not in ("default", "lane", "lane0"):
        raise ValueError(f"unsupported warp register layout: {typ.layout}")
    shape = typ.static_shape
    if not shape:
        raise ValueError("warp registers require a nonzero rank")
    if typ.layout == "lane0" and shape[-1] != 1:
        raise ValueError("lane0 requires one column")
    return math.prod(shape[:-1]) * math.ceil(shape[-1] / 32)


def register_indices(
    typ: RegMemTensorType, value: Value | None = None, elements_per_register: int = 1
) -> tuple[int, ...]:
    """Storage indices selected by a tensor or static view in its root.

    Logical elements may be packed by the backend. WGMMA's adjacent pairs
    allow one or two elements per storage register; private and cyclic warp
    tensors currently require one.
    """
    if value is not None and value.type != typ:
        raise ValueError("register view type does not match the tensor type")
    if value is not None and isinstance(value.definition, RegisterViewOp):
        if elements_per_register != 1:
            raise ValueError('scalar register views cannot be packed')
        return (value.definition.index,)
    if isinstance(typ, WgrmemTensorType):
        if elements_per_register not in (1, 2):
            raise ValueError(
                "warpgroup storage supports one or two elements per register"
            )
        if typ.layout in ("wgmma_layout_0", "wgmma_warp"):
            pairs = [
                p for p, _, _ in wgmma_view_pairs(value or Value(name=None, type=typ))
            ]
            return tuple(
                p if elements_per_register == 2 else 2 * p + e
                for p in pairs
                for e in range(2 // elements_per_register)
            )
        if value is not None and isinstance(value.definition, TileOp):
            raise NotImplementedError(
                f"WGRMEM views are unsupported for layout {typ.layout!r}"
            )
        count = typ.num_elements_per_thread
        if count % elements_per_register:
            raise ValueError("register packing must divide the fragment element count")
        return tuple(range(count // elements_per_register))
    if elements_per_register != 1:
        raise ValueError(
            "private and cyclic warp layouts require one element per register"
        )
    count = (
        _warp_register_count(typ)
        if isinstance(typ, WrmemTensorType)
        else math.prod(typ.static_shape)
    )
    return tuple(range(count))


def wgmma_pair_offset(pair_index: int, n_dim: int) -> tuple[int, int]:
    """Thread-independent root offset of a WGMMA pair's first element.

    The runtime warp/lane offset selects the thread's base coordinate. This
    returns the additional logical ``(row, column)`` offset for ``pair_index``
    in that thread's register fragment.
    """
    assert n_dim % 8 == 0
    pairs_per_m_tile = n_dim // 4
    m, pair_in_m_tile = divmod(pair_index, pairs_per_m_tile)
    w, pair_in_16_columns = divmod(pair_in_m_tile, 4)
    di, dj = [(0, 0), (8, 0), (0, 8), (8, 8)][pair_in_16_columns]
    return (m * 64 + di, w * 16 + dj)


def static_view_origin(value: Value) -> tuple[int, ...]:
    """Logical element coordinates of a static view's origin in its root."""
    value_type = value.type
    assert isinstance(value_type, WgrmemTensorType)
    offsets = [0] * value_type.rank
    while isinstance(value.definition, TileOp):
        tile = value.definition
        for dim, index in enumerate(tile.static_indices):
            if index is None:
                raise NotImplementedError(
                    "WGRMEM views currently require static tile indices"
                )
            result_type = tile.result().type
            assert isinstance(result_type, WgrmemTensorType)
            offsets[dim] += index * cast(int, tile.index_steps[dim])
        value = tile.arg
    return tuple(offsets)


def wgmma_view_pairs(value: Value) -> list[tuple[int, int, int]]:
    """Map a WGMMA WGRMEM value or view to pairs in its root.

    Each tuple is ``(root_pair_index, view_row_offset, view_column_offset)``.
    Pair ``p`` names F32 registers ``2*p`` and ``2*p + 1``, or the packed
    BF16/F16 b32 register ``p``. The offsets are the thread-independent
    logical coordinates of the pair's first element relative to the view;
    store lowering adds the runtime warp/lane coordinate separately.
    """
    root = root_buffer(value)
    assert root is not None
    view_type, root_type = value.type, root.type
    assert isinstance(view_type, WgrmemTensorType)
    assert isinstance(root_type, WgrmemTensorType)
    assert root_type.layout in ("wgmma_layout_0", "wgmma_warp")
    assert view_type.layout in ("wgmma_layout_0", "wgmma_warp")
    assert view_type.rank == root_type.rank == 2
    assert root_type.warps == (1 if root_type.layout == "wgmma_warp" else 4)
    assert view_type.warps == (1 if view_type.layout == "wgmma_warp" else 4)
    if view_type.layout == "wgmma_warp":
        if view_type.sizes[0] != 8 or root_type.warps == 4 and static_view_origin(value)[0] % 64 not in (0, 8):
            raise ValueError("warp WGMMA fragments require one owned eight-row half")
    elif root_type.layout != view_type.layout:
        raise ValueError("a warp fragment cannot be widened to a warpgroup")
    assert view_type.dtype == root_type.dtype
    assert isinstance(view_type.dtype, (F32Type, BF16Type, F16Type)), (
        f"WGMMA views support f32, bf16, and f16; got {view_type.dtype}"
    )
    assert root_type.num_elements_per_thread % 2 == 0
    row_start, col_start = static_view_origin(value)
    view_m, view_n = cast(tuple[int, int], view_type.sizes)
    root_n = cast(int, root_type.sizes[1])
    pairs = [
        (pair_index, row - row_start, col - col_start)
        for pair_index in range(root_type.num_elements_per_thread // 2)
        for row, col in [wgmma_pair_offset(pair_index, root_n) if root_type.warps == 4 else (0, pair_index * 8)]
        if row_start <= row < row_start + view_m
        and col_start <= col < col_start + view_n
    ]
    assert len(pairs) == view_m * view_n // (view_type.warps * 64)
    return pairs


@dataclass(frozen=True)
class RegisterGroup:
    destination: RegisterElement
    sources: tuple[RegisterElement, ...]
    shuffle_lanes: int


def register_groups(
    source: RegisterLayout, destination: RegisterLayout
) -> tuple[RegisterGroup, ...]:
    """Plan local folds from logical coordinates, independent of PTX."""
    p = destination.shape[-1]
    if source.distributed_rows != destination.distributed_rows:
        raise NotImplementedError("reduction operands require matching row ownership")
    if p != 1 and (
        p % source.column_period or source.column_period != destination.column_period
    ):
        raise NotImplementedError(
            "partial groups change lane ownership; use a logical gather"
        )
    return tuple(
        RegisterGroup(
            d,
            tuple(
                s
                for s in source.row_elements(d.coordinates[:-1])
                if p == 1 or s.coordinates[-1] % p == d.coordinates[-1]
            ),
            source.column_lanes if p == 1 else 1,
        )
        for d in destination.elements
    )


def broadcast_register(
    source: RegisterLayout, destination: RegisterLayout, element: RegisterElement
) -> tuple[int, bool]:
    if not source.shape:
        return 0, False
    if source.shape == destination.shape and (
        source.column_period,
        source.distributed_rows,
        source.replicas,
    ) == (
        destination.column_period,
        destination.distributed_rows,
        destination.replicas,
    ):
        coordinates = element.coordinates
    elif (
        source.shape[:-1] == destination.shape[:-1]
        and source.shape[-1] == 1
        and source.distributed_rows == destination.distributed_rows
    ):
        coordinates = (*element.coordinates[:-1], 0)
    else:
        raise NotImplementedError(
            f"unsupported register broadcast: {source.shape} -> {destination.shape}"
        )
    index = next(e.register for e in source.elements if e.coordinates == coordinates)
    shuffle = (
        source.column_lanes > 1 and source.shape[-1] == 1 and destination.shape[-1] != 1
    )
    return index, shuffle
