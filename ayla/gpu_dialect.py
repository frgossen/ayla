"""Concrete IR ops and types for the Ayla dialect."""
from __future__ import annotations

import enum
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Iterator, Literal, cast

from .base.ir import Block, Dialect, Op, Type, Value, find_ancestor_of_type
from .base.printer import Printer


class GpuDialect(Dialect):
    def __init__(self) -> None:
        super().__init__("gpu")

    def parse_module(self, source: str) -> ModuleOp:
        module = self.parse(source)
        assert isinstance(module, ModuleOp)
        return module


gpu_dialect = GpuDialect()


class GpuOp(Op):
    """Base for GPU operations with no operation-defined attributes."""

    @property
    def extra_attrs(self) -> dict[str, object]:
        return dict(self.attrs)


@gpu_dialect.register_op("module")
class ModuleOp(GpuOp):
    _DEFAULT_ATTRS = {
        "version": "9.0",
        "target": "sm_90a",
        "address_size": 64,
    }

    @classmethod
    def build(
        cls,
        body: Block,
        attrs: dict[str, object] | None = None,
    ) -> "ModuleOp":
        merged = {**cls._DEFAULT_ATTRS, **(attrs or {})}
        return cls(
            operands=[], results=[], attrs=merged, blocks=[body],
        )

    @property
    def body(self) -> Block:
        return self.blocks[0]

    @property
    def version(self) -> str:
        value = self.attrs["version"]
        assert isinstance(value, str)
        return value

    @property
    def target(self) -> str:
        value = self.attrs["target"]
        assert isinstance(value, str)
        return value

    @property
    def address_size(self) -> int:
        value = self.attrs["address_size"]
        assert isinstance(value, int)
        return value

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key not in self._DEFAULT_ATTRS
        }

    def print(self, p: Printer) -> None:
        p.write("module")
        non_default = {
            k: v for k, v in self.attrs.items()
            if self._DEFAULT_ATTRS.get(k) != v
        }
        p.write_extra_attrs(non_default)
        p.write(" ")
        self.body.print(p, skip_label=True)

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        attrs = p.read_extra_attrs()
        brace = p.read_literal("{")
        body = Block.parse(brace, p, arguments=[])
        return cls.build(body, attrs=attrs)
@gpu_dialect.register_op("kernel")
class KernelOp(GpuOp):
    @classmethod
    def build(
        cls,
        name: str,
        body: Block,
        grid_dim: tuple[int | None, int | None, int | None] = (None, None, None),
        block_dim: tuple[int | None, int | None, int | None] = (None, None, None),
        cluster_dim: tuple[int | None, int | None, int | None] = (None, None, None),
        num_warp_groups: int | None = None,
        ptx_optimization_level: int | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "KernelOp":
        attrs = dict(attrs or {})
        if "ptx_optimization_level" in attrs:
            raise ValueError(
                "pass ptx_optimization_level as an explicit argument"
            )
        if (
            ptx_optimization_level is not None
            and ptx_optimization_level not in range(5)
        ):
            raise ValueError("ptx_optimization_level must be in [0, 4]")
        if ptx_optimization_level is not None:
            attrs["ptx_optimization_level"] = ptx_optimization_level
        return cls(
            operands=[],
            results=[],
            attrs={
                "name": name,
                "grid_dim": list(grid_dim),
                "block_dim": list(block_dim),
                "cluster_dim": list(cluster_dim),
                "num_warp_groups": num_warp_groups,
                **attrs,
            },
            blocks=[body],
        )

    @property
    def name(self) -> str:
        return self.attrs["name"]  # type: ignore[return-value]

    @property
    def grid_dim(self) -> tuple[int | None, int | None, int | None]:
        return tuple(self.attrs["grid_dim"])  # type: ignore[return-value]

    @grid_dim.setter
    def grid_dim(self, value) -> None:
        self.attrs["grid_dim"] = list(value)

    @property
    def block_dim(self) -> tuple[int | None, int | None, int | None]:
        return tuple(self.attrs["block_dim"])  # type: ignore[return-value]

    @block_dim.setter
    def block_dim(self, value) -> None:
        self.attrs["block_dim"] = list(value)

    @property
    def cluster_dim(self) -> tuple[int | None, int | None, int | None]:
        return tuple(self.attrs["cluster_dim"])  # type: ignore[return-value]

    @property
    def num_warp_groups(self) -> int | None:
        return self.attrs["num_warp_groups"]  # type: ignore[return-value]

    @num_warp_groups.setter
    def num_warp_groups(self, value: int | None) -> None:
        self.attrs["num_warp_groups"] = value

    @property
    def ptx_optimization_level(self) -> int | None:
        return self.attrs.get("ptx_optimization_level")  # type: ignore[return-value]

    @property
    def num_threads(self) -> int | None:
        """Number of threads, or ``None`` if a block axis is unresolved."""
        if any(dim is None for dim in self.block_dim):
            return None
        return math.prod(self.block_dim)  # type: ignore[arg-type]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            k: v for k, v in self.attrs.items()
            if k not in (
                "name", "grid_dim", "block_dim", "cluster_dim",
                "num_warp_groups", "ptx_optimization_level",
            )
        }

    @property
    def signature(self) -> list[Type]:
        return [arg.type for arg in self.body.arguments]

    @property
    def body(self) -> Block:
        return self.blocks[0]

    def print(self, p: Printer) -> None:
        entry = self.body
        p.write(f"kernel @{self.name}(")
        for i, v in enumerate(entry.arguments):
            if i:
                p.write(", ")
            v.print(p)
        p.write(")")
        if any(d is not None for d in self.grid_dim):
            p.write(" grid_dim=")
            p.write_list_attr(list(self.grid_dim))
        if any(d is not None for d in self.block_dim):
            p.write(" block_dim=")
            p.write_list_attr(list(self.block_dim))
        if any(d is not None for d in self.cluster_dim):
            p.write(" cluster_dim=")
            p.write_list_attr(list(self.cluster_dim))
        if self.num_warp_groups is not None:
            p.write(f" num_warp_groups={self.num_warp_groups}")
        p.write_extra_attrs(
            self.attrs,
            skip_keys={
                "name", "grid_dim", "block_dim", "cluster_dim",
                "num_warp_groups",
            },
        )
        p.write(" ")
        self.body.print(p, skip_label=True)

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("@")
        name = p.read_identifier()
        p.read_literal("(")
        args: list[Value] = []
        if not p.can_read_literal(")"):
            args.append(Value.parse(p))
            while p.can_read_literal(","):
                p.read_literal(",")
                args.append(Value.parse(p))
        p.read_literal(")")
        grid_dim = (None, None, None)
        block_dim = (None, None, None)
        cluster_dim = (None, None, None)
        if p.can_read_literal("grid_dim"):
            p.read_literal("grid_dim")
            p.read_literal("=")
            grid_dim = tuple(p.read_list_attr())
        if p.can_read_literal("block_dim"):
            p.read_literal("block_dim")
            p.read_literal("=")
            block_dim = tuple(p.read_list_attr())
        if p.can_read_literal("cluster_dim"):
            p.read_literal("cluster_dim")
            p.read_literal("=")
            cluster_dim = tuple(p.read_list_attr())
        num_warp_groups = None
        if p.can_read_literal("num_warp_groups"):
            p.read_literal("num_warp_groups")
            p.read_literal("=")
            num_warp_groups = p.read_int()
        attrs = p.read_extra_attrs()
        brace = p.read_literal("{")
        body = Block.parse(brace, p, arguments=args)
        ptx_optimization_level = attrs.pop("ptx_optimization_level", None)
        assert ptx_optimization_level is None or isinstance(
            ptx_optimization_level, int,
        )
        return cls.build(
            name,
            body,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            num_warp_groups=num_warp_groups,
            ptx_optimization_level=ptx_optimization_level,
            attrs=attrs,
        )


class CollectiveLevel(enum.IntEnum):
    # TODO: Evaluate whether WARP and WG should remain named levels or be
    # represented generically as fixed-size thread collectives such as 32t
    # and 128t.
    THREAD = 0
    WARP = 1
    WG = 2
    BLOCK = 3
    CLUSTER = 4
    GRID = 5

    # TODO: Replace uses of join with max; collective levels form a total
    # order, and max may be easier to read.
    def join(self, other: CollectiveLevel) -> CollectiveLevel:
        return max(self, other)


class CollectiveAxis(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


def is_grid_collective(
    collective: dict[CollectiveAxis, CollectiveLevel],
    axes: tuple[CollectiveAxis, ...] = tuple(CollectiveAxis),
) -> bool:
    return all(
        collective[axis] >= CollectiveLevel.GRID for axis in axes
    )


def _print_collective(
    p: Printer,
    collective: dict[CollectiveAxis, CollectiveLevel],
) -> None:
    levels: dict[CollectiveLevel, str] = {}
    for axis in CollectiveAxis:
        dim = axis.name.lower()
        level = collective[axis]
        levels[level] = levels.get(level, "") + dim
    if levels == {CollectiveLevel.THREAD: "xyz"}:
        return
    p.write(" collective=")
    bracketed = len(levels) > 1
    if bracketed:
        p.write("[")
    for i, (level, dims) in enumerate(levels.items()):
        if i:
            p.write(", ")
        p.write(f"{dims}.{level.name.lower()}")
    if bracketed:
        p.write("]")


def _parse_collective(p) -> dict[CollectiveAxis, CollectiveLevel]:
    if not p.can_read_literal("collective"):
        return {
            axis: CollectiveLevel.THREAD for axis in CollectiveAxis
        }
    p.read_literal("collective")
    p.read_literal("=")
    bracketed = p.can_read_literal("[")
    if bracketed:
        p.read_literal("[")
    collective = {}
    while True:
        dims = p.read_identifier()
        p.read_literal(".")
        level = p.read_identifier()
        for dim in dims:
            collective[CollectiveAxis[dim.upper()]] = (
                CollectiveLevel[level.upper()]
            )
        if not bracketed or not p.can_read_literal(","):
            break
        p.read_literal(",")
    if bracketed:
        p.read_literal("]")
    return collective


@gpu_dialect.register_op("matmul")
class MatmulOp(GpuOp):
    @classmethod
    def build(
        cls,
        lhs: Value,
        rhs: Value,
        dst: Value,
        collective: dict[CollectiveAxis, CollectiveLevel],
        attrs: dict[str, object] | None = None,
        init: Value | None = None,
    ) -> "MatmulOp":
        a: dict[str, object] = dict(attrs or {})
        if "collective" in a:
            raise ValueError("pass collective as an explicit argument")
        a["collective"] = dict(collective)
        return cls(
            operands=[lhs, rhs, dst, *([init] if init is not None else [])],
            results=[], attrs=a,
        )

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]

    @property
    def dst(self) -> Value:
        return self.operands[2]

    @property
    def init(self) -> Value | None:
        """Accumulator initializer: a rank-0 scalar or a destination-shaped
        tensor. Without one, the matmul accumulates into ``dst``.

        TODO: Drop the initializer so a matmul always accumulates. A scalar
        becomes an ``elementwise fill`` of ``dst`` ahead of the matmul and a
        tensor a ``move`` into it; PromoteBuffers carries that prologue onto
        the promoted accumulator, as it already does the epilogue.
        """
        return self.operands[3] if len(self.operands) == 4 else None

    @property
    def collective(self) -> dict[CollectiveAxis, CollectiveLevel]:
        return self.attrs["collective"]  # type: ignore[return-value]

    def collective_level(self, axis: CollectiveAxis) -> CollectiveLevel:
        return self.collective[axis]

    def narrow_collective_level(
        self, axis: CollectiveAxis, level: CollectiveLevel,
    ) -> None:
        current = self.collective_level(axis)
        assert level.join(current) == current
        self.collective[axis] = level

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            k: v for k, v in self.attrs.items()
            if k != "collective"
        }

    def print(self, p: Printer) -> None:
        p.write("matmul(")
        for i, operand in enumerate(self.operands):
            if i:
                p.write(", ")
            operand.print(p)
        p.write(")")
        _print_collective(p, self.collective)
        p.write_extra_attrs(self.attrs, skip_keys=("collective",))

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        assert not results
        p.read_literal("(")
        operands = [Value.parse(p)]
        for _ in range(2):
            p.read_literal(",")
            operands.append(Value.parse(p))
        if p.can_read_literal(","):
            p.read_literal(",")
            operands.append(Value.parse(p))
        p.read_literal(")")
        collective = _parse_collective(p)
        attrs = p.read_extra_attrs()
        return cls.build(
            *operands[:3],
            collective=collective,
            attrs=attrs,
            init=operands[3] if len(operands) == 4 else None,
        )


@gpu_dialect.register_op("tile")
class TileOp(GpuOp):
    """Select a sub-region (view) of a buffer.

    Each index advances the view origin by the corresponding result size.

    ``maybe_partial`` marks a tile whose logical extent may cross the valid
    bounds of its source for some index values. Consumers must then provide
    defined out-of-bounds behavior.
    """

    @classmethod
    def build(
        cls,
        arg: Value,
        indices: list[Value | int],
        type: Type,
        attrs: dict[str, object] | None = None,
        maybe_partial: bool = False,
    ) -> "TileOp":
        operands: list[Value] = [arg]
        static_indices: list[int | None] = []
        for idx in indices:
            if isinstance(idx, int):
                static_indices.append(idx)
            else:
                static_indices.append(None)
                operands.append(idx)
        result = Value(name=None, type=type)
        attrs = dict(attrs or {})
        if maybe_partial:
            attrs["maybe_partial"] = True
        return cls(
            operands=operands,
            results=[result],
            attrs={"static_indices": static_indices, **attrs},
        )

    @property
    def arg(self) -> Value:
        return self.operands[0]

    @property
    def static_indices(self) -> list[int | None]:
        assert "static_indices" in self.attrs, (
            "TileOp: missing required `static_indices` attr"
        )
        return self.attrs["static_indices"]  # type: ignore[return-value]

    @property
    def indices(self) -> list[Value | int]:
        dyn = iter(self.operands[1:])
        return [
            s if s is not None else next(dyn) for s in self.static_indices
        ]

    @property
    def maybe_partial(self) -> bool:
        return bool(self.attrs.get("maybe_partial", False))

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key not in ("static_indices", "maybe_partial")
        }

    def print(self, p: Printer) -> None:
        for i, v in enumerate(self.results):
            if i:
                p.write(", ")
            v.print(p)
        p.write(f" = {self.op_id}(")
        self.arg.print(p)
        for idx in self.indices:
            p.write(", ")
            if isinstance(idx, int):
                p.write_literal_attr(idx)
            else:
                idx.print(p)
        p.write(")")
        p.write_extra_attrs(self.attrs, skip_keys=("static_indices",))

    @classmethod
    def parse_infix(cls, results: list[Value], prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("(")
        arg = Value.parse(p)
        operands: list[Value] = [arg]
        static_indices: list[int | None] = []
        while p.can_read_literal(","):
            p.read_literal(",")
            if p.is_at("%"):
                static_indices.append(None)
                operands.append(Value.parse(p))
            else:
                lit = p.read_literal_attr()
                if not isinstance(lit, int):
                    raise ValueError(
                        f"TileOp: static index must be int, got {lit!r}"
                    )
                static_indices.append(lit)
        p.read_literal(")")
        attrs = p.read_extra_attrs()
        return cls(
            operands=operands,
            results=results,
            attrs={"static_indices": static_indices, **attrs},
        )

@gpu_dialect.register_op("reshape")
class ReshapeOp(GpuOp):
    """Reinterpret a dense tensor with a different shape without moving data."""

    @classmethod
    def build(cls, arg: Value, type: Type) -> "ReshapeOp":
        return cls(
            operands=[arg],
            results=[Value(name=None, type=type)],
        )

    @property
    def arg(self) -> Value:
        return self.operands[0]


@gpu_dialect.register_op("reduce")
class ReduceOp(GpuOp):
    """Accumulate a reduction into a destination.

    The final source dimension is always the reduction dimension. Source and
    destination have the same rank, and all preceding dimensions are matching
    parallel dimensions.

    The ``collective`` attribute describes the collective level at which the
    reduction is executed. Output ownership remains encoded by the destination
    type, for example ``wrmem``.

    Equal source and destination shapes represent a partial-reduction step;
    a smaller destination combines those partials.
    """

    NEUTRAL_ELEMENTS: ClassVar[dict[str, float | int | None]] = {
        "max": float("-inf"),
        "sum": 0.0,
    }

    @classmethod
    def build(
        cls,
        dst: Value,
        src: Value,
        fn: str,
        collective: dict[CollectiveAxis, CollectiveLevel],
        attrs: dict[str, object] | None = None,
        init: float | int | None = None,
    ) -> "ReduceOp":
        a: dict[str, object] = dict(attrs or {})
        if "fn" in a:
            raise ValueError("pass fn as an explicit argument")
        if "collective" in a:
            raise ValueError("pass collective as an explicit argument")
        if "init" in a:
            raise ValueError("pass init as an explicit argument")
        a["fn"] = fn
        a["collective"] = dict(collective)
        if init is not None:
            a["init"] = init
        return cls(
            operands=[dst, src],
            results=[],
            attrs=a,
        )

    @property
    def dst(self) -> Value:
        return self.operands[0]

    @property
    def src(self) -> Value:
        return self.operands[1]

    @property
    def fn(self) -> str:
        return self.attrs["fn"]  # type: ignore[return-value]

    @property
    def init(self) -> float | int | None:
        return self.attrs.get("init")  # type: ignore[return-value]

    @property
    def collective(self) -> dict[CollectiveAxis, CollectiveLevel]:
        return self.attrs["collective"]  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key not in ("collective", "fn", "init")
        }

    def print(self, p: Printer) -> None:
        p.write(f"{self.op_id} {self.fn}(")
        for index, operand in enumerate(self.operands):
            if index:
                p.write(", ")
            operand.print(p)
        p.write(")")
        _print_collective(p, self.collective)
        p.write_extra_attrs(self.attrs, skip_keys=("collective", "fn"))

    @classmethod
    def parse_infix(
        cls, results: list[Value], prefix: str, p,
    ) -> "ReduceOp":  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        fn = p.read_identifier()
        p.read_literal("(")
        operands: list[Value] = []
        if not p.can_read_literal(")"):
            operands.append(Value.parse(p))
            while p.can_read_literal(","):
                p.read_literal(",")
                operands.append(Value.parse(p))
        p.read_literal(")")
        collective = _parse_collective(p)
        attrs = p.read_extra_attrs()
        if "fn" in attrs:
            raise ValueError("reduce function must not also appear in attrs")
        if "collective" in attrs:
            raise ValueError(
                "reduce collective must not also appear in attrs"
            )
        return cls(
            operands=operands,
            results=results,
            attrs={"fn": fn, "collective": collective, **attrs},
        )


@gpu_dialect.register_op("elementwise")
class ElementwiseOp(GpuOp):
    """One scalar function over corresponding elements of its sources.

    ``fn`` names the function and is part of the op's textual syntax. Exp,
    tanh, relu and the rest still share one operation because they differ only
    in the instructions they lower to.

    The first operand is the destination and the rest are sources. Passing the
    same value as the destination and first source computes in place.

    Lowering minimizes temporary register live ranges by default; ``ilp``
    asks for one wide schedule with more instruction-level parallelism
    instead. A source is only overwritten when it is also the destination;
    the backend remains responsible for register reuse between distinct values.

    ``fill`` broadcasts a rank-0 source into every destination element.

    TODO: generalize to several destinations.
    """

    ARITIES: ClassVar[dict[str, int]] = {
        "convert": 1,
        "fill": 1,
        "mul": 2,
        "sigmoid": 1,
        "silu": 1,
    }

    @classmethod
    def build(
        cls,
        dst: Value,
        srcs: list[Value],
        fn: str,
        attrs: dict[str, object] | None = None,
        *,
        ilp: bool = False,
    ) -> "ElementwiseOp":
        return cls(
            operands=[dst, *srcs],
            results=[],
            attrs={"fn": fn, **(attrs or {}), **({"ilp": True} if ilp else {})},
        )

    @property
    def dst(self) -> Value:
        return self.operands[0]

    def src(self, index: int) -> Value:
        return self.operands[index + 1]

    @property
    def num_srcs(self) -> int:
        return len(self.operands) - 1

    @property
    def fn(self) -> str:
        return self.attrs["fn"]  # type: ignore[return-value]

    @property
    def ilp(self) -> bool:
        """Whether lowering should prefer one wide schedule over few registers."""
        return self.attrs.get("ilp", False)  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {key: value for key, value in self.attrs.items() if key != "fn"}

    def print(self, p: Printer) -> None:
        p.write(f"{self.op_id} {self.fn}(")
        for index, operand in enumerate(self.operands):
            if index:
                p.write(", ")
            operand.print(p)
        p.write(")")
        p.write_extra_attrs(self.attrs, skip_keys=("fn",))

    @classmethod
    def parse_infix(
        cls, results: list[Value], prefix: str, p,
    ) -> "ElementwiseOp":  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        fn = p.read_identifier()
        p.read_literal("(")
        operands: list[Value] = []
        if not p.can_read_literal(")"):
            operands.append(Value.parse(p))
            while p.can_read_literal(","):
                p.read_literal(",")
                operands.append(Value.parse(p))
        p.read_literal(")")
        attrs = p.read_extra_attrs()
        if "fn" in attrs:
            raise ValueError(
                "elementwise function must not also appear in attrs"
            )
        return cls(
            operands=operands,
            results=results,
            attrs={"fn": fn, **attrs},
        )


@gpu_dialect.register_op("alloc")
class AllocOp(GpuOp):
    """Allocate a register- or shared-memory tensor.

    An allocation only provides storage; ``elementwise fill`` initializes it.

    The `id` attr names the buffer: two allocations are the same one only if
    they carry the same id, which is what lets CSE treat an allocation like any
    other value-identical op. Every allocation carries one.

    A pass that wants a buffer of its own calls `build_with_fresh_id`, which
    takes the next id from a counter shared by the process. `build` takes the id
    as an argument instead, for when it has to be a particular one.
    """

    _id_counter: ClassVar[Iterator[int]] = itertools.count()

    @classmethod
    def fresh_id(cls) -> int:
        return next(cls._id_counter)

    @classmethod
    def build(
        cls,
        type: Type,
        id: int,
        attrs: dict[str, object] | None = None,
    ) -> "AllocOp":
        a: dict[str, object] = dict(attrs or {})
        if "init" in a:
            raise ValueError("alloc does not initialize; use elementwise fill")
        a["id"] = id
        result = Value(name=None, type=type)
        return cls(operands=[], results=[result], attrs=a)

    @classmethod
    def build_with_fresh_id(
        cls,
        type: Type,
        attrs: dict[str, object] | None = None,
    ) -> "AllocOp":
        """Build an allocation that is nobody else's buffer."""
        return cls.build(type, id=cls.fresh_id(), attrs=attrs)

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items() if key != "id"
        }


@gpu_dialect.register_op("reinterpret")
class ReinterpretOp(GpuOp):
    """Reinterpret a typed region at a byte offset into existing storage.

    The operation does not allocate, convert, or move data. Its result aliases
    ``arg`` and may describe part of it with a different tensor shape,
    dtype, or physical layout. Only shared-memory storage and result types are
    supported today.
    """

    @classmethod
    def build(
        cls,
        arg: Value,
        type: Type,
        byte_offset: int = 0,
    ) -> "ReinterpretOp":
        if not isinstance(arg.type, SmemTensorType):
            raise ValueError("reinterpret: arg must be shared memory")
        if not isinstance(type, SmemTensorType):
            raise ValueError("reinterpret: result must be shared memory")
        if not isinstance(byte_offset, int) or byte_offset < 0:
            raise ValueError("reinterpret: byte_offset must be non-negative")
        return cls(
            operands=[arg],
            results=[Value(name=None, type=type)],
            attrs={"byte_offset": byte_offset},
        )

    @property
    def arg(self) -> Value:
        return self.operands[0]

    @property
    def byte_offset(self) -> int:
        return self.attrs["byte_offset"]  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "byte_offset"
        }


@gpu_dialect.register_op("move")
class MoveOp(GpuOp):
    """Move data into a destination, with optional indices and collectivity."""

    @classmethod
    def build(
        cls,
        dst: Value,
        src: Value,
        indices: list[Value] | None = None,
        *,
        collective: dict[CollectiveAxis, CollectiveLevel],
        multicast: bool = False,
        attrs: dict[str, object] | None = None,
    ) -> "MoveOp":
        operands = [dst, src] + (indices or [])
        attrs = dict(attrs or {})
        if "multicast" in attrs:
            raise ValueError("pass multicast as an explicit argument")
        attrs["collective"] = dict(collective)
        if multicast is not False:
            attrs["multicast"] = multicast
        return cls(
            operands=operands, results=[], attrs=attrs,
        )

    @property
    def dst(self) -> Value:
        return self.operands[0]

    @property
    def src(self) -> Value:
        return self.operands[1]

    @property
    def indices(self) -> list[Value]: # TODO: Does MoveOp need indices?
        return self.operands[2:]

    @property
    def collective(self) -> dict[CollectiveAxis, CollectiveLevel]:
        return self.attrs["collective"]  # type: ignore[return-value]

    @property
    def multicast(self) -> bool:
        return self.attrs.get("multicast", False)  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key not in ("collective", "multicast")
        }

    def print(self, p: Printer) -> None:
        p.write("move(")
        for i, operand in enumerate(self.operands):
            if i:
                p.write(", ")
            operand.print(p)
        p.write(")")
        _print_collective(p, self.collective)
        p.write_extra_attrs(self.attrs, skip_keys=("collective",))

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        assert not results
        p.read_literal("(")
        operands = [Value.parse(p)]
        while p.can_read_literal(","):
            p.read_literal(",")
            operands.append(Value.parse(p))
        p.read_literal(")")
        collective = _parse_collective(p)
        attrs = p.read_extra_attrs()
        multicast = attrs.pop("multicast", False)
        return cls.build(
            operands[0], operands[1], operands[2:],
            collective=collective,
            multicast=multicast,  # type: ignore[arg-type]
            attrs=attrs,
        )


@gpu_dialect.register_op("tma_move")
class TmaMoveOp(GpuOp):
    """Synchronous TMA bulk transfer with explicit collectivity.

    One operand is a TMA descriptor (gmem side), the other is smem. Indices are
    per-dim element coordinates for the TMA engine. Direction (load/store) is
    determined by which side is smem vs tma_desc. Completion tracking is the
    surrounding ``async { ... }`` wrapper's job (it owns the lock), so this op
    carries no lock itself — mirroring ``move`` and ``matmul``.
    """

    @classmethod
    def build(
        cls,
        dst: Value,
        src: Value,
        *,
        indices: list[Value],
        collective: dict[CollectiveAxis, CollectiveLevel],
        attrs: dict[str, object] | None = None,
        l2_cache_hint: str | None = None,
        multicast: bool = False,
    ) -> "TmaMoveOp":
        operands = [dst, src] + indices
        attrs = dict(attrs or {})
        if "multicast" in attrs:
            raise ValueError("pass multicast as an explicit argument")
        attrs["collective"] = dict(collective)
        if l2_cache_hint is not None:
            attrs["l2_cache_hint"] = l2_cache_hint
        if multicast is not False:
            attrs["multicast"] = multicast
        return cls(operands=operands, results=[], attrs=attrs)

    @property
    def dst(self) -> Value:
        return self.operands[0]

    @property
    def src(self) -> Value:
        return self.operands[1]

    @property
    def indices(self) -> list[Value]:
        return self.operands[2:]

    @property
    def l2_cache_hint(self) -> str | None:
        hint = self.attrs.get("l2_cache_hint")
        if hint not in (None, "evict_first", "evict_last"):
            raise ValueError(f"unsupported TMA L2 cache hint {hint!r}")
        return hint  # type: ignore[return-value]

    @property
    def multicast(self) -> bool:
        return self.attrs.get("multicast", False)  # type: ignore[return-value]

    @property
    def collective(self) -> dict[CollectiveAxis, CollectiveLevel]:
        return self.attrs["collective"]  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key not in ("collective", "l2_cache_hint", "multicast")
        }

    def print(self, p: Printer) -> None:
        p.write("tma_move(")
        for i, operand in enumerate(self.operands):
            if i:
                p.write(", ")
            operand.print(p)
        p.write(")")
        _print_collective(p, self.collective)
        p.write_extra_attrs(self.attrs, skip_keys=("collective",))

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        assert not results
        p.read_literal("(")
        operands = [Value.parse(p)]
        while p.can_read_literal(","):
            p.read_literal(",")
            operands.append(Value.parse(p))
        p.read_literal(")")
        collective = _parse_collective(p)
        attrs = p.read_extra_attrs()
        l2_cache_hint = attrs.pop("l2_cache_hint", None)
        multicast = attrs.pop("multicast", False)
        return cls.build(
            operands[0], operands[1], indices=operands[2:],
            collective=collective, attrs=attrs,
            l2_cache_hint=l2_cache_hint,  # type: ignore[arg-type]
            multicast=multicast,  # type: ignore[arg-type]
        )


@gpu_dialect.register_type("lock")
class LockType(Type):
    """Shared synchronization handle for async operations.

    Allocated once, passed to multiple async ops that register work on it,
    then passed to ``await`` to drain all registered work.  Maps to an
    mbarrier (TMA) or the wgmma commit-group counter at the hardware level.

    The ``kind`` attribute encodes the hardware mechanism:
      - ``"mbar"``: mbarrier-based (lock value = mbarrier smem address)
      - ``"cluster_mbar"``: cluster-to-leader rendezvous mbarrier in shared
        memory (lock value = the rank-0 CTA's mbarrier address)
      - ``"wgmma"``: wgmma commit-group counter (lock value unused)
      - ``"bulk"``: bulk async commit-group (TMA stores)
      - ``None``: unspecified (set by lower_to_async before PTX lowering)
    """

    @classmethod
    def build(cls, kind: str | None = None, attrs: dict[str, object] | None = None) -> "LockType":
        return cls(attrs={"kind": kind, **(attrs or {})})

    @property
    def kind(self) -> str | None:
        kind = self.attrs.get("kind")
        assert kind is None or isinstance(kind, str)
        return kind

    def print(self, p: Printer) -> None:
        p.write("lock<")
        p.write(self.kind if self.kind is not None else "?")
        p.write_extra_attrs(self.attrs, skip_keys=("kind",))
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p) -> "LockType":
        assert prefix == "lock"
        p.read_literal("<")
        if p.can_read_literal("?"):
            p.read_literal("?")
            kind = None
        else:
            kind = p.read_identifier()
        extra = p.read_extra_attrs()
        p.read_literal(">")
        return cls.build(kind=kind, attrs=extra)



@gpu_dialect.register_op("await")
class AwaitOp(GpuOp):
    """Wait for all async work registered on a lock."""

    @classmethod
    def build(cls, lock: Value, attrs: dict[str, object] | None = None) -> "AwaitOp":
        assert isinstance(lock.type, LockType), (
            f"AwaitOp: expected LockType; got {lock.type}"
        )
        return cls(operands=[lock], results=[], attrs=attrs or {})

    @property
    def lock(self) -> Value:
        return self.operands[0]

    @lock.setter
    def lock(self, value: Value) -> None:
        self.operands[0] = value


@gpu_dialect.register_op("barrier")
class BarrierOp(GpuOp):
    """A rendezvous / synchronization point across a set of threads.

    It only establishes a happens-before across its participating threads.
    Most hardware-backed scopes need no operand. A ``cluster_to_leader`` barrier
    carries its cluster mbarrier, while a software grid barrier carries its
    zero-initialized global-memory
    synchronization buffer. Kernels using a grid barrier must be scheduled so
    all participating blocks can be resident concurrently.

    The ``scope`` attribute records the participant set and direction, one of:
    ``thread``, ``warp``, ``warpgroup``, ``block_leader``, ``block``,
    ``cluster_to_leader``, ``leader_to_cluster``, ``cluster``, ``grid``.
    """

    @classmethod
    def build(
        cls,
        scope: str = "block",
        sync: Value | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "BarrierOp":
        return cls(
            operands=[] if sync is None else [sync],
            results=[], attrs={"scope": scope, **(attrs or {})},
        )

    @property
    def scope(self) -> str:
        return self.attrs["scope"]  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "scope"
        }

    # TODO: Replace explicit synchronization operands once their scratch
    # allocations have an owner.
    @property
    def sync(self) -> Value | None:
        return self.operands[0] if self.operands else None


@gpu_dialect.register_op("if")
class IfOp(GpuOp):
    """Execute a region when its predicate is true."""

    @classmethod
    def build(
        cls, condition: Value, body: Block,
    ) -> "IfOp":
        assert isinstance(condition.type, PredicateType), (
            f"IfOp: expected PredicateType; got {condition.type}"
        )
        return cls(
            operands=[condition], results=[],
            blocks=[body],
        )

    @property
    def condition(self) -> Value:
        return self.operands[0]

    @property
    def body(self) -> Block:
        return self.blocks[0]

    def print(self, p: Printer) -> None:
        p.write("if(")
        self.condition.print(p)
        p.write(") ")
        self.body.print(p, skip_label=True)

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        assert not results
        p.read_literal("(")
        condition = Value.parse(p)
        p.read_literal(")")
        brace = p.read_literal("{")
        body = Block.parse(brace, p, arguments=[])
        return cls.build(condition, body)


@gpu_dialect.register_op("async")
class AsyncOp(GpuOp):
    """A group of async work issued together and completed by one ``await``.

    Wraps a body block of ops that run asynchronously under one ``lock``. The
    lock is the only operand — what an async op carries beyond the plain op(s)
    it wraps — while the work ops inside keep their own (matmul/move) operands.
    Completion is signalled elsewhere by ``await(%lock)``. Its collective
    records the scope of the work it groups.
    """

    @classmethod
    def build(
        cls,
        lock: Value,
        collective: dict[CollectiveAxis, CollectiveLevel],
        body: Block | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "AsyncOp":
        assert isinstance(lock.type, LockType), (
            f"AsyncOp: expected LockType; got {lock.type}"
        )
        if body is None:
            body = Block()
        elif body.arguments:
            raise ValueError(
                f"async: body block takes no arguments, got "
                f"{len(body.arguments)}"
            )
        attrs = dict(attrs or {})
        attrs["collective"] = dict(collective)
        return cls(operands=[lock], results=[], attrs=attrs, blocks=[body])

    @property
    def lock(self) -> Value:
        return self.operands[0]

    @lock.setter
    def lock(self, value: Value) -> None:
        self.operands[0] = value

    @property
    def body(self) -> Block:
        return self.blocks[0]

    @property
    def collective(self) -> dict[CollectiveAxis, CollectiveLevel]:
        return self.attrs["collective"]  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "collective"
        }

    @property
    def is_tma_multicast(self) -> bool:
        return any(
            isinstance(op, (MoveOp, TmaMoveOp)) and op.multicast is True
            for op in self.body.ops
        )

    def print(self, p: Printer) -> None:
        p.write(f"{self.op_id}(")
        self.lock.print(p)
        p.write(")")
        _print_collective(p, self.collective)
        p.write_extra_attrs(self.attrs, skip_keys=("collective",))
        p.write(" ")
        if len(self.body.ops) == 1:
            # Sugar: a single-op body prints inline, without braces.
            self.body.ops[0].print(p)
        else:
            self.body.print(p, skip_label=True)

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("(")
        lock = Value.parse(p)
        p.read_literal(")")
        collective = _parse_collective(p)
        attrs = p.read_extra_attrs()
        if p.can_read_literal("{"):
            brace = p.read_literal("{")
            body = Block.parse(brace, p, arguments=[])
        else:
            # Sugar: a single op stands in for a one-op body block.
            body = Block(ops=[Op.parse(p)])
        return cls.build(
            lock, collective=collective, body=body, attrs=attrs,
        )


@gpu_dialect.register_op("ptx_template")
class PTXTemplateOp(GpuOp):
    @classmethod
    def build(
        cls,
        operands: list[Value],
        result_types: list[Type],
        template: str,
        operand_names: list[str] | None = None,
        result_names: list[str] | None = None,
        description: str | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "PTXTemplateOp":
        results = [Value(name=None, type=t) for t in result_types]
        a: dict[str, object] = {"template": template, **(attrs or {})}
        if operand_names is not None:
            a["operand_names"] = operand_names
        if result_names is not None:
            a["result_names"] = result_names
        if description is not None:
            a["description"] = description
        return cls(operands=operands, results=results, attrs=a)

    @property
    def template(self) -> str:
        return self.attrs["template"]  # type: ignore[return-value]

    @property
    def operand_names(self) -> list[str] | None:
        return self.attrs.get("operand_names")  # type: ignore[return-value]

    @property
    def result_names(self) -> list[str] | None:
        return self.attrs.get("result_names")  # type: ignore[return-value]

    @property
    def description(self) -> str | None:
        return self.attrs.get("description")  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {k: v for k, v in self.attrs.items()
                if k not in ("template", "operand_names", "result_names", "description")}


class MemTensorType(Type):
    """Common base for all tensor-typed memory: shape + dtype.

    Subclasses split along the addressing model:
      - AddrMemTensorType: byte-addressable linear memory (gmem, smem).
      - RegMemTensorType: SIMT-distributed registers (rmem, wrmem, wgrmem).

    All subclasses encode as `<NAME><N x M x ... x DTYPE [extra] attrs={...}>`,
    differing only in the `[extra]` clauses they print/parse (strides for
    AddrMemTensorType; layout/swizzle live in the generic attrs dict).
    """

    @classmethod
    def build(
        cls,
        sizes: Sequence[int | None],
        dtype: Type,
        attrs: dict[str, object] | None = None,
    ):
        return cls(attrs={
            "sizes": list(sizes),
            "dtype": dtype,
            **(attrs or {}),
        })

    @property
    def sizes(self) -> tuple[int | None, ...]:
        return self.attrs["sizes"]  # type: ignore[return-value]

    @sizes.setter
    def sizes(
        self, value: list[int | None] | tuple[int | None, ...],
    ) -> None:
        self.attrs["sizes"] = list(value)

    @property
    def rank(self) -> int:
        return len(self.sizes)

    @property
    def num_elements(self) -> int | None:
        sizes = self.sizes
        if None in sizes:
            return None
        return math.prod(cast(tuple[int, ...], sizes))

    @property
    def is_dynamic(self) -> bool:
        return any(size is None for size in self.sizes)

    @property
    def dtype(self) -> Type:
        return self.attrs["dtype"]  # type: ignore[return-value]

    def print(self, p: Printer) -> None:
        p.write(self.name)
        p.write("<")
        p.write_shape(self.sizes, self.dtype)
        p.write_extra_attrs(self.attrs, skip_keys=("sizes", "dtype"))
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p):  # type: ignore[override]
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("<")
        sizes, dtype = p.read_shape_with_dtype()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        return cls(attrs={
            "sizes": sizes,
            "dtype": dtype,
            **attrs,
        })


class AddrMemTensorType(MemTensorType):
    """Byte-addressable tensor: lives in linear memory with byte strides.

    Pointer arithmetic is meaningful here; tensors can alias by sharing
    address ranges; offset arithmetic uses byte strides.
    """

    @staticmethod
    def canonical_strides(
        sizes: Sequence[int | None],
    ) -> list[int | None]:
        """Row-major dense strides (in elements)."""
        strides: list[int | None] = []
        running: int | None = 1
        for s in reversed(sizes):
            strides.insert(0, running)
            running = None if running is None or s is None else running * s
        return strides

    @property
    def is_canonical_strides(self) -> bool:
        return list(self.strides) == self.canonical_strides(list(self.sizes))

    @classmethod
    def build(  # type: ignore[override]
        cls,
        sizes: Sequence[int | None],
        dtype: Type,
        strides: Sequence[int | None] | None = None,
        attrs: dict[str, object] | None = None,
    ):
        if strides is None:
            strides = cls.canonical_strides(sizes)
        return cls(attrs={
            "sizes": list(sizes),
            "dtype": dtype,
            "strides": list(strides),
            **(attrs or {}),
        })

    @property
    def strides(self) -> tuple[int | None, ...]:
        """Per-dim stride in elements."""
        return tuple(self.attrs["strides"])  # type: ignore[arg-type]

    def write_strides(self, p: Printer) -> None:
        if not self.is_canonical_strides:
            p.write(" strides=")
            p.write_list_attr(list(self.strides))

    @staticmethod
    def read_strides(
        sizes: Sequence[int | None], p,
    ) -> list[int | None]:
        strides = list(AddrMemTensorType.canonical_strides(sizes))
        if p.can_read_literal("strides"):
            p.read_literal("strides")
            p.read_literal("=")
            strides = p.read_list_attr()
        return strides

    def print(self, p: Printer) -> None:
        p.write(self.name)
        p.write("<")
        p.write_shape(self.sizes, self.dtype)
        self.write_strides(p)
        p.write_extra_attrs(
            self.attrs, skip_keys=("sizes", "dtype", "strides")
        )
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p):  # type: ignore[override]
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("<")
        sizes, dtype = p.read_shape_with_dtype()
        strides = cls.read_strides(sizes, p)
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        return cls(attrs={
            "sizes": sizes,
            "dtype": dtype,
            "strides": strides,
            **attrs,
        })


class RegMemTensorType(MemTensorType):
    """Register-resident tensor: SIMT-distributed registers, no byte
    addressing. Storage is a fixed set of named registers per thread
    (rmem), warp (wrmem), or warp group (wgrmem); no pointer arithmetic,
    no aliasing.
    """


@gpu_dialect.register_type("gmem")
class GmemTensorType(AddrMemTensorType):
    """Global-memory tensor."""


@gpu_dialect.register_type("smem")
class SmemTensorType(AddrMemTensorType):
    """Shared-memory tensor (per-block)."""

    @classmethod
    def build(  # type: ignore[override]
        cls,
        sizes: Sequence[int],
        dtype: Type,
        strides: Sequence[int] | None = None,
        swizzle: str | None = None,
        attrs: dict[str, object] | None = None,
    ):
        if strides is None:
            canonical_strides = cls.canonical_strides(sizes)
            assert None not in canonical_strides
            strides = cast(list[int], canonical_strides)
        return cls(attrs={
            "sizes": list(sizes),
            "dtype": dtype,
            "strides": list(strides),
            "swizzle": swizzle,
            **(attrs or {}),
        })

    @property
    def swizzle(self) -> str | None:
        swizzle = self.attrs["swizzle"]
        assert swizzle is None or isinstance(swizzle, str)
        return swizzle

    def print(self, p: Printer) -> None:
        p.write(self.name)
        p.write("<")
        p.write_shape(self.sizes, self.dtype)
        self.write_strides(p)
        if self.swizzle is not None:
            p.write(f' swizzle="{self.swizzle}"')
        p.write_extra_attrs(
            self.attrs, skip_keys=("sizes", "dtype", "strides", "swizzle"),
        )
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p):  # type: ignore[override]
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("<")
        sizes, dtype = p.read_shape_with_dtype()
        strides = cls.read_strides(sizes, p)
        swizzle: str | None = None
        if p.can_read_literal("swizzle"):
            p.read_literal("swizzle")
            p.read_literal("=")
            swizzle = p.read_str_attr()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        assert None not in sizes
        assert None not in strides
        return cls.build(
            cast(list[int], sizes),
            dtype,
            cast(list[int], strides),
            swizzle,
            attrs=attrs,
        )


@gpu_dialect.register_type("ptr")
class PtrType(Type):
    """Raw pointer into an address space (gmem or smem).

    Carries no shape or stride information — just an address.
    gmem pointers are 64-bit, smem pointers are 32-bit.
    """

    @classmethod
    def build(cls, space: str) -> "PtrType":
        assert space in ("gmem", "smem"), (
            f"PtrType: space must be 'gmem' or 'smem'; got {space!r}"
        )
        return cls(attrs={"space": space})

    @property
    def space(self) -> str:
        space = self.attrs["space"]
        assert isinstance(space, str)
        return space

    def print(self, p: Printer) -> None:
        p.write(f"ptr<{self.space}")
        p.write_extra_attrs(self.attrs, skip_keys=("space",))
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p) -> "PtrType":
        assert prefix == "ptr"
        p.read_literal("<")
        space = p.read_identifier()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        return cls(attrs={"space": space, **attrs})


@gpu_dialect.register_op("to_ptr")
class ToPtrOp(GpuOp):
    """Convert an addressable memory tensor to a raw pointer.

    Takes the base address of the tensor.
    """

    @classmethod
    def build(cls, arg: Value) -> "ToPtrOp":
        assert isinstance(arg.type, AddrMemTensorType), (
            f"ToPtrOp: expected addressable memory tensor; got {arg.type}"
        )
        if isinstance(arg.type, GmemTensorType):
            ptr_type = PtrType.build("gmem")
        else:
            ptr_type = PtrType.build("smem")
        result = Value(name=None, type=ptr_type)
        return cls(operands=[arg], results=[result])

    @property
    def arg(self) -> Value:
        return self.operands[0]

    def expr_str(self, prec=0, val_str=None) -> str:
        arg = _resolve_val_str(self.arg, 0, val_str)
        return f"to_ptr({arg})"


@gpu_dialect.register_op("ptr_add")
class PtrAddOp(GpuOp):
    """Add a byte offset to a pointer, returning a new pointer."""

    @classmethod
    def build(cls, ptr: Value, byte_offset: Value) -> "PtrAddOp":
        assert isinstance(ptr.type, PtrType), (
            f"PtrAddOp: expected PtrType; got {ptr.type}"
        )
        result = Value(name=None, type=ptr.type)
        return cls(operands=[ptr, byte_offset], results=[result])

    @property
    def ptr(self) -> Value:
        return self.operands[0]

    @property
    def byte_offset(self) -> Value:
        return self.operands[1]

    def expr_str(self, prec=0, val_str=None) -> str:
        p = _resolve_val_str(self.ptr, 1, val_str)
        off = _resolve_val_str(self.byte_offset, 1, val_str)
        s = f"{p} + {off}"
        if prec > 1:
            s = f"({s})"
        return s


@gpu_dialect.register_type("tma_desc")
class TmaDescType(Type):
    """TMA tensor map descriptor for bulk async copies (Hopper+).

    Wraps a `GmemTensorType` (global tensor shape/strides/dtype) and
    adds TMA-specific config: tile (box) shape, swizzle mode, and
    which kernel param (src) provides the base pointer at launch time.
    """

    @classmethod
    def build(
        cls,
        arg_idx: int,
        gmem_type: "GmemTensorType",
        tile_shape: list[int],
        swizzle: str | None = None,
        l2_promotion: str | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "TmaDescType":
        attrs = dict(attrs or {})
        if "l2_promotion" in attrs:
            raise ValueError("pass l2_promotion as an explicit argument")
        if l2_promotion not in (None, "64B", "128B", "256B"):
            raise ValueError(f"unsupported TMA L2 promotion {l2_promotion!r}")
        if l2_promotion is not None:
            attrs["l2_promotion"] = l2_promotion
        return cls(attrs={
            "arg_idx": arg_idx,
            "gmem_type": gmem_type,
            "tile_shape": list(tile_shape),
            "swizzle": swizzle,
            **attrs,
        })

    @property
    def arg_idx(self) -> int:
        arg_idx = self.attrs["arg_idx"]
        assert isinstance(arg_idx, int)
        return arg_idx

    @property
    def gmem_type(self) -> "GmemTensorType":
        gmem_type = self.attrs["gmem_type"]
        assert isinstance(gmem_type, GmemTensorType)
        return gmem_type

    @property
    def dtype(self) -> Type:
        return self.gmem_type.dtype

    @property
    def tile_sizes(self) -> tuple[int, ...]:
        tile_shape = self.attrs["tile_shape"]
        assert isinstance(tile_shape, list)
        assert all(isinstance(size, int) for size in tile_shape)
        return tuple(tile_shape)

    @property
    def tile_num_elements(self) -> int:
        return math.prod(self.tile_sizes)

    @property
    def l2_promotion(self) -> str | None:
        return self.attrs.get("l2_promotion")  # type: ignore[return-value]

    @property
    def swizzle(self) -> str | None:
        swizzle = self.attrs["swizzle"]
        assert swizzle is None or isinstance(swizzle, str)
        return swizzle

    def print(self, p: Printer) -> None:
        p.write(f"{self.name}<")
        p.write(f"arg_idx={self.arg_idx} ")
        self.gmem_type.print(p)
        p.write(" tile=")
        p.write_shape(self.tile_sizes)
        if self.swizzle is not None:
            p.write(f' swizzle="{self.swizzle}"')
        p.write_extra_attrs(
            self.attrs,
            skip_keys=("arg_idx", "gmem_type", "tile_shape", "swizzle"),
        )
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p) -> "TmaDescType":
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("<")
        p.read_literal("arg_idx")
        p.read_literal("=")
        arg_idx = p.read_int()
        gmem_prefix = p.read_identifier()
        gmem_type = GmemTensorType.parse(gmem_prefix, p)
        p.read_literal("tile")
        p.read_literal("=")
        tile_shape = p.read_shape()
        swizzle: str | None = None
        if p.can_read_literal("swizzle"):
            p.read_literal("swizzle")
            p.read_literal("=")
            swizzle = p.read_str_attr()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        l2_promotion = attrs.pop("l2_promotion", None)
        assert None not in tile_shape
        assert l2_promotion is None or isinstance(l2_promotion, str)
        return cls.build(
            arg_idx,
            gmem_type,
            cast(list[int], tile_shape),
            swizzle=swizzle,
            l2_promotion=l2_promotion,
            attrs=attrs,
        )


def physical_layout(
    tensor_type: SmemTensorType | TmaDescType,
) -> tuple[int, str] | None:
    """Return validated ``(block_width, order)`` layout metadata."""
    layout = tensor_type.attrs.get("physical_layout")
    if layout is None:
        return None
    assert isinstance(layout, dict)
    block_width = layout.get("block_width")
    order = layout.get("order", "row_major")
    assert isinstance(block_width, int) and block_width > 0
    assert isinstance(order, str) and order in ("row_major", "block_major")
    return block_width, order


@gpu_dialect.register_type("rmem")
class RmemTensorType(RegMemTensorType):
    """Register-memory tensor (per-thread).

    Each thread independently holds the full tile in its own registers.
    There is no inter-lane sharing, so no layout policy is needed.
    """


@gpu_dialect.register_type("wrmem")
class WrmemTensorType(RegMemTensorType):
    """Register-memory tensor collectively owned by one warp.

    Each lane holds the registers assigned to it by the producing operation;
    the logical value is the union of the 32 lane fragments.
    ``default`` is a placeholder layout whose mapping is not yet specified.
    """

    # TODO: Move warp-level MMA support from WgrmemTensorType to WrmemTensorType,
    # including the mma layout, bf16/f16 register mapping, matmul and elementwise
    # lowering, and memory effects.
    # TODO: Define the supported layout mappings and validate layout names.

    @classmethod
    def build(  # type: ignore[override]
        cls,
        sizes: list[int],
        dtype: Type,
        layout: str = "default",
        attrs: dict[str, object] | None = None,
    ):
        return cls(attrs={
            "sizes": list(sizes),
            "dtype": dtype,
            "layout": layout,
            **(attrs or {}),
        })

    @property
    def layout(self) -> str:
        return self.attrs["layout"]  # type: ignore[return-value]

    def print(self, p: Printer) -> None:
        p.write(self.name)
        p.write("<")
        p.write_shape(self.sizes, self.dtype)
        p.write(f' layout="{self.layout}"')
        p.write_extra_attrs(
            self.attrs, skip_keys=("sizes", "dtype", "layout"),
        )
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p):  # type: ignore[override]
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("<")
        sizes, dtype = p.read_shape_with_dtype()
        p.read_literal("layout")
        p.read_literal("=")
        layout = p.read_str_attr()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        return cls.build(sizes, dtype, layout, attrs=attrs)


@gpu_dialect.register_type("wgrmem")
class WgrmemTensorType(RegMemTensorType):
    """Warp-group register-memory tensor.

    Logical tensor distributed across multiple warps. Each thread holds
    a fragment in its own registers; the full value is the union of all
    threads' fragments. `warps` specifies how many warps collectively
    own the value (1 for Volta mma.sync, 4 for Hopper wgmma, 2 for
    Blackwell tcgen05).

    The ``layout`` attribute encodes the element-to-thread-register mapping.
    The mapping is determined by the instruction that writes the tensor.
    Supported layouts:
      - ``"mma"``: mma.sync m16n8k16, warps=1
      - ``"wgmma_layout_0"``: warp group MMA m64nNk16, warps=4
    All lowerings that read from a wgrmem tensor (store, convert, matmul)
    must check the layout to emit correct code.
    """

    @classmethod
    def build(  # type: ignore[override]
        cls,
        sizes: list[int],
        dtype: Type,
        layout: str,
        warps: int = 1,
        attrs: dict[str, object] | None = None,
    ):
        assert layout in ("mma", "wgmma_layout_0"), (
            f"WgrmemTensorType: layout must be 'mma' or 'wgmma_layout_0'; "
            f"got {layout!r}"
        )
        return cls(attrs={
            "sizes": list(sizes),
            "dtype": dtype,
            "warps": warps,
            "layout": layout,
            **(attrs or {}),
        })

    @property
    def warps(self) -> int:
        return self.attrs["warps"]  # type: ignore[return-value]

    @property
    def layout(self) -> str:
        return self.attrs["layout"]  # type: ignore[return-value]

    @property
    def num_elements_per_thread(self) -> int:
        num_elements = self.num_elements
        assert num_elements is not None
        return num_elements // (self.warps * 32)

    def print(self, p: Printer) -> None:
        p.write(self.name)
        p.write("<")
        p.write_shape(self.sizes, self.dtype)
        if self.warps != 1:
            p.write(f" warps={self.warps}")
        p.write(f" layout=\"{self.layout}\"")
        p.write_extra_attrs(
            self.attrs, skip_keys=("sizes", "dtype", "warps", "layout"),
        )
        p.write(">")

    @classmethod
    def parse(cls, prefix: str, p):  # type: ignore[override]
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("<")
        sizes, dtype = p.read_shape_with_dtype()
        warps = 1
        if p.can_read_literal("warps"):
            p.read_literal("warps")
            p.read_literal("=")
            warps = p.read_literal_attr()
        p.read_literal("layout")
        p.read_literal("=")
        layout = p.read_str_attr()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        return cls.build(sizes, dtype, layout, warps=warps, attrs=attrs)


@gpu_dialect.register_op("for")
class ForOp(GpuOp):
    """Sequential loop over a 1D range: for (iv = lb to ub [step s])."""

    @classmethod
    def build(
        cls,
        lb: int,
        ub: int,
        step: int = 1,
        body: Block | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "ForOp":
        if body is None:
            iv = Value(name=None, type=IndexType())
            body = Block(arguments=[iv])
        else:
            args = body.arguments
            if len(args) != 1:
                raise ValueError(
                    f"for: body block must have 1 argument, got {len(args)}"
                )
            if not isinstance(args[0].type, IndexType):
                raise ValueError(
                    f"for: body block argument must have IndexType, "
                    f"got {type(args[0].type).__name__}"
                )
        return cls(
            operands=[],
            results=[],
            attrs={
                "lb": lb,
                "ub": ub,
                "step": step,
                **(attrs or {}),
            },
            blocks=[body],
        )

    @property
    def lb(self) -> int:
        return self.attrs["lb"]  # type: ignore[return-value]

    @lb.setter
    def lb(self, value: int) -> None:
        self.attrs["lb"] = value

    @property
    def ub(self) -> int:
        return self.attrs["ub"]  # type: ignore[return-value]

    @ub.setter
    def ub(self, value: int) -> None:
        self.attrs["ub"] = value

    @property
    def step(self) -> int:
        return self.attrs["step"]  # type: ignore[return-value]

    @step.setter
    def step(self, value: int) -> None:
        self.attrs["step"] = value

    @property
    def trip_count(self) -> int:
        return max(0, (self.ub - self.lb + self.step - 1) // self.step)

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {k: v for k, v in self.attrs.items() if k not in ("lb", "ub", "step")}

    @property
    def body(self) -> Block:
        return self.blocks[0]

    @property
    def iv(self) -> Value:
        return self.body.arguments[0]

    def print(self, p: Printer) -> None:
        p.write(f"{self.op_id} (")
        self.iv.print(p)
        p.write(f" = {self.lb} to {self.ub}")
        if self.step != 1:
            p.write(f" step {self.step}")
        p.write(")")
        p.write_extra_attrs(
            self.attrs, skip_keys=("lb", "ub", "step"),
        )
        p.write(" ")
        self.body.print(p, skip_label=True)

    @classmethod
    def parse_infix(cls, results, prefix: str, p):  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("(")
        iv = Value.parse(p)
        p.read_literal("=")
        lb = p.read_literal_attr()
        p.read_literal("to")
        ub = p.read_literal_attr()
        step = 1
        if p.can_read_literal("step"):
            p.read_literal("step")
            step = p.read_literal_attr()
        assert isinstance(lb, int)
        assert isinstance(ub, int)
        assert isinstance(step, int)
        p.read_literal(")")
        attrs = p.read_extra_attrs()
        brace = p.read_literal("{")
        body = Block.parse(brace, p, arguments=[iv])
        return cls.build(lb, ub, step, body=body, attrs=attrs)


@gpu_dialect.register_type("index")
class IndexType(Type):
    pass


@gpu_dialect.register_type("pred")
class PredicateType(Type):
    pass


@gpu_dialect.register_type("f16")
class F16Type(Type):
    byte_size = 2


@gpu_dialect.register_type("bf16")
class BF16Type(Type):
    byte_size = 2


@gpu_dialect.register_type("f32")
class F32Type(Type):
    byte_size = 4


@gpu_dialect.register_type("f64")
class F64Type(Type):
    byte_size = 8


@gpu_dialect.register_op("constant")
class ConstantOp(GpuOp):
    """Produce an immutable floating-point tensor constant.

    A scalar payload is a splat; a list payload is flat row-major data. A
    rank-0 tensor is the canonical scalar data value. Unlike
    ``IndexConstantOp``, this operation represents numerical program data
    rather than an index used for addressing or control flow.
    """

    @classmethod
    def build(
        cls,
        value: int | float | list[int | float],
        result_type: MemTensorType,
    ) -> "ConstantOp":
        return cls(
            operands=[],
            results=[Value(name=None, type=result_type)],
            attrs={"value": value},
        )

    @property
    def value(self) -> int | float | list[int | float]:
        return cast(int | float | list[int | float], self.attrs["value"])

    @staticmethod
    def splat_value(value: Value) -> int | float | None:
        """The literal of the splat constant defining ``value``, if any."""
        if not isinstance(value.definition, ConstantOp):
            return None
        literal = value.definition.value
        return None if isinstance(literal, list) else literal

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "value"
        }


@gpu_dialect.register_op("thread_id")
class ThreadIdOp(GpuOp):
    """Produce one axis of the thread index."""

    @classmethod
    def build(
        cls,
        axis: CollectiveAxis,
        attrs: dict[str, object] | None = None,
    ) -> "ThreadIdOp":
        attrs = dict(attrs or {})
        if "axis" in attrs:
            raise ValueError("pass axis as an explicit argument")
        axis_name = axis.name.lower()
        return cls(
            operands=[],
            results=[Value(name=f"t{axis_name}", type=IndexType())],
            attrs={"axis": axis_name, **attrs},
        )

    @property
    def axis(self) -> CollectiveAxis:
        axis = self.attrs["axis"]
        assert isinstance(axis, str)
        return CollectiveAxis[axis.upper()]

    @property
    def value(self) -> Value:
        return self.results[0]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "axis"
        }

    def print(self, p: Printer) -> None:
        self.value.print(p)
        p.write(f" = {self.op_id}.{self.axis.name.lower()}")
        p.write_extra_attrs(self.attrs, skip_keys=("axis",))

    @classmethod
    def parse_infix(
        cls, results: list[Value], prefix: str, p,
    ) -> "ThreadIdOp":  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal(".")
        axis_name = p.read_identifier()
        try:
            axis = CollectiveAxis[axis_name.upper()]
        except KeyError as error:
            raise ValueError(f"invalid thread ID axis {axis_name!r}") from error
        attrs = p.read_extra_attrs()
        if "axis" in attrs:
            raise ValueError("thread ID axis must not also appear in attrs")
        return cls(
            operands=[],
            results=results,
            attrs={"axis": axis.name.lower(), **attrs},
        )


@gpu_dialect.register_op("block_id")
class BlockIdOp(GpuOp):
    """Produce one axis of the block index."""

    @classmethod
    def build(
        cls,
        axis: CollectiveAxis,
        attrs: dict[str, object] | None = None,
    ) -> "BlockIdOp":
        attrs = dict(attrs or {})
        if "axis" in attrs:
            raise ValueError("pass axis as an explicit argument")
        axis_name = axis.name.lower()
        return cls(
            operands=[],
            results=[Value(name=f"b{axis_name}", type=IndexType())],
            attrs={"axis": axis_name, **attrs},
        )

    @property
    def axis(self) -> CollectiveAxis:
        axis = self.attrs["axis"]
        assert isinstance(axis, str)
        return CollectiveAxis[axis.upper()]

    @property
    def value(self) -> Value:
        return self.results[0]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "axis"
        }

    def print(self, p: Printer) -> None:
        self.value.print(p)
        p.write(f" = {self.op_id}.{self.axis.name.lower()}")
        p.write_extra_attrs(self.attrs, skip_keys=("axis",))

    @classmethod
    def parse_infix(
        cls, results: list[Value], prefix: str, p,
    ) -> "BlockIdOp":  # type: ignore[override]
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal(".")
        axis_name = p.read_identifier()
        try:
            axis = CollectiveAxis[axis_name.upper()]
        except KeyError as error:
            raise ValueError(f"invalid block ID axis {axis_name!r}") from error
        attrs = p.read_extra_attrs()
        if "axis" in attrs:
            raise ValueError("block ID axis must not also appear in attrs")
        return cls(
            operands=[],
            results=results,
            attrs={"axis": axis.name.lower(), **attrs},
        )


@gpu_dialect.register_op("cluster_block_ids")
class ClusterBlockIdsOp(GpuOp):
    """Produce the 3D CTA index within the enclosing cluster."""

    @classmethod
    def build(cls) -> "ClusterBlockIdsOp":
        idx = IndexType()
        return cls(
            operands=[],
            results=[
                Value(name=name, type=idx)
                for name in ["cbx", "cby", "cbz"]
            ],
        )

    @property
    def x(self) -> Value:
        return self.results[0]

    @property
    def y(self) -> Value:
        return self.results[1]

    @property
    def z(self) -> Value:
        return self.results[2]


@gpu_dialect.register_op("warp_group_id")
class WarpGroupIdOp(GpuOp):
    """Produce the 3D warp group index within the block."""

    @classmethod
    def build(cls) -> "WarpGroupIdOp":
        idx = IndexType()
        return cls(
            operands=[],
            results=[Value(name=name, type=idx) for name in ["wgx", "wgy", "wgz"]],
        )

    @property
    def x(self) -> Value:
        return self.results[0]

    @property
    def y(self) -> Value:
        return self.results[1]

    @property
    def z(self) -> Value:
        return self.results[2]


@gpu_dialect.register_op("index_constant")
class IndexConstantOp(GpuOp):
    """Produce a constant index value."""

    @classmethod
    def build(cls, value: int) -> "IndexConstantOp":
        result = Value(name=f"c{value}", type=IndexType())
        return cls(operands=[], results=[result], attrs={"value": value})

    @property
    def value(self) -> int:
        value = self.attrs["value"]
        assert isinstance(value, int)
        return value

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {
            key: value for key, value in self.attrs.items()
            if key != "value"
        }

    def expr_str(self, prec=0, val_str=None) -> str:
        return str(self.value)


@gpu_dialect.register_op("index_lt")
class IndexLtOp(GpuOp):
    """Compare two index values using unsigned less-than."""

    @classmethod
    def build(cls, lhs: Value, rhs: Value) -> "IndexLtOp":
        assert isinstance(lhs.type, IndexType), (
            f"IndexLtOp: expected IndexType lhs; got {lhs.type}"
        )
        assert isinstance(rhs.type, IndexType), (
            f"IndexLtOp: expected IndexType rhs; got {rhs.type}"
        )
        return cls(
            operands=[lhs, rhs],
            results=[Value(name=None, type=PredicateType())],
        )

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]


def _resolve_val_str(v, prec, val_str):
    """Resolve a value to a string. Try val_str first, then expr_str on the definition."""
    if hasattr(v, "definition") and hasattr(v.definition, "expr_str"):
        return v.definition.expr_str(prec, val_str)
    if val_str is not None:
        s = val_str(v, prec)
        if s is not None:
            return s
    return "?"


def _binop_expr_str(op, operand, prec, parent_prec, val_str):
    """Helper for binary index op expr_str with precedence."""
    lhs = _resolve_val_str(op.lhs, prec, val_str)
    rhs = _resolve_val_str(op.rhs, prec + 1, val_str)
    s = f"{lhs} {operand} {rhs}"
    if parent_prec > prec:
        s = f"({s})"
    return s


@gpu_dialect.register_op("index_add")
class IndexAddOp(GpuOp):
    """Add two index values: result = lhs + rhs."""

    @classmethod
    def build(cls, lhs: Value, rhs: Value) -> "IndexAddOp":
        result = Value(name=None, type=IndexType())
        return cls(operands=[lhs, rhs], results=[result])

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]

    def expr_str(self, prec=0, val_str=None) -> str:
        return _binop_expr_str(self, "+", 1, prec, val_str)


@gpu_dialect.register_op("index_mul")
class IndexMulOp(GpuOp):
    """Multiply two index values: result = lhs * rhs."""

    @classmethod
    def build(cls, lhs: Value, rhs: Value) -> "IndexMulOp":
        result = Value(name=None, type=IndexType())
        return cls(operands=[lhs, rhs], results=[result])

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]

    def expr_str(self, prec=0, val_str=None) -> str:
        return _binop_expr_str(self, "*", 3, prec, val_str)


@gpu_dialect.register_op("index_div")
class IndexDivOp(GpuOp):
    """Unsigned integer division: result = lhs / rhs."""

    @classmethod
    def build(cls, lhs: Value, rhs: Value) -> "IndexDivOp":
        result = Value(name=None, type=IndexType())
        return cls(operands=[lhs, rhs], results=[result])

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]

    def expr_str(self, prec=0, val_str=None) -> str:
        return _binop_expr_str(self, "/", 2, prec, val_str)


@gpu_dialect.register_op("index_mod")
class IndexModOp(GpuOp):
    """Unsigned integer remainder: result = lhs % rhs."""

    @classmethod
    def build(cls, lhs: Value, rhs: Value) -> "IndexModOp":
        result = Value(name=None, type=IndexType())
        return cls(operands=[lhs, rhs], results=[result])

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]

    def expr_str(self, prec=0, val_str=None) -> str:
        return _binop_expr_str(self, "%", 2, prec, val_str)


@gpu_dialect.register_op("select")
class SelectOp(GpuOp):
    """Pick one of several values by index: ``result = cases[index]``.

    Introduced to name a choice the IR cannot yet make statically -- notably
    "which of N buffers does this loop iteration use". Folding collapses it as
    soon as the index is known, which is the only form the PTX backend can
    lower: there is no runtime selection between SSA values.
    """

    @classmethod
    def build(cls, index: Value, cases: list[Value]) -> "SelectOp":
        assert cases, "select needs at least one case"
        assert all(case.type == cases[0].type for case in cases), (
            f"select: cases disagree on type, got {[c.type for c in cases]}"
        )
        result = Value(name=None, type=cases[0].type)
        return cls(operands=[index, *cases], results=[result])

    @property
    def index(self) -> Value:
        return self.operands[0]

    @property
    def cases(self) -> list[Value]:
        return self.operands[1:]


def root_buffers(value: Value) -> set[Value]:
    """Return every physical buffer that may underlie ``value``."""
    definition = value.definition
    if isinstance(definition, ReinterpretOp):
        return root_buffers(definition.arg)
    if isinstance(definition, (ReshapeOp, TileOp)):
        return root_buffers(definition.arg)
    if isinstance(definition, SelectOp):
        return {
            root
            for case in definition.cases
            for root in root_buffers(case)
        }
    if isinstance(definition, ToPtrOp):
        return root_buffers(definition.arg)
    if isinstance(definition, PtrAddOp):
        return root_buffers(definition.ptr)
    if isinstance(value.type, (MemTensorType, TmaDescType)):
        return {value}
    return set()

def root_buffer(value: Value) -> Value | None:
    """Return the unique physical buffer underlying ``value``, if any."""
    roots = root_buffers(value)
    assert len(roots) <= 1, (
        f"root_buffer: {value} has multiple root buffers: {roots}"
    )
    return next(iter(roots), None)


class CollectiveScope(enum.IntEnum):
    r"""The granularity at which an effect is completely independent of concurrent
    operations. A partial order, not a chain. ``BLOCK_LEADER`` is work at block
    scope but performed exclusively by thread 0, e.g. a TMA.
    ``CLUSTER_LEADER`` is the corresponding single issuer acting for a cluster,
    so ``BLOCK_LEADER <= CLUSTER_LEADER``. They coincide operationally for a
    single-CTA cluster but remain distinct lattice elements. ``BLOCK`` (all
    threads in one CTA) and ``CLUSTER_LEADER`` (one issuer for the cluster) are
    incomparable::

             GRID
              |
           CLUSTER
           /       \
        BLOCK   CLUSTER_LEADER
        /   \       |
 WARPGROUP   BLOCK_LEADER
        |       |
      WARP      |
         \     /
          THREAD
    """
    THREAD = 0
    WARP = 1
    WARPGROUP = 2
    BLOCK_LEADER = 3
    BLOCK = 4
    CLUSTER_LEADER = 5
    CLUSTER = 6
    GRID = 7

    def join(self, other: CollectiveScope) -> CollectiveScope:
        """The join (least upper bound) on the CollectiveScope lattice."""
        common_upper_bounds = [
            scope for scope in CollectiveScope
            if self.leq(scope) and other.leq(scope)
        ]
        least_upper_bounds = [
            candidate for candidate in common_upper_bounds
            if all(candidate.leq(scope) for scope in common_upper_bounds)
        ]
        assert len(least_upper_bounds) == 1, (
            f"CollectiveScope has no unique join for {self!r} and {other!r}"
        )
        return least_upper_bounds[0]

    def leq(self, other: CollectiveScope) -> bool:
        """The partial order (<=) of the CollectiveScope lattice."""
        return self == other or any(
            upper.leq(other)
            for upper in _COLLECTIVE_SCOPE_IMMEDIATE_UPPER_BOUNDS[self]
        )


_COLLECTIVE_SCOPE_IMMEDIATE_UPPER_BOUNDS = {
    CollectiveScope.THREAD: (
        CollectiveScope.WARP,
        CollectiveScope.BLOCK_LEADER,
    ),
    CollectiveScope.WARP: (CollectiveScope.WARPGROUP,),
    CollectiveScope.WARPGROUP: (CollectiveScope.BLOCK,),
    CollectiveScope.BLOCK_LEADER: (
        CollectiveScope.BLOCK,
        CollectiveScope.CLUSTER_LEADER,
    ),
    CollectiveScope.BLOCK: (CollectiveScope.CLUSTER,),
    CollectiveScope.CLUSTER_LEADER: (CollectiveScope.CLUSTER,),
    CollectiveScope.CLUSTER: (CollectiveScope.GRID,),
    CollectiveScope.GRID: (),
}


def _type_collective_scope(t: Type) -> CollectiveScope:
    if isinstance(t, RmemTensorType):
        return CollectiveScope.THREAD
    if isinstance(t, WrmemTensorType):
        return CollectiveScope.WARP
    if isinstance(t, WgrmemTensorType):
        return CollectiveScope.WARPGROUP
    if isinstance(t, SmemTensorType):
        return CollectiveScope.BLOCK
    if isinstance(t, GmemTensorType):
        return CollectiveScope.GRID
    raise NotImplementedError(
        f"_type_collective_scope: unmodeled buffer type {type(t).__name__}"
    )


@dataclass(frozen=True)
class MemoryEffect:
    """A memory access before its view is resolved to physical roots."""

    kind: Literal["R", "W"]
    value: Value
    scope: CollectiveScope
    tiling_scheme: int | None = None

    _tiling_scheme_id_counter: ClassVar[Iterator[int]] = itertools.count()

    @classmethod
    def fresh_tiling_scheme_id(cls) -> int:
        return next(cls._tiling_scheme_id_counter)

    @property
    def roots(self) -> set[Value]:
        return root_buffers(self.value)


def _memory_effect(
    kind: Literal["R", "W"],
    value: Value,
    scope: CollectiveScope,
    tiling_scheme: int | None = None,
) -> list[MemoryEffect]:
    return [MemoryEffect(kind, value, scope, tiling_scheme)]


def _resolve_tma_descriptor(op: TmaMoveOp, value: Value) -> Value:
    if not isinstance(value.type, TmaDescType):
        return value
    kernel = find_ancestor_of_type(op, KernelOp)
    if kernel is None:
        return value
    return kernel.body.arguments[value.type.arg_idx]


def memory_effects(op: Op) -> list[MemoryEffect] | None:
    """Best-effort memory effect summary for ``op``.

    Each effect retains the accessed view so consumers can recover its precise
    region. ``effect.roots`` provides its underlying allocations/arguments for
    conservative alias checks. A non-null ``tiling_scheme`` identifies a family
    of accesses known to be pairwise disjoint.

    Returns ``None`` for ops we don't know how to model; callers must treat that
    as a worst-case conflict.
    """
    if is_side_effect_free(op):
        return []
    if isinstance(op, MoveOp):
        # A TMA move is issued by t0 of its block, or by t0 of cluster rank 0
        # for multicast. A cooperative move writes at the buffer's own scope.
        src_scope = (
            CollectiveScope.THREAD
            if isinstance(op.src.type, WgrmemTensorType)
            else _type_collective_scope(op.src.type)
        )
        # TODO: Derive the issuing scope from ``op.collective`` once collective
        # levels can express leader-only execution. For now, the destination
        # memory space distinguishes a TMA load from a store, while ``multicast``
        # distinguishes the cluster leader from an ordinary block leader.
        if op.multicast is True:
            dst_scope = CollectiveScope.CLUSTER_LEADER
        elif op.attrs.get("tma") and isinstance(op.dst.type, SmemTensorType):
            dst_scope = CollectiveScope.BLOCK_LEADER
        elif isinstance(op.dst.type, WgrmemTensorType):
            dst_scope = CollectiveScope.THREAD
        else:
            dst_scope = _type_collective_scope(op.dst.type)
        return (
            _memory_effect(
                "W", op.dst, dst_scope,
                cast(int | None, op.attrs.get("dst_tiling_scheme")),
            )
            + _memory_effect(
                "R", op.src, src_scope,
                cast(int | None, op.attrs.get("src_tiling_scheme")),
            )
        )
    if isinstance(op, TmaMoveOp):
        # A block leader issues ordinary TMA, while the cluster leader issues
        # multicast. Global memory is shared across the grid. The descriptor is
        # metadata, so resolve it to the underlying kernel argument.
        dst_scope = (
            CollectiveScope.CLUSTER_LEADER
            if op.multicast is True
            else CollectiveScope.BLOCK_LEADER
            if isinstance(op.dst.type, SmemTensorType)
            else CollectiveScope.GRID
        )
        src_scope = (
            CollectiveScope.BLOCK_LEADER
            if isinstance(op.src.type, SmemTensorType)
            else CollectiveScope.GRID
        )
        return (
            _memory_effect(
                "W", _resolve_tma_descriptor(op, op.dst), dst_scope,
                cast(int | None, op.attrs.get("dst_tiling_scheme")),
            )
            + _memory_effect(
                "R", _resolve_tma_descriptor(op, op.src), src_scope,
                cast(int | None, op.attrs.get("src_tiling_scheme")),
            )
        )
    if isinstance(op, MatmulOp):
        # The wgmma accumulator (dst) is per-thread registers -> THREAD.
        init_effects = [] if op.init is None else _memory_effect(
            "R", op.init, _type_collective_scope(op.init.type),
        )
        return (
            _memory_effect(
                "R", op.lhs, _type_collective_scope(op.lhs.type),
                cast(int | None, op.attrs.get("lhs_tiling_scheme")),
            )
            + _memory_effect(
                "R", op.rhs, _type_collective_scope(op.rhs.type),
                cast(int | None, op.attrs.get("rhs_tiling_scheme")),
            )
            + init_effects
            + _memory_effect(
                "R", op.dst, CollectiveScope.THREAD,
                cast(int | None, op.attrs.get("dst_tiling_scheme")),
            )
            + _memory_effect(
                "W", op.dst, CollectiveScope.THREAD,
                cast(int | None, op.attrs.get("dst_tiling_scheme")),
            )
        )
    if isinstance(op, ElementwiseOp):
        effects = _memory_effect("W", op.dst, CollectiveScope.THREAD)
        for i in range(op.num_srcs):
            effects += _memory_effect(
                "R", op.src(i), CollectiveScope.THREAD,
            )
        return effects
    if isinstance(op, ReduceOp):
        # Register accumulators are independently read and updated by each
        # thread; the source determines whether the reduction is collective.
        dst_scope = (
            CollectiveScope.THREAD
            if isinstance(
                op.dst.type,
                (RmemTensorType, WrmemTensorType, WgrmemTensorType),
            )
            else _type_collective_scope(op.dst.type)
        )
        effects = (
            [] if op.init is not None else _memory_effect("R", op.dst, dst_scope)
        )
        return (
            effects
            + _memory_effect("W", op.dst, dst_scope)
            + _memory_effect(
                "R", op.src, _type_collective_scope(op.src.type),
            )
        )
    if isinstance(op, AwaitOp):
        # An await's effects are the union of its async peers' — the async starts
        # sharing its lock — found directly via the lock's use-list, no scan.
        peers = [
            u for u in op.lock.users() if isinstance(u, (AsyncOp, TmaMoveOp))
        ]
        if not peers:
            return None
        effects = []
        for peer in peers:
            peer_effects = memory_effects(peer)
            if peer_effects is None:
                return None
            effects.extend(peer_effects)
        return effects
    if isinstance(op, (ForOp, IfOp, AsyncOp, KernelOp)):
        effects: list[MemoryEffect] = []
        for block in op.blocks:
            for body_op in block.ops:
                body_effects = memory_effects(body_op)
                if body_effects is None:
                    return None
                effects.extend(body_effects)
        return effects
    return None


def has_ordering_effect(op: Op) -> bool:
    """Whether ``op`` or a nested operation constrains operation ordering."""
    if isinstance(op, BarrierOp):
        return True
    return any(
        has_ordering_effect(nested)
        for block in op.blocks
        for nested in block.ops
    )


def is_side_effect_free(op: Op) -> bool:
    """Safe to move, and interchangeable with an identical op.

    Same operands and attrs always produce the same result, and nothing is read
    or written along the way.
    """
    return isinstance(op, (
        AllocOp,
        BlockIdOp,
        ClusterBlockIdsOp,
        IndexAddOp,
        IndexConstantOp,
        IndexDivOp,
        IndexLtOp,
        IndexModOp,
        IndexMulOp,
        PtrAddOp,
        ConstantOp,
        SelectOp,
        ReinterpretOp,
        ReshapeOp,
        ThreadIdOp,
        TileOp,
        ToPtrOp,
        WarpGroupIdOp,
    ))


def is_elementwise(op: Op) -> bool:
    """Applies one scalar function to each element, independently.

    Element i of the result depends on element i of the input and nothing else,
    so the op carries no cross-element dependency: it can be split along any
    axis, fused with another of its kind, or applied to a tile in place.
    """
    return isinstance(op, ElementwiseOp)


def effective_operands(op: Op) -> list[Value]:
    """An op's explicit operands plus the values its blocks capture.

    A block-bearing op (e.g. ``async``, ``for``) references SSA values defined
    in the enclosing scope from inside its body. Those captures are real data
    dependencies but are not in ``op.operands`` (the body refers to them
    directly). This returns ``op.operands`` followed by each captured value —
    used somewhere in the blocks, defined outside them — so dependency/SSA
    analysis sees an op's full read-set whether a value is an explicit operand
    or a block capture. For an op with no blocks it is just ``op.operands``.

    Single program-order walk leaning on SSA: a value used before it is defined
    within ``op`` must come from the enclosing scope, so it is a capture. For
    each op we record any operands not yet defined, recurse into its blocks
    (whose arguments — and, in SSA order, earlier siblings' results — are
    already defined), then mark its results defined for later siblings.
    """
    defined: set[Value] = set()
    effective: list[Value] = []
    effective_set: set[Value] = set()

    def visit(o: Op) -> None:
        for v in o.operands:
            if v not in defined and v not in effective_set:
                effective_set.add(v)
                effective.append(v)
        for block in o.blocks:
            defined.update(block.arguments)
            for inner in block.ops:
                visit(inner)
        defined.update(o.results)

    visit(op)
    return effective
