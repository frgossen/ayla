"""Concrete IR ops and types for the Ayla dialect."""
from __future__ import annotations

import enum
import itertools
import math
from typing import ClassVar, Iterator

from .base.ir import Block, Dialect, Op, Type, Value
from .base.printer import Printer

gpu_dialect = Dialect("gpu")


@gpu_dialect.register_op("module")
class ModuleOp(Op):
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
        return self.attrs["version"]

    @property
    def target(self) -> str:
        return self.attrs["target"]

    @property
    def address_size(self) -> int:
        return self.attrs["address_size"]

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
class KernelOp(Op):
    @classmethod
    def build(
        cls,
        name: str,
        body: Block,
        grid_dim: tuple[int | None, int | None, int | None] = (None, None, None),
        block_dim: tuple[int | None, int | None, int | None] = (None, None, None),
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
    def num_warp_groups(self) -> int | None:
        return self.attrs["num_warp_groups"]  # type: ignore[return-value]

    @num_warp_groups.setter
    def num_warp_groups(self, value: int | None) -> None:
        self.attrs["num_warp_groups"] = value

    @property
    def ptx_optimization_level(self) -> int | None:
        return self.attrs.get("ptx_optimization_level")  # type: ignore[return-value]

    @property
    def num_threads(self) -> int:
        """Number of threads in the resolved block dimensions."""
        return math.prod(self.block_dim)  # type: ignore[arg-type]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {k: v for k, v in self.attrs.items() if k not in ("name", "grid_dim", "block_dim", "num_warp_groups")}

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
        if self.num_warp_groups is not None:
            p.write(f" num_warp_groups={self.num_warp_groups}")
        p.write_extra_attrs(
            self.attrs, skip_keys={"name", "grid_dim", "block_dim", "num_warp_groups"},
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
        if p.can_read_literal("grid_dim"):
            p.read_literal("grid_dim")
            p.read_literal("=")
            grid_dim = tuple(p.read_list_attr())
        if p.can_read_literal("block_dim"):
            p.read_literal("block_dim")
            p.read_literal("=")
            block_dim = tuple(p.read_list_attr())
        num_warp_groups = None
        if p.can_read_literal("num_warp_groups"):
            p.read_literal("num_warp_groups")
            p.read_literal("=")
            num_warp_groups = p.read_int()
        attrs = p.read_extra_attrs()
        brace = p.read_literal("{")
        body = Block.parse(brace, p, arguments=args)
        ptx_optimization_level = attrs.pop("ptx_optimization_level", None)
        return cls.build(
            name,
            body,
            grid_dim=grid_dim,
            block_dim=block_dim,
            num_warp_groups=num_warp_groups,
            ptx_optimization_level=ptx_optimization_level,
            attrs=attrs,
        )


@gpu_dialect.register_op("matmul")
class MatmulOp(Op):
    @classmethod
    def build(
        cls,
        lhs: Value,
        rhs: Value,
        dst: Value,
        attrs: dict[str, object] | None = None,
        init: float | int | None = None,
    ) -> "MatmulOp":
        a: dict[str, object] = dict(attrs or {})
        if init is not None:
            a["init"] = init
        return cls(
            operands=[lhs, rhs, dst], results=[], attrs=a,
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
    def init(self) -> float | int | None:
        return self.attrs.get("init")  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {k: v for k, v in self.attrs.items() if k != "init"}



@gpu_dialect.register_op("tile")
class TileOp(Op):
    """Select a sub-region (view) of a buffer.

    ``maybe_partial`` marks a tile whose logical extent may cross the valid
    bounds of its source for some index values. Consumers must then provide
    defined out-of-bounds behavior.

    Invariant: distinct tile results of the same buffer are
    non-conflicting — they are either disjoint (safe for concurrent
    writes) or overlapping read-only. This is the responsibility of
    whoever constructs the tiles. It allows memory effect analysis to
    use the tile Value directly as the buffer identity without
    resolving to the root alloc.
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

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        assert len(self.results) == 1, "TileOp must produce exactly one result"
        # A tile is a view into its source — strides are inherited.
        arg_type = self.arg.type
        res_type = self.result(0).type
        if isinstance(arg_type, WrmemTensorType):
            assert isinstance(res_type, WrmemTensorType), (
                "TileOp: wrmem arg requires wrmem result"
            )
        else:
            assert (
                isinstance(arg_type, AddrMemTensorType)
                and isinstance(res_type, AddrMemTensorType)
            ), "TileOp: arg and result must be AddrMemTensorType or WrmemTensorType"
            assert list(arg_type.strides) == list(res_type.strides), (
                f"TileOp: result strides {list(res_type.strides)} must "
                f"match arg strides {list(arg_type.strides)}"
            )

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


@gpu_dialect.register_op("convert")
class ConvertOp(Op):
    """Element-wise dtype conversion: convert(dst, src).

    Reads from src, converts each element, writes to dst.
    Both must have the same shape and memory space but different dtypes.
    """

    @classmethod
    def build(
        cls,
        dst: Value,
        src: Value,
        attrs: dict[str, object] | None = None,
    ) -> "ConvertOp":
        return cls(operands=[dst, src], results=[], attrs=attrs or {})

    @property
    def dst(self) -> Value:
        return self.operands[0]

    @property
    def src(self) -> Value:
        return self.operands[1]


@gpu_dialect.register_op("elementwise")
class ElementwiseOp(Op):
    """One scalar function over each element: elementwise(dst, src).

    ``fn`` names the function -- an attribute, not an op each, since exp,
    tanh, relu and the rest differ only in the instructions they lower to.

    Passing the same value as both operands computes in place.

    TODO: generalize to several sources and destinations. One of each admits
    only unary functions -- add, mul and select need more than one input, and
    a lowering may want to produce more than one value at a time.
    """

    FUNCTIONS: ClassVar[tuple[str, ...]] = ("sigmoid",)

    @classmethod
    def build(
        cls,
        dst: Value,
        src: Value,
        fn: str,
        attrs: dict[str, object] | None = None,
    ) -> "ElementwiseOp":
        if fn not in cls.FUNCTIONS:
            raise ValueError(
                f"elementwise: unknown fn {fn!r}; expected one of "
                f"{', '.join(cls.FUNCTIONS)}"
            )
        return cls(
            operands=[dst, src], results=[], attrs={"fn": fn, **(attrs or {})},
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


@gpu_dialect.register_op("alloc")
class AllocOp(Op):
    """Allocate a register- or shared-memory tensor.

    Optional `init` attr fills the allocation with a constant value.

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
        init: float | int | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "AllocOp":
        a: dict[str, object] = dict(attrs or {})
        a["id"] = id
        if init is not None:
            a["init"] = init
        result = Value(name=None, type=type)
        return cls(operands=[], results=[result], attrs=a)

    @classmethod
    def build_with_fresh_id(
        cls,
        type: Type,
        init: float | int | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "AllocOp":
        """Build an allocation that is nobody else's buffer."""
        return cls.build(type, id=cls.fresh_id(), init=init, attrs=attrs)

    @property
    def init(self) -> float | int | None:
        return self.attrs.get("init")  # type: ignore[return-value]

    @property
    def extra_attrs(self) -> dict[str, object]:
        return {k: v for k, v in self.attrs.items() if k != "init"}


@gpu_dialect.register_op("move")
class MoveOp(Op):
    """Move data into a destination, with optional indices."""

    @classmethod
    def build(
        cls,
        dst: Value,
        src: Value,
        indices: list[Value] | None = None,
        attrs: dict[str, object] | None = None,
    ) -> "MoveOp":
        operands = [dst, src] + (indices or [])
        return cls(
            operands=operands, results=[], attrs=attrs or {},
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


@gpu_dialect.register_op("tma_move")
class TmaMoveOp(Op):
    """TMA bulk transfer between smem and gmem (block-collective), sync form.

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
        indices: list[Value],
        attrs: dict[str, object] | None = None,
        l2_cache_hint: str | None = None,
    ) -> "TmaMoveOp":
        operands = [dst, src] + indices
        attrs = dict(attrs or {})
        if l2_cache_hint is not None:
            attrs["l2_cache_hint"] = l2_cache_hint
        return cls(operands=operands, results=[], attrs=attrs)

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        dst_is_tma = isinstance(self.dst.type, TmaDescType)
        src_is_tma = isinstance(self.src.type, TmaDescType)
        assert dst_is_tma != src_is_tma, (
            f"TmaMoveOp: exactly one operand must be a TMA descriptor; "
            f"got dst={type(self.dst.type).__name__}, "
            f"src={type(self.src.type).__name__}"
        )

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


@gpu_dialect.register_type("lock")
class LockType(Type):
    """Shared synchronization handle for async operations.

    Allocated once, passed to multiple async ops that register work on it,
    then passed to ``await`` to drain all registered work.  Maps to an
    mbarrier (TMA) or the wgmma commit-group counter at the hardware level.

    The ``kind`` attribute encodes the hardware mechanism:
      - ``"mbar"``: mbarrier-based (lock value = mbarrier smem address)
      - ``"wgmma"``: wgmma commit-group counter (lock value unused)
      - ``"bulk"``: bulk async commit-group (TMA stores)
      - ``None``: unspecified (set by lower_to_async before PTX lowering)
    """

    @classmethod
    def build(cls, kind: str | None = None, attrs: dict[str, object] | None = None) -> "LockType":
        return cls(attrs={"kind": kind, **(attrs or {})})

    @property
    def kind(self) -> str | None:
        return self.attrs.get("kind")

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
class AwaitOp(Op):
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
class BarrierOp(Op):
    """A rendezvous / synchronization point across a set of threads.

    Carries no data and has no lock — it only establishes a happens-before
    across its participating threads.

    The ``scope`` attribute records the participant set the sync spans, one of:
    ``thread``, ``warp``, ``warpgroup``, ``block_leader``, ``block``, ``grid``.
    """

    @classmethod
    def build(
        cls,
        scope: str = "block",
        attrs: dict[str, object] | None = None,
    ) -> "BarrierOp":
        return cls(operands=[], results=[], attrs={"scope": scope, **(attrs or {})})

    @property
    def scope(self) -> str:
        return self.attrs["scope"]  # type: ignore[return-value]


@gpu_dialect.register_op("async")
class AsyncOp(Op):
    """A group of async work issued together and completed by one ``await``.

    Wraps a body block of ops that run asynchronously under one ``lock``. The
    lock is the only operand — what an async op carries beyond the plain op(s)
    it wraps — while the work ops inside keep their own (matmul/move) operands.
    Completion is signalled elsewhere by ``await(%lock)``.
    """

    @classmethod
    def build(
        cls,
        lock: Value,
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
        return cls(operands=[lock], results=[], attrs=attrs or {}, blocks=[body])

    @property
    def lock(self) -> Value:
        return self.operands[0]

    @lock.setter
    def lock(self, value: Value) -> None:
        self.operands[0] = value

    @property
    def body(self) -> Block:
        return self.blocks[0]

    def print(self, p: Printer) -> None:
        p.write(f"{self.op_id}(")
        self.lock.print(p)
        p.write(")")
        p.write_extra_attrs(self.attrs)
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
        attrs = p.read_extra_attrs()
        if p.can_read_literal("{"):
            brace = p.read_literal("{")
            body = Block.parse(brace, p, arguments=[])
        else:
            # Sugar: a single op stands in for a one-op body block.
            body = Block(ops=[Op.parse(p)])
        return cls.build(lock, body=body, attrs=attrs)


@gpu_dialect.register_op("ptx_template")
class PTXTemplateOp(Op):
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
      - RegMemTensorType: SIMT-distributed registers (rmem, wrmem).

    All subclasses encode as `<NAME><N x M x ... x DTYPE [extra] attrs={...}>`,
    differing only in the `[extra]` clauses they print/parse (strides for
    AddrMemTensorType; layout/swizzle live in the generic attrs dict).
    """

    @classmethod
    def build(
        cls,
        sizes: list[int | None],
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
        if self.is_dynamic:
            return None
        return math.prod(self.sizes)

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
        sizes, dtype = p.read_shape(with_dtype=True)
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
        sizes: list[int | None],
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
        sizes: list[int | None],
        dtype: Type,
        strides: list[int | None] | None = None,
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
    def read_strides(sizes: list[int | None], p) -> list[int | None]:
        strides = AddrMemTensorType.canonical_strides(sizes)
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
        sizes, dtype = p.read_shape(with_dtype=True)
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
    (rmem) or per warp (wrmem); no pointer arithmetic, no aliasing.
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
        sizes: list[int],
        dtype: Type,
        strides: list[int] | None = None,
        swizzle: str | None = None,
        attrs: dict[str, object] | None = None,
    ):
        if strides is None:
            strides = cls.canonical_strides(sizes)
        return cls(attrs={
            "sizes": list(sizes),
            "dtype": dtype,
            "strides": list(strides),
            "swizzle": swizzle,
            **(attrs or {}),
        })

    @property
    def swizzle(self) -> str | None:
        return self.attrs["swizzle"]

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
        sizes, dtype = p.read_shape(with_dtype=True)
        strides = cls.read_strides(sizes, p)
        swizzle: str | None = None
        if p.can_read_literal("swizzle"):
            p.read_literal("swizzle")
            p.read_literal("=")
            swizzle = p.read_str_attr()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        return cls.build(sizes, dtype, strides, swizzle, attrs=attrs)


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
        return self.attrs["space"]

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
class ToPtrOp(Op):
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
class PtrAddOp(Op):
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
        return self.attrs["arg_idx"]

    @property
    def gmem_type(self) -> "GmemTensorType":
        return self.attrs["gmem_type"]

    @property
    def dtype(self) -> Type:
        return self.gmem_type.dtype

    @property
    def tile_sizes(self) -> tuple[int, ...]:
        return tuple(self.attrs["tile_shape"])

    @property
    def tile_num_elements(self) -> int:
        return math.prod(self.tile_sizes)

    @property
    def l2_promotion(self) -> str | None:
        return self.attrs.get("l2_promotion")  # type: ignore[return-value]

    @property
    def swizzle(self) -> str | None:
        return self.attrs["swizzle"]

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
        tile_shape = p.read_shape(with_dtype=False)
        swizzle: str | None = None
        if p.can_read_literal("swizzle"):
            p.read_literal("swizzle")
            p.read_literal("=")
            swizzle = p.read_str_attr()
        attrs = p.read_extra_attrs()
        p.read_literal(">")
        l2_promotion = attrs.pop("l2_promotion", None)
        return cls.build(
            arg_idx,
            gmem_type,
            tile_shape,
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
class WrmemTensorType(RegMemTensorType):# TODO: rename wrmem to wgrmem
    """Warp register-memory tensor.

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
    All lowerings that read from a wrmem tensor (store, convert, matmul)
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
            f"WrmemTensorType: layout must be 'mma' or 'wgmma_layout_0'; "
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
        return self.num_elements // (self.warps * 32)

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
        sizes, dtype = p.read_shape(with_dtype=True)
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
class ForOp(Op):
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

    @property
    def ub(self) -> int:
        return self.attrs["ub"]  # type: ignore[return-value]

    @property
    def step(self) -> int:
        return self.attrs["step"]  # type: ignore[return-value]

    @property
    def trip_count(self) -> int:
        return (self.ub - self.lb) // self.step

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
        p.read_literal(")")
        attrs = p.read_extra_attrs()
        brace = p.read_literal("{")
        body = Block.parse(brace, p, arguments=[iv])
        return cls.build(lb, ub, step, body=body, attrs=attrs)


@gpu_dialect.register_type("index")
class IndexType(Type):
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


@gpu_dialect.register_op("thread_ids")
class ThreadIdsOp(Op):
    """Produce the 3D thread index (tid.x, tid.y, tid.z)."""

    @classmethod
    def build(cls) -> "ThreadIdsOp":
        idx = IndexType()
        return cls(
            operands=[],
            results=[Value(name=name, type=idx) for name in ["tx", "ty", "tz"]],
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


@gpu_dialect.register_op("block_ids")
class BlockIdsOp(Op):
    """Produce the 3D block index (ctaid.x, ctaid.y, ctaid.z)."""

    @classmethod
    def build(cls) -> "BlockIdsOp":
        idx = IndexType()
        return cls(
            operands=[],
            results=[Value(name=name, type=idx) for name in ["bx", "by", "bz"]],
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
class WarpGroupIdOp(Op):
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
class IndexConstantOp(Op):
    """Produce a constant index value."""

    @classmethod
    def build(cls, value: int) -> "IndexConstantOp":
        result = Value(name=f"c{value}", type=IndexType())
        return cls(operands=[], results=[result], attrs={"value": value})

    @property
    def value(self) -> int:
        return self.attrs["value"]

    def expr_str(self, prec=0, val_str=None) -> str:
        return str(self.value)


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
class IndexAddOp(Op):
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
class IndexMulOp(Op):
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
class IndexDivOp(Op):
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
class IndexModOp(Op):
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
class SelectOp(Op):
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


@gpu_dialect.register_op("lane_id")
class LaneIdOp(Op):
    """Produce the lane index within the warp (0..31)."""

    @classmethod
    def build(cls) -> "LaneIdOp":
        return cls(operands=[], results=[Value(name=None, type=IndexType())])


def root_buffer(value: Value) -> Value:
    """Walk back through TileOps to the underlying SSA root.

    For an alloc / kernel-arg / non-tile value this returns the
    value itself.
    """
    while isinstance(value.definition, TileOp):
        value = value.definition.arg
    if isinstance(value.definition, SelectOp):
        # Returning the select would claim it is a buffer of its own, distinct
        # from the cases it actually aliases -- the unsafe direction, since a
        # caller would read that as "no dependency". A select is meant to be
        # folded away before anything reasons about buffers.
        raise ValueError(
            f"root_buffer: {value.definition.op_id} has no single root buffer; "
            "it aliases every one of its cases. Fold it first."
        )
    return value


class CollectiveScope(enum.IntEnum):
    r"""The granularity at which an effect is completely independent of concurrent
    operations. A partial order, not a chain. ``BLOCK_LEADER`` is work at block
    scope but performed exclusively by thread 0, e.g. a TMA::

             GRID
              |
            BLOCK
            /   \
    WARPGROUP   BLOCK_LEADER
           |     |
         WARP    |
            \   /
            THREAD
    """
    THREAD = 0
    WARP = 1
    WARPGROUP = 2
    BLOCK_LEADER = 3
    BLOCK = 4
    GRID = 5

    def join(self, other: CollectiveScope) -> CollectiveScope:
        """The join (least upper bound) on the CollectiveScope lattice."""
        if self == CollectiveScope.GRID:
            return self
        if self == CollectiveScope.THREAD:
            return other
        if self == CollectiveScope.BLOCK:
            return other if other == CollectiveScope.GRID else self
        if self == CollectiveScope.BLOCK_LEADER:
            if other in {CollectiveScope.THREAD, CollectiveScope.BLOCK_LEADER}:
                return self
            if other in {CollectiveScope.BLOCK, CollectiveScope.GRID}:
                return other
            return CollectiveScope.BLOCK  # other is WARP or WARPGROUP
        if self == CollectiveScope.WARPGROUP:
            if other in {CollectiveScope.BLOCK, CollectiveScope.GRID}:
                return other
            if other == CollectiveScope.BLOCK_LEADER:
                return CollectiveScope.BLOCK
            return self  # other is THREAD, WARP, or WARPGROUP
        if self == CollectiveScope.WARP:
            if other in {CollectiveScope.WARP, CollectiveScope.THREAD}:
                return self
            if other == CollectiveScope.BLOCK_LEADER:
                return CollectiveScope.BLOCK
            return other  # other is WARPGROUP, BLOCK, or GRID
        raise NotImplementedError(
            f"CollectiveScope.join: unhandled pair {self!r}, {other!r}"
        )

    def leq(self, other: CollectiveScope) -> bool:
        """The partial order (<=) of the CollectiveScope lattice."""
        return self.join(other) == other


def _type_collective_scope(t: Type) -> CollectiveScope:
    if isinstance(t, RmemTensorType):
        return CollectiveScope.THREAD
    if isinstance(t, WrmemTensorType):
        return CollectiveScope.WARPGROUP
    if isinstance(t, SmemTensorType):
        return CollectiveScope.BLOCK
    if isinstance(t, GmemTensorType):
        return CollectiveScope.GRID
    raise NotImplementedError(
        f"_type_collective_scope: unmodeled buffer type {type(t).__name__}"
    )


def memory_effects(op: Op) -> list[tuple[str, Value, CollectiveScope]] | None:
    """Best-effort memory effect summary for ``op``.

    Returns a list of ``(kind, root, scope)`` tuples where ``kind`` is ``"R"`` or
    ``"W"``, ``root`` is the underlying alloc/arg (after stripping tile views via
    ``root_buffer``), and ``scope`` is the :class:`CollectiveScope` at which the
    access is independent of concurrent threads.

    Returns ``None`` for ops we don't know how to model; callers must treat that
    as a worst-case conflict.
    """
    if is_side_effect_free(op):
        return []
    if isinstance(op, MoveOp):
        # A TMA move is issued by the leader thread (t0), so its smem write is
        # BLOCK_LEADER; a cooperative move writes at the buffer's own scope.
        src_scope = (
            CollectiveScope.THREAD
            if isinstance(op.src.type, WrmemTensorType)
            else _type_collective_scope(op.src.type)
        )
        dst_scope = (
            CollectiveScope.BLOCK_LEADER
            if op.attrs.get("tma") and isinstance(op.dst.type, SmemTensorType)
            else _type_collective_scope(op.dst.type)
        )
        return [("W", root_buffer(op.dst), dst_scope),
                ("R", root_buffer(op.src), src_scope)]
    if isinstance(op, TmaMoveOp):
        # Leader-issued (t0): the smem side is BLOCK_LEADER (dst on a load, src on
        # a store).
        smem_is_dst = isinstance(op.dst.type, SmemTensorType)
        smem = op.dst if smem_is_dst else op.src
        kind = "W" if smem_is_dst else "R"
        return [(kind, root_buffer(smem), CollectiveScope.BLOCK_LEADER)]
    if isinstance(op, MatmulOp):
        # The wgmma accumulator (dst) is per-thread registers -> THREAD.
        dst = root_buffer(op.dst)
        return [("R", root_buffer(op.lhs), _type_collective_scope(op.lhs.type)),
                ("R", root_buffer(op.rhs), _type_collective_scope(op.rhs.type)),
                ("R", dst, CollectiveScope.THREAD),
                ("W", dst, CollectiveScope.THREAD)]
    if isinstance(op, (ConvertOp, ElementwiseOp)):
        return [("W", root_buffer(op.dst), CollectiveScope.THREAD),
                ("R", root_buffer(op.src), CollectiveScope.THREAD)]
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
    if isinstance(op, AllocOp):
        # Only an initialized alloc writes (fills the buffer); an uninitialized
        # one is side-effect-free, handled above. A wrmem accumulator is per-thread.
        buf = op.result()
        scope = (
            CollectiveScope.THREAD
            if isinstance(buf.type, WrmemTensorType)
            else _type_collective_scope(buf.type)
        )
        return [("W", root_buffer(buf), scope)]
    if isinstance(op, (ForOp, AsyncOp)):
        effects: list[tuple[str, Value, CollectiveScope]] = []
        for block in op.blocks:
            for body_op in block.ops:
                body_effects = memory_effects(body_op)
                if body_effects is None:
                    return None
                effects.extend(body_effects)
        return effects
    return None


def is_side_effect_free(op: Op) -> bool:
    """Safe to move, and interchangeable with an identical op.

    Same operands and attrs always produce the same result, and nothing is read
    or written along the way.
    """
    if isinstance(op, AllocOp):
        # The initialiser writes the buffer, so such an allocation neither
        # moves freely nor stands in for another.
        return op.init is None
    return isinstance(op, (
        BlockIdsOp,
        IndexAddOp,
        IndexConstantOp,
        IndexDivOp,
        IndexModOp,
        IndexMulOp,
        LaneIdOp,
        PtrAddOp,
        SelectOp,
        ThreadIdsOp,
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
    return isinstance(op, (
        ConvertOp,
        ElementwiseOp,
    ))


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
