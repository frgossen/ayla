from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import ClassVar

from .parser import Parser
from .printer import Printer  # noqa: F401


# Observable containers that keep the use-list self-maintaining.
#
# The invariant: ``value.uses`` contains ``(op, i)`` iff ``op.operands[i] is
# value`` and ``op`` is anchored (held by a block, directly or nested inside an
# anchored op's regions). Two list subclasses capture the only two mutation
# channels so no pass registers/deregisters by hand:
#
# - ``_OperandList`` wraps an op's ``operands`` — writes rewire its uses.
# - ``_OpList`` wraps a block's ``ops`` — ops entering/leaving anchor/unanchor
#   (which cascades into their regions).
#
# Both react the same way: snapshot, run the real list method, reconcile via
# ``_sync(old)``. Every mutator is generated from that one pattern below.
# Whole-list reassignment (``op.operands = [...]`` / ``block.ops = [...]``) is
# handled by ``Op``/``Block`` ``__setattr__``, which re-wraps into these types.


class _ObservableList(list):
    """A ``list`` that calls ``self._sync(old)`` after any in-place mutation,
    where ``old`` is the pre-mutation snapshot. Subclasses implement ``_sync``.
    """

    __slots__ = ()

    def _sync(self, old: list) -> None:
        raise NotImplementedError

def _observed(name: str):
    real = getattr(list, name)

    def method(self, *args, **kwargs):
        old = list(self)
        result = real(self, *args, **kwargs)
        self._sync(old)
        return result

    method.__name__ = name
    return method


for _name in (
    "__setitem__", "__delitem__", "__iadd__", "__imul__", "append", "extend",
    "insert", "pop", "remove", "clear", "sort", "reverse",
):
    setattr(_ObservableList, _name, _observed(_name))


class _OperandList(_ObservableList):
    """An op's ``operands``; rewires its values' ``uses`` when anchored."""

    __slots__ = ("_op",)

    def __init__(self, iterable=(), *, op: "Op | None" = None) -> None:
        super().__init__(iterable)
        self._op = op

    def _sync(self, old: list) -> None:
        op = self._op
        if op is None or not op._anchored:
            return
        for i, v in enumerate(old):
            v.uses.discard((op, i))
        for i, v in enumerate(self):
            v.uses.add((op, i))


class _OpList(_ObservableList):
    """A block's ``ops``; anchors/unanchors ops as they enter/leave. Stateless:
    ``_sync`` reconciles purely from the element ops."""

    __slots__ = ()

    def _sync(self, old: list) -> None:
        # Unanchor removed ops before anchoring added ones so transplanted ops
        # (in both old and new) stay anchored.
        new_ids = {id(op) for op in self}
        old_ids = {id(op) for op in old}
        for op in old:
            if id(op) not in new_ids:
                op._unanchor()
        for op in self:
            if id(op) not in old_ids:
                op._anchor()


# Generic IR


@dataclass
class Type:
    attrs: dict[str, object] = field(default_factory=dict)

    # Set by `Dialect.register_type` on each subclass.
    TYPE_ID: ClassVar[str] = "generic"

    @property
    def name(self) -> str:
        return type(self).TYPE_ID

    def attr(self, name: str) -> object:
        return self.attrs[name]

    @classmethod
    def build(cls, attrs: dict[str, object] | None = None):
        """Default builder: just constructs `cls(attrs=...)`. Subclasses
        with extra params (e.g. `MemTensorType` which takes sizes/dtype)
        should override.
        """
        return cls(attrs=dict(attrs or {}))

    def print(self, p: Printer) -> None:
        p.write(self.name)
        if self.attrs:
            p.write("<")
            p.write_dict_attr(self.attrs, braces=False)
            p.write(">")

    @classmethod
    def parse(cls, prefix: str, p: Parser) -> "Type":
        if prefix != cls.TYPE_ID:
            raise ValueError(
                f"expected type {cls.TYPE_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        attrs: dict[str, object] = {}
        if p.is_at("<"):
            p.read_literal("<")
            attrs = p.read_dict_attr(braces=False)
            p.read_literal(">")
        return cls(attrs=attrs)

    def __repr__(self) -> str:
        p = Printer()
        self.print(p)
        return p.render()


@dataclass(eq=False)
class Value:
    name: str | None
    type: Type
    definition: Op | Block = field(init=False)
    result_index: int = field(init=False)
    uses: set[tuple[Op, int]] = field(
        default_factory=set, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        assert self.name is None or not self.name.isdigit(), (
            f"value name {self.name!r} must not be all digits "
            "(reserved for anonymous ids)"
        )

    def users(self) -> list[Op]:
        """
        Ops consuming this value.
        """
        seen: set[int] = set()
        out: list[Op] = []
        for op, _ in self.uses:
            if id(op) not in seen:
                seen.add(id(op))
                out.append(op)
        return out

    def print(self, p: Printer) -> None:
        basename = f"%{self.name}" if self.name is not None else "%"
        name = p.unique_names(
            self, basename, "value", always_suffix=self.name is None,
        )
        p.write(f"{name}: ")
        self.type.print(p)

    @staticmethod
    def parse(p: Parser) -> "Value":
        p.read_literal("%")
        name = p.read_identifier(allow_leading_digits=True)
        p.read_literal(":")
        typ = p.read_type()
        existing = p.values.get(name)
        if existing is not None:
            return existing
        # Numeric-only names are auto-IDs the printer assigned to anonymous
        # values; restore them as None so they round-trip identically.
        actual_name = None if name.isdigit() else name
        v = Value(name=actual_name, type=typ)
        p.values[name] = v
        return v

    def __repr__(self) -> str:
        p = Printer()
        self.print(p)
        return p.render()


@dataclass(eq=False)
class Op:
    operands: list[Value]
    results: list[Value]
    attrs: dict[str, object] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)

    # Set by `Dialect.register_op` on each subclass.
    OP_ID: ClassVar[str] = "generic"

    # Whether this op is currently held by a block (directly, or nested inside
    # an anchored op). Only anchored ops contribute to `Value.uses`. Class-level
    # default so it always resolves, even mid-construction before __post_init__.

    # TODO: anchoring triggers on entering any blocks ops, including free-floating
    # blocks not reachable from the root, so their ops wrongly contribute to Value.uses.
    # region.blocks / op.regions are plain lists, so attachment doesn't propagate anchoring
    # upward. Fix by making those lists observable as well or by collapsing regions into blocks.
    _anchored = False

    def __post_init__(self) -> None:
        self.op_id = type(self).OP_ID
        for i, v in enumerate(self.results):
            v.definition = self
            v.result_index = i

    def __setattr__(self, name: str, value: object) -> None:
        # `operands` is a plain dataclass field, so `op.operands = [...]` would
        # bypass _OperandList. Re-wrap it and run the same reconcile as an
        # in-place mutation.
        if name == "operands":
            old = self.__dict__.get("operands", [])
            new_list = _OperandList(value, op=self)
            super().__setattr__(name, new_list)
            new_list._sync(old)
        else:
            super().__setattr__(name, value)

    def _anchor(self) -> None:
        """Mark this op (and its whole subtree) as held by a block, registering
        operand uses. Idempotent: set-keyed and guarded by ``_anchored``."""
        if self._anchored:
            return
        self._anchored = True
        for i, v in enumerate(self.operands):
            v.uses.add((self, i))
        for region in self.regions:
            for block in region.blocks:
                for op in block.ops:
                    op._anchor()

    def _unanchor(self) -> None:
        """Inverse of ``_anchor``: deregister this op's subtree's operand uses."""
        if not self._anchored:
            return
        self._anchored = False
        for i, v in enumerate(self.operands):
            v.uses.discard((self, i))
        for region in self.regions:
            for block in region.blocks:
                for op in block.ops:
                    op._unanchor()

    def operand(self, i: int = 0) -> Value:
        return self.operands[i]

    def result(self, i: int = 0) -> Value:
        return self.results[i]

    def region(self, i: int = 0) -> Region:
        return self.regions[i]

    def attr(self, name: str) -> object:
        return self.attrs[name]

    @classmethod
    def build(
        cls,
        operands: list["Value"] | None = None,
        result_types: list["Type"] | None = None,
        attrs: dict[str, object] | None = None,
        regions: list["Region"] | None = None,
    ) -> "Op":
        """Default builder. Constructs results from `result_types`.
        Subclasses with richer ergonomic signatures (lhs/rhs/dst/...)
        should override.
        """
        results = [
            Value(name=None, type=t) for t in (result_types or [])
        ]
        return cls(
            operands=list(operands or []),
            results=results,
            attrs=dict(attrs or {}),
            regions=list(regions or []),
        )

    def __iter__(self):
        return iter(self.results)

    def print(self, p: Printer) -> None:
        if self.results:
            for i, v in enumerate(self.results):
                if i:
                    p.write(", ")
                v.print(p)
            p.write(" = ")
        p.write(f"{self.op_id}(")
        for i, v in enumerate(self.operands):
            if i:
                p.write(", ")
            v.print(p)
        p.write(")")
        if self.attrs:
            p.write(" attrs=")
            p.write_attr(self.attrs)
        if self.regions:
            p.write(" ")
        for i, r in enumerate(self.regions):
            if i:
                p.write_newline()
            r.print(p)

    @staticmethod
    def parse(p: Parser) -> "Op":
        """Parse an op with optional result values, dispatching by op name."""
        results: list[Value] = []
        if p.is_at("%"):
            results.append(Value.parse(p))
            while p.can_read_literal(","):
                p.read_literal(",")
                results.append(Value.parse(p))
            p.read_literal("=")
        return p.read_op(results)

    @classmethod
    def parse_infix(cls, results: list[Value], prefix: str, p: Parser) -> "Op":
        if prefix != cls.OP_ID:
            raise ValueError(
                f"expected op {cls.OP_ID!r}, got {prefix!r} at pos {p.pos}"
            )
        p.read_literal("(")
        operands: list[Value] = []
        if not p.can_read_literal(")"):
            operands.append(Value.parse(p))
            while p.can_read_literal(","):
                p.read_literal(",")
                operands.append(Value.parse(p))
        p.read_literal(")")
        attrs: dict[str, object] = {}
        if p.can_read_literal("attrs"):
            p.read_literal("attrs")
            p.read_literal("=")
            attrs = p.read_dict_attr()
        regions: list[Region] = []
        while p.can_read_literal("{"):
            prefix_brace = p.read_literal("{")
            regions.append(Region.parse(prefix_brace, p))
        return cls(
            operands=operands,
            results=results,
            attrs=attrs,
            regions=regions,
        )

    def __repr__(self) -> str:
        p = Printer()
        self.print(p)
        return p.render()


@dataclass(eq=False)
class Block:
    arguments: list[Value] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    name: str | None = None

    def __post_init__(self) -> None:
        assert self.name is None or not self.name.isdigit(), (
            f"block name {self.name!r} must not be all digits "
            "(reserved for anonymous ids)"
        )
        for i, v in enumerate(self.arguments):
            v.definition = self
            v.result_index = i

    def __setattr__(self, name: str, value: object) -> None:
        # `ops` is a plain dataclass field, so `block.ops = [...]` would bypass
        # _OpList. Re-wrap it and run the same reconcile as an in-place mutation.
        if name == "ops":
            old = self.__dict__.get("ops", [])
            new_list = _OpList(value)
            super().__setattr__(name, new_list)
            new_list._sync(old)
        else:
            super().__setattr__(name, value)

    def append(self, op: Op) -> Op:
        self.ops.append(op)
        return op

    def append_argument(self, value: Value) -> None:
        value.definition = self
        value.result_index = len(self.arguments)
        self.arguments.append(value)

    def argument(self, i: int = 0) -> Value:
        return self.arguments[i]

    def print(self, p: Printer, skip_label: bool = False) -> None:
        if not skip_label:
            basename = f"^{self.name}" if self.name is not None else "^bb"
            label = p.unique_names(
                self, basename, "block",
                always_suffix=self.name is None,
            )
            p.write(label)
            if self.arguments:
                p.write("(")
                for i, v in enumerate(self.arguments):
                    if i:
                        p.write(", ")
                    v.print(p)
                p.write(")")
            p.write(":")
            p.write_newline()
        with p.indented():
            for op in self.ops:
                op.print(p)
                p.write_newline()

    @classmethod
    def parse(
        cls,
        prefix: str | None,
        p: Parser,
        arguments: list[Value] | None = None,
    ) -> "Block":
        name: str | None = None
        if arguments is None:
            if prefix != "^":
                raise ValueError(
                    f"expected block prefix '^', got {prefix!r} at pos {p.pos}"
                )
            name = p.read_identifier()
            arguments = []
            if p.can_read_literal("("):
                p.read_literal("(")
                if not p.can_read_literal(")"):
                    arguments.append(Value.parse(p))
                    while p.can_read_literal(","):
                        p.read_literal(",")
                        arguments.append(Value.parse(p))
                p.read_literal(")")
            p.read_literal(":")
        ops: list[Op] = []
        while not (p.can_read_literal("^") or p.can_read_literal("}")):
            ops.append(Op.parse(p))
        return cls(name=name, arguments=arguments, ops=ops)

    def __repr__(self) -> str:
        p = Printer()
        self.print(p)
        return p.render()


@dataclass(eq=False)
class Region:
    blocks: list[Block] = field(default_factory=list)

    def block(self, i: int = 0) -> Block:
        return self.blocks[i]

    @property
    def entry_block(self) -> Block:
        return self.blocks[0]

    def print(
        self, p: Printer, skip_entry_block_label: bool = False,
    ) -> None:
        p.write("{")
        p.write_newline()
        for i, b in enumerate(self.blocks):
            skip_label = i == 0 and skip_entry_block_label
            b.print(p, skip_label=skip_label)
        p.write("}")

    @classmethod
    def parse(
        cls,
        prefix: str,
        p: Parser,
        entry_block_arguments: list[Value] | None = None,
    ) -> "Region":
        if prefix != "{":
            raise ValueError(
                f"expected region prefix '{{', got {prefix!r} at pos {p.pos}"
            )
        blocks: list[Block] = []
        if entry_block_arguments is not None:
            blocks.append(
                Block.parse(None, p, arguments=entry_block_arguments)
            )
        while not p.can_read_literal("}"):
            prefix_caret = p.read_literal("^")
            blocks.append(Block.parse(prefix_caret, p))
        p.read_literal("}")
        return cls(blocks=blocks)

    def __repr__(self) -> str:
        p = Printer()
        self.print(p)
        return p.render()


class Dialect:
    """A named bag of Op and Type subclasses."""

    def __init__(self, name: str) -> None:
        self.name = name
        # The base `generic` Op/Type are always available.
        self._ops: dict[str, type[Op]] = {"generic": Op}
        self._types: dict[str, type[Type]] = {"generic": Type}

    def register_op(self, op_id: str):
        """Class decorator: register an Op subclass under `op_id`."""
        def deco(cls: type[Op]) -> type[Op]:
            if op_id in self._ops:
                raise ValueError(
                    f"{self.name}: op_id {op_id!r} already registered to "
                    f"{self._ops[op_id].__name__}"
                )
            cls.OP_ID = op_id
            self._ops[op_id] = cls
            return cls
        return deco

    def register_type(self, type_id: str):
        """Class decorator: register a Type subclass under `type_id`."""
        def deco(cls: type[Type]) -> type[Type]:
            if type_id in self._types:
                raise ValueError(
                    f"{self.name}: type_id {type_id!r} already registered "
                    f"to {self._types[type_id].__name__}"
                )
            cls.TYPE_ID = type_id
            self._types[type_id] = cls
            return cls
        return deco

    def parse(self, source: str) -> Op:
        p = Parser(
            type_parse_fns={
                tid: cls.parse for tid, cls in self._types.items()
            },
            op_parse_fns={
                oid: cls.parse_infix for oid, cls in self._ops.items()
            },
        )
        p.source = source
        p.pos = 0
        p.values = {}
        p.skip_whitespace()
        result = Op.parse(p)
        if not p.eof():
            raise ValueError(f"trailing input at pos {p.pos}")
        return result

    @staticmethod
    def empty() -> "Dialect":
        """Create an anonymous dialect with only the generic Op/Type."""
        return Dialect("")


def iter_ops(
    root: Op, predicate: Callable[[Op], bool] = lambda op: True,
) -> Iterator[Op]:
    if predicate(root):
        yield root
    for region in root.regions:
        for block in region.blocks:
            for op in block.ops:
                yield from iter_ops(op, predicate)


def find_ops(root: Op, predicate: Callable[[Op], bool]) -> list[Op]:
    return list(iter_ops(root, predicate))


def find_op(root: Op, predicate: Callable[[Op], bool]) -> Op | None:
    return next(iter_ops(root, predicate), None)


def clone_type(t: "Type") -> "Type":
    return type(t)(attrs=copy.deepcopy(t.attrs))


def clone_op(op: Op, vmap: dict[Value, Value]) -> Op:
    new_operands = [vmap.get(v, v) for v in op.operands]
    new_results = [Value(name=v.name, type=clone_type(v.type)) for v in op.results]
    new_regions = [clone_region(r, vmap) for r in op.regions]

    new_op = type(op)(
        operands=new_operands,
        results=new_results,
        attrs=dict(op.attrs),
        regions=new_regions,
    )

    for old_r, new_r in zip(op.results, new_op.results):
        vmap[old_r] = new_r

    return new_op


def clone_block(block: Block, vmap: dict[Value, Value]) -> Block:
    inner_vmap = dict(vmap)
    new_args = []
    for arg in block.arguments:
        new_arg = Value(name=arg.name, type=clone_type(arg.type))
        inner_vmap[arg] = new_arg
        new_args.append(new_arg)
    return Block(arguments=new_args, ops=[clone_op(op, inner_vmap) for op in block.ops])


def clone_region(region: Region, vmap: dict[Value, Value]) -> Region:
    inner_vmap = dict(vmap)
    return Region(blocks=[clone_block(block, inner_vmap) for block in region.blocks])
