from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import ClassVar

from .parser import Parser
from .printer import Printer  # noqa: F401


# Observable containers that keep the use-list and parent links
# self-maintaining.
#
# The invariants: ``op.parent_block`` / ``block.parent_op`` point at the
# container holding the node, and ``value.uses`` contains ``(op, i)`` iff
# ``op.operands[i] is value`` — unconditionally, for as long as the op exists.
# Three list subclasses capture the three mutation channels so no pass
# registers/deregisters by hand:
#
# - ``_OperandList`` wraps an op's ``operands`` — writes rewire its uses.
# - ``_OpList`` wraps a block's ``ops`` — sets each op's ``parent_block``.
# - ``_BlockList`` wraps an op's ``blocks`` — sets each block's ``parent_op``.
#
# ``uses`` is deliberately structural rather than filtered: an op that has been
# spliced out of the IR, or built but not yet inserted, still appears there.
# Liveness is a question about *position*, so it is answered by position, at
# query time — see ``Value.users``, which keeps only users sharing a tree with
# the value's definition.
#
# All react the same way: snapshot, run the real list method, reconcile via
# ``_sync(old)``. Every mutator is generated from that one pattern below.
# Whole-list reassignment (``op.operands = [...]`` / ``block.ops = [...]`` /
# ``op.blocks = [...]``) is handled by ``Op``/``Block`` ``__setattr__``, which
# re-wraps into these types.


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
    """An op's ``operands``; rewires its values' ``uses``."""

    __slots__ = ("_op",)

    def __init__(self, iterable=(), *, op: "Op") -> None:
        super().__init__(iterable)
        self._op = op

    def _sync(self, old: list) -> None:
        op = self._op
        for i, v in enumerate(old):
            v.uses.discard((op, i))
        for i, v in enumerate(self):
            v.uses.add((op, i))


class _OpList(_ObservableList):
    """A block's ``ops``; points each op's ``parent_block`` back at it."""

    __slots__ = ("_block",)

    def __init__(self, iterable=(), *, block: "Block") -> None:
        super().__init__(iterable)
        self._block = block

    def _sync(self, old: list) -> None:
        block = self._block
        new_ids = {id(op) for op in self}
        old_ids = {id(op) for op in old}
        for op in old:
            # Only if still ours: a transplanted op already points at its new
            # block, and this stale sync must not null that out.
            if id(op) not in new_ids and op.parent_block is block:
                op.parent_block = None
        for op in self:
            if id(op) not in old_ids:
                op.parent_block = block


class _BlockList(_ObservableList):
    """An op's ``blocks``; points each block's ``parent_op`` back at it."""

    __slots__ = ("_op",)

    def __init__(self, iterable=(), *, op: "Op") -> None:
        super().__init__(iterable)
        self._op = op

    def _sync(self, old: list) -> None:
        op = self._op
        new_ids = {id(b) for b in self}
        old_ids = {id(b) for b in old}
        for b in old:
            if id(b) not in new_ids and b.parent_op is op:
                b.parent_op = None
        for b in self:
            if id(b) not in old_ids:
                b.parent_op = op


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
    definition: Op | Block | None = field(default=None, init=False)
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
        """Ops consuming this value from within the same IR tree.

        ``uses`` is structural, so it also holds ops that have been spliced out
        of the IR or built but never inserted. Those reference the value but are
        not part of the program any more (or yet), so they are filtered out
        here: a user counts when it shares a tree root with this value.
        """
        root = root_ancestor(self)
        seen: set[int] = set()
        out: list[Op] = []
        for op, _ in self.uses:
            if id(op) in seen:
                continue
            if root_ancestor(op) is not root:
                continue
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
    blocks: list[Block] = field(default_factory=list)

    parent_block: "Block | None" = field(default=None, init=False, repr=False)

    # Set by `Dialect.register_op` on each subclass.
    OP_ID: ClassVar[str] = "generic"

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
        elif name == "blocks":
            old = self.__dict__.get("blocks", [])
            new_list = _BlockList(value, op=self)
            super().__setattr__(name, new_list)
            new_list._sync(old)
        else:
            super().__setattr__(name, value)

    def operand(self, i: int = 0) -> Value:
        return self.operands[i]

    def result(self, i: int = 0) -> Value:
        return self.results[i]

    def block(self, i: int = 0) -> Block:
        return self.blocks[i]

    def attr(self, name: str) -> object:
        return self.attrs[name]

    @classmethod
    def build(
        cls,
        operands: list["Value"] | None = None,
        result_types: list["Type"] | None = None,
        attrs: dict[str, object] | None = None,
        blocks: list["Block"] | None = None,
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
            blocks=list(blocks or []),
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
        if self.blocks:
            p.write(" ")
        for i, b in enumerate(self.blocks):
            if i:
                p.write_newline()
            b.print(p)

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
        blocks: list[Block] = []
        while p.can_read_literal("{"):
            prefix_brace = p.read_literal("{")
            blocks.append(Block.parse(prefix_brace, p))
        return cls(
            operands=operands,
            results=results,
            attrs=attrs,
            blocks=blocks,
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

    parent_op: "Op | None" = field(default=None, init=False, repr=False)

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
            new_list = _OpList(value, block=self)
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
        p.write("{")
        p.write_newline()
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
        p.write("}")

    @classmethod
    def parse(
        cls,
        prefix: str,
        p: Parser,
        arguments: list[Value] | None = None,
    ) -> "Block":
        """Parse ``{ ^name(%a: t, ...): ops... }``. Passing `arguments`
        supplies the block's arguments from the enclosing op's own syntax
        (see `Block.print`'s `skip_label`), so no label is expected.
        """
        if prefix != "{":
            raise ValueError(
                f"expected block prefix '{{', got {prefix!r} at pos {p.pos}"
            )
        name: str | None = None
        if arguments is None:
            arguments = []
            p.read_literal("^")
            name = p.read_identifier()
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
        while not p.can_read_literal("}"):
            ops.append(Op.parse(p))
        p.read_literal("}")
        return cls(name=name, arguments=arguments, ops=ops)

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


def root_ancestor(node: "Op | Block | Value") -> "Op | Block | Value":
    """The top of ``node``'s tree, following the links that lead upward —
    ``node`` itself if nothing holds it.

    Two nodes share a root exactly when they belong to the same IR tree, which
    is how `Value.users` tells a live user from one that has been spliced out.
    A value sits under whatever defines it, so an undefined one is its own
    root and shares a tree with nothing.
    """
    while True:
        if isinstance(node, Value):
            parent = node.definition
        elif isinstance(node, Op):
            parent = node.parent_block
        elif isinstance(node, Block):
            parent = node.parent_op
        else:
            assert False, f"root_ancestor: unsupported node {type(node).__name__}"
        if parent is None:
            return node
        node = parent


def iter_ops(
    root: Op, predicate: Callable[[Op], bool] = lambda op: True,
) -> Iterator[Op]:
    if predicate(root):
        yield root
    for block in root.blocks:
        for op in block.ops:
            yield from iter_ops(op, predicate)


def find_ops(root: Op, predicate: Callable[[Op], bool]) -> list[Op]:
    return list(iter_ops(root, predicate))


def find_op(root: Op, predicate: Callable[[Op], bool]) -> Op | None:
    return next(iter_ops(root, predicate), None)


def clone_type(
    t: "Type", override_attrs: dict[str, object] | None = None,
) -> "Type":
    """Clone `t`.

    `override_attrs` is merged over the original's, for a copy that differs in
    one of them.
    """
    return type(t)(
        attrs={**copy.deepcopy(t.attrs), **(override_attrs or {})},
    )


def clone_op(
    op: Op, vmap: dict[Value, Value], override_attrs: dict[str, object] | None = None,
) -> Op:
    """Clone `op`, remapping operands through `vmap`.

    `override_attrs` is merged over the original's, for a copy that differs in
    one of them.
    """
    new_operands = [vmap.get(v, v) for v in op.operands]
    new_results = [Value(name=v.name, type=clone_type(v.type)) for v in op.results]
    new_blocks = [clone_block(b, vmap) for b in op.blocks]

    new_op = type(op)(
        operands=new_operands,
        results=new_results,
        attrs={**copy.deepcopy(op.attrs), **(override_attrs or {})},
        blocks=new_blocks,
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
