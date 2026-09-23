"""IR text parser. Currently a shell — minimal stream + identifier helpers.

Whitespace invariant: outside of `parse`'s initial setup, the parser is
never positioned at whitespace. Every `read_*` method consumes its token
and then skips any trailing whitespace.
"""
from __future__ import annotations

import string
from collections.abc import Callable
from typing import Any


class Parser:
    def __init__(
        self,
        type_parse_fns: dict[str, Callable[[str, "Parser"], Any]],
        op_parse_fns: dict[
            str, Callable[[list, str, "Parser"], Any],
        ],
    ) -> None:
        self.type_parse_fns = type_parse_fns
        self.op_parse_fns = op_parse_fns
        self.source: str = ""
        self.pos: int = 0
        # Name → Value scope for cross-referencing operands to defs.
        self.values: dict[str, Any] = {}

    def eof(self) -> bool:
        return self.pos >= len(self.source)

    def is_at(self, chars: str) -> bool:
        """True if not at EOF and the next char is in `chars`."""
        return not self.eof() and self.source[self.pos] in chars

    def skip_whitespace(self) -> None:
        while True:
            while self.is_at(string.whitespace):
                self.pos += 1
            if not self.eof() and (
                self.source.startswith("//", self.pos)
                or self.source[self.pos] == "#"
            ):
                nl = self.source.find("\n", self.pos)
                self.pos = nl + 1 if nl >= 0 else len(self.source)
                continue
            break

    def can_read_literal(self, lit: str) -> bool:
        """True if `lit` is at the current position (non-destructive)."""
        return self.source.startswith(lit, self.pos)

    def read_literal(self, lit: str) -> str:
        """Consume `lit` exactly, raising if it isn't there."""
        if not self.source.startswith(lit, self.pos):
            raise ValueError(f"expected {lit!r} at pos {self.pos}")
        self.pos += len(lit)
        self.skip_whitespace()
        return lit

    def can_read_int(self) -> bool:
        """True if a digit is at the current position (non-destructive)."""
        return self.is_at(string.digits)

    def read_int(self) -> int:
        """Consume `[0-9]+`, raising if absent."""
        start = self.pos
        if not self.is_at(string.digits):
            raise ValueError(f"expected int at pos {self.pos}")
        while self.is_at(string.digits):
            self.pos += 1
        result = int(self.source[start:self.pos])
        self.skip_whitespace()
        return result

    def read_dim(self) -> int | None:
        if self.can_read_literal("?"):
            self.read_literal("?")
            return None
        return self.read_int()

    def read_shape(self) -> list[int | None]:
        """Read `NxMx...`.

        A ``?`` dimension is represented as ``None``.
        """
        sizes = [self.read_dim()]
        while self.can_read_literal("x"):
            self.read_literal("x")
            sizes.append(self.read_dim())
        return sizes

    def read_shape_with_dtype(self) -> tuple[list[int | None], Any]:
        """Read `NxMx...xDTYPE`."""
        sizes: list[int | None] = []
        while self.can_read_literal("?") or self.can_read_int():
            sizes.append(self.read_dim())
            self.read_literal("x")
        return sizes, self.read_type()

    def read_identifier(self, allow_leading_digits: bool = False) -> str:
        """Consume an identifier. By default `[A-Za-z_][A-Za-z0-9_]*`; with
        `allow_leading_digits=True` accepts `[A-Za-z0-9_]+` so digit-prefixed
        names like `%0` round-trip from the printer's auto-IDs.
        """
        start = self.pos
        first_chars = string.ascii_letters + "_"
        if allow_leading_digits:
            first_chars += string.digits
        if not self.is_at(first_chars):
            raise ValueError(f"expected identifier at pos {self.pos}")
        while self.is_at(string.ascii_letters + string.digits + "_"):
            self.pos += 1
        result = self.source[start:self.pos]
        self.skip_whitespace()
        return result

    def read_attr(self) -> object:
        """Inverse of Printer.write_attr. Reads one attribute value
        (str/int/float/bool/None/list/dict) and returns it.
        """
        if self.is_at("{"):
            return self.read_dict_attr()
        if self.is_at('"'):
            return self.read_str_attr()
        if self.is_at("["):
            return self.read_list_attr()
        return self.read_literal_attr()

    def read_literal_attr(self) -> int | float | bool | None:
        """Read an atomic literal: int / float / bool / None.

        Booleans accept both `True`/`False` and `true`/`false`.
        """
        LITERAL_CHARS = string.ascii_letters + string.digits + "_.+-"
        start = self.pos
        while self.is_at(LITERAL_CHARS):
            self.pos += 1
        text = self.source[start:self.pos]
        if not text:
            raise ValueError(f"expected literal at pos {start}")
        self.skip_whitespace()
        if text in ("True", "true"):
            return True
        if text in ("False", "false"):
            return False
        if text in ("None", "none"):
            return None
        try:
            return int(text)
        except ValueError:
            return float(text)

    def read_str_attr(self) -> str:
        """Read a quoted string literal (with `\\\\`, `\\"`, `\\n`, `\\t`
        escapes) and return its value.
        """
        assert self.is_at('"'), (
            f"read_str_attr: expected '\"' at pos {self.pos}"
        )
        start = self.pos
        self.pos += 1
        chars: list[str] = []
        while True:
            if self.eof():
                raise ValueError(
                    f"unterminated string starting at pos {start}"
                )
            c = self.source[self.pos]
            if c == "\\":
                if self.pos + 1 >= len(self.source):
                    raise ValueError(
                        f"unterminated escape at pos {self.pos}"
                    )
                nxt = self.source[self.pos + 1]
                chars.append({
                    "\\": "\\", '"': '"', "n": "\n", "t": "\t",
                }.get(nxt, nxt))
                self.pos += 2
                continue
            if c == '"':
                self.pos += 1
                break
            chars.append(c)
            self.pos += 1
        self.skip_whitespace()
        return "".join(chars)

    def read_dict_attr(self, braces: bool = True) -> dict[str, object]:
        """Read a comma-separated dict of `key: value` pairs (identifier
        keys, attr values).

        When `braces=True`, a surrounding `{...}` is required.
        """
        if braces:
            self.read_literal("{")
        items: list[tuple[str, object]] = []
        while True:
            k = self.read_identifier()
            self.read_literal(":")
            items.append((k, self.read_attr()))
            if not self.can_read_literal(","):
                break
            self.read_literal(",")
        if braces:
            self.read_literal("}")
        return dict(items)

    def read_extra_attrs(self) -> dict[str, object]:
        """Optionally read an `attrs={...}` clause. Returns `{}` if no
        `attrs` keyword is present.
        """
        if not self.can_read_literal("attrs"):
            return {}
        self.read_literal("attrs")
        self.read_literal("=")
        return self.read_dict_attr()

    def read_list_attr(self, brackets: bool = True) -> list:
        """Read a comma-separated list of attrs.

        When `brackets=True`, a surrounding `[...]` is required.
        """
        if brackets:
            self.read_literal("[")
        xs: list[object] = []
        if not self.is_at("]"):
            xs.append(self.read_attr())
            while self.can_read_literal(","):
                self.read_literal(",")
                xs.append(self.read_attr())
        if brackets:
            self.read_literal("]")
        return xs

    def read_tuple_attr(
        self,
        types: tuple[type, ...] | None = None,
        parens: bool = True,
    ) -> tuple:
        """Read a comma-separated tuple of attrs.

        When `parens=True`, a surrounding `(...)` is required. When
        `parens=False`, no surrounding delimiters; reads at least one
        element and continues while the next token is `,`.

        If `types` is given, asserts the tuple has exactly that length
        and each element matches the corresponding type.
        """
        if parens:
            self.read_literal("(")
        xs: list[object] = [self.read_attr()]
        while self.can_read_literal(","):
            self.read_literal(",")
            xs.append(self.read_attr())
        if parens:
            self.read_literal(")")
        v = tuple(xs)
        if types is not None:
            if len(v) != len(types):
                raise ValueError(
                    f"expected tuple of length {len(types)}, "
                    f"got length {len(v)}"
                )
            for i, (x, t) in enumerate(zip(v, types)):
                if not isinstance(x, t):
                    raise ValueError(
                        f"tuple element {i} expected {t.__name__}, "
                        f"got {type(x).__name__}"
                    )
        return v

    def read_op(self, results: list) -> Any:
        """Read op name, dispatch to registered parser, validate results."""
        prefix = self.read_identifier()
        if prefix not in self.op_parse_fns:
            raise ValueError(
                f"no op parser registered for {prefix!r} at pos {self.pos}"
            )
        op = self.op_parse_fns[prefix](results, prefix, self)
        for i, v in enumerate(results):
            assert v.definition is op and v.result_index == i, (
                f"op parser for {prefix!r} did not register result {i} "
                f"with the op (got definition={v.definition!r}, "
                f"result_index={v.result_index!r})"
            )
        self.skip_whitespace()
        return op

    def read_type(self) -> Any:
        """Read one inline type (identifier + dispatch) at the current pos."""
        prefix = self.read_identifier()
        if prefix not in self.type_parse_fns:
            raise ValueError(
                f"no type parser registered for {prefix!r} at pos {self.pos}"
            )
        result = self.type_parse_fns[prefix](prefix, self)
        self.skip_whitespace()
        return result
