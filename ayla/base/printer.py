"""Pretty-printer with indentation, anchors, and unique-name minting."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from typing import cast

from .. import pytree


class Printer:
    def __init__(self, indent_str: str = "  ") -> None:
        self._lines: list[str] = []
        self._cur: str = ""
        self._level: int = 0
        self._indent_str = indent_str
        self._assigned_names: dict[tuple[str, object], str | tuple] = {}
        self._used_names: dict[str, set[str]] = defaultdict(set)
        self._suffix_counters: dict[tuple[str, str], int] = defaultdict(int)

    def unique_names[T: str | tuple](
        self,
        obj: object,
        basenames: T,
        namespace: str = "default",
        always_suffix: bool = False,
    ) -> T:
        key = (namespace, obj)
        if key in self._assigned_names:
            return cast(T, self._assigned_names[key])

        def mint_name(base: str) -> str:
            return self._mint_name(base, namespace, always_suffix)

        result = pytree.map(mint_name, basenames)
        self._assigned_names[key] = result
        return cast(T, result)

    def _mint_name(
        self, base: str, namespace: str, always_suffix: bool,
    ) -> str:
        used = self._used_names[namespace]
        sep = "_" if base and base[-1].isdigit() else ""
        suffix = self._suffix_counters[(namespace, base)]
        candidate = (
            f"{base}{sep}{suffix}"
            if always_suffix or suffix > 0
            else base
        )
        while candidate in used:
            suffix += 1
            candidate = f"{base}{sep}{suffix}"
        self._suffix_counters[(namespace, base)] = suffix + 1
        used.add(candidate)
        return candidate

    def name_count(self, basename: str, namespace: str = "default") -> int:
        """Number of names minted with `basename` in `namespace`. For
        `always_suffix` names this is also the next suffix index.
        """
        return self._suffix_counters.get((namespace, basename), 0)

    def write(self, s: str) -> None:
        if not self._cur:
            self._cur = self._indent_str * self._level
        self._cur += s

    def write_attr(self, v: object) -> None:
        if isinstance(v, str):
            self.write_str_attr(v)
            return
        if isinstance(v, list):
            self.write_list_attr(v)
            return
        if isinstance(v, dict):
            self.write_dict_attr(v)
            return
        self.write_literal_attr(v)  # type: ignore[arg-type]

    def write_literal_attr(
        self, v: int | float | bool | None,
    ) -> None:
        """Write an atomic literal: int / float / bool / None."""
        assert v is None or isinstance(v, (bool, int, float)), (
            f"write_literal_attr: expected int/float/bool/None, got "
            f"{type(v).__name__}: {v!r}"
        )
        self.write(repr(v))

    def write_str_attr(self, v: str) -> None:
        """Write a quoted string literal, escaping `\\\\`, `\\"`, `\\n`,
        `\\t`.
        """
        s = (
            v.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        self.write(f'"{s}"')

    def write_shape(
        self, sizes: list[int | None] | tuple[int | None, ...], dtype=None,
    ) -> None:
        """Write `NxMx...` or `NxMx...xDTYPE` if dtype is given."""
        self.write("x".join("?" if s is None else str(s) for s in sizes))
        if dtype is not None:
            if sizes:
                self.write("x")
            dtype.print(self)

    def write_list_attr(self, v: list, brackets: bool = True) -> None:
        """Write `[v, ...]`."""
        if brackets:
            self.write("[")
        for i, x in enumerate(v):
            if i:
                self.write(", ")
            self.write_attr(x)
        if brackets:
            self.write("]")

    def write_dict_attr(self, v: dict, braces: bool = True) -> None:
        """Write ``{k: v, ...}`` with identifier or integer keys."""
        if braces:
            self.write("{")
        items = sorted(
            v.items(),
            key=lambda item: (
                0 if type(item[0]) is int else 1,
                item[0],
            ),
        )
        for i, (k, x) in enumerate(items):
            if i:
                self.write(", ")
            if type(k) is int:
                self.write_literal_attr(k)
            else:
                assert isinstance(k, str) and k.isidentifier(), (
                    "write_dict_attr: key must be an integer or an "
                    f"identifier-like str, got {k!r}"
                )
                self.write(k)
            self.write(": ")
            self.write_attr(x)
        if braces:
            self.write("}")

    def write_extra_attrs(
        self, attrs: dict, skip_keys: Iterable[str] = (),
    ) -> None:
        """Write ` attrs={...}` for the entries of `attrs` not in
        `skip_keys`. Writes nothing if no entries remain.
        """
        skip = set(skip_keys)
        filtered = {k: v for k, v in attrs.items() if k not in skip}
        if not filtered:
            return
        self.write(" attrs=")
        self.write_attr(filtered)

    def write_newline(self) -> None:
        self._lines.append(self._cur)
        self._cur = ""

    def write_newline_once(self) -> None:
        """Write a blank line only if the last line isn't already blank."""
        if self._lines and self._lines[-1].strip() == "" and self._cur == "":
            return
        self.write_newline()

    def write_line(self, s: str) -> None:
        if s:
            self.write(s)
        self.write_newline()

    @contextmanager
    def indented(self):
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    def mark(self) -> tuple[int, int]:
        """Capture the current line index + indent level for later
        insertion via `insert_at`.
        """
        return (len(self._lines), self._level)

    @contextmanager
    def insert_at(self, mark: tuple[int, int]):
        """Temporarily redirect the printer so writes accumulate into a
        buffer that is spliced into the line list at `mark` on exit.
        Indent is restored to the mark's level.
        """
        idx, level = mark
        saved_lines, saved_level = self._lines, self._level
        buffer: list[str] = []
        self._lines, self._level = buffer, level
        try:
            yield
        finally:
            # Flush any pending in-progress line to the buffer.
            if self._cur:
                buffer.append(self._cur)
                self._cur = ""
            self._lines, self._level = saved_lines, saved_level
            saved_lines[idx:idx] = buffer

    def render(self) -> str:
        if self._cur:
            self._lines.append(self._cur)
            self._cur = ""
        return "\n".join(self._lines)
