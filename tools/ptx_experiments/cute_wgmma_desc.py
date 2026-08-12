"""Rewrite CuTe's four WGMMA descriptors to use one base plus 64-bit adds.

This is an artifact-level experiment.  CuTe currently materializes offsets
6, 4, and 2 with low-word ORs and then joins each low/high pair.  Ayla instead
keeps the descriptor whole and derives offsets 2, 4, and 6 with ``add.u64``.
The latter is also correct if a low-word addition ever carries.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


_DESCRIPTORS = re.compile(
    r"(?P<indent>\s*)or\.b32\s+%r\d+,\s*(?P<lo>%r\d+),\s*6;\s*\n"
    r"\s*mov\.b64\s+(?P<d6>%rd\d+),\s*\{%r\d+,\s*(?P<hi>%r\d+)\};\s*\n"
    r"\s*or\.b32\s+%r\d+,\s*(?P=lo),\s*4;\s*\n"
    r"\s*mov\.b64\s+(?P<d4>%rd\d+),\s*\{%r\d+,\s*(?P=hi)\};\s*\n"
    r"\s*or\.b32\s+%r\d+,\s*(?P=lo),\s*2;\s*\n"
    r"\s*mov\.b64\s+(?P<d2>%rd\d+),\s*\{%r\d+,\s*(?P=hi)\};\s*\n"
    r"\s*mov\.b64\s+(?P<d0>%rd\d+),\s*\{(?P=lo),\s*(?P=hi)\};"
)


def transform(ptx: str) -> tuple[str, int]:
    def replacement(match: re.Match[str]) -> str:
        values = match.groupdict()
        indent = values["indent"]
        return "\n".join([
            f'{indent}mov.b64 {values["d0"]}, '
            f'{{{values["lo"]}, {values["hi"]}}};',
            f'{indent}add.u64 {values["d2"]}, {values["d0"]}, 2;',
            f'{indent}add.u64 {values["d4"]}, {values["d0"]}, 4;',
            f'{indent}add.u64 {values["d6"]}, {values["d0"]}, 6;',
        ])

    return _DESCRIPTORS.subn(replacement, ptx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result, count = transform(args.input.read_text())
    if count != 4:
        raise RuntimeError(f"expected four descriptor pairs, found {count}")
    args.output.write_text(result)
    print(f"rewrote {count} descriptor sets in {args.output}")


if __name__ == "__main__":
    main()
