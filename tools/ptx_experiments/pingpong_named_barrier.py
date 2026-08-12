"""Asymmetric WG ping-pong using Ayla's existing named barriers.

WG0's 128 threads arrive and continue. WG1's 128 threads arrive-and-wait; once
all 256 participants have reached the boundary, WG1 elects tid 128 to refill
the released operand slot. This changes ownership without adding mbarriers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


_BARRIER = re.compile(
    r'^\s*// PTX template `barrier\(\) attrs=\{id: (\d+), scope: "block_leader"\}`'
)
_INPUT_ASYNC = re.compile(
    r'^\s*// PTX template `async\(%0: lock<mbar>\) attrs=\{label: "mm_lhs_smem_'
)
_GUARD = re.compile(r"^(\s*)setp\.ne\.s32 (%p\d+), (%r\d+), 0;\s*$")


def transform(ptx: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    changed = 0
    for cursor, line in enumerate(lines):
        match = _BARRIER.match(line)
        if match is None:
            continue
        next_marker = next(
            (i for i in range(cursor + 1, len(lines))
             if lines[i].lstrip().startswith("// PTX template")),
            None,
        )
        if next_marker is None or not _INPUT_ASYNC.match(lines[next_marker]):
            continue

        barrier_id = match.group(1)
        barrier_ops = lines[cursor + 1:next_marker]
        setp = next(i for i, op in enumerate(barrier_ops) if "setp.lt.u32" in op)
        predicate = re.search(r"(%p\d+)", barrier_ops[setp]).group(1)
        barrier_ops[setp] = barrier_ops[setp].replace(", 32;", ", 128;")
        sync = next(i for i, op in enumerate(barrier_ops) if "bar.sync" in op)
        arrive = next(i for i, op in enumerate(barrier_ops) if "bar.arrive" in op)
        indent = re.match(r"\s*", barrier_ops[sync]).group(0)
        barrier_ops[sync] = f"{indent}@{predicate} bar.arrive {barrier_id}, 256;"
        barrier_ops[arrive] = f"{indent}@!{predicate} bar.sync {barrier_id}, 256;"
        lines[cursor + 1:next_marker] = barrier_ops

        async_end = next(
            (i for i in range(next_marker + 1, len(lines))
             if lines[i].lstrip().startswith("// PTX template")
             or lines[i].lstrip().startswith("// END:")),
            len(lines),
        )
        guards = 0
        for i in range(next_marker, async_end):
            guard = _GUARD.match(lines[i])
            if guard:
                lines[i] = f"{guard.group(1)}setp.ne.s32 {guard.group(2)}, {guard.group(3)}, 128;"
                guards += 1
        if guards != 1:
            raise RuntimeError(f"expected one TMA issuer guard, found {guards}")
        changed += 1

    if verbose:
        print(f"converted {changed} releases to named-barrier WG ping-pong")
    return "\n".join(lines) + ("\n" if ptx.endswith("\n") else ""), changed


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:3])
    result, count = transform(source.read_text(), verbose=True)
    destination.write_text(result)
    print(f"wrote {destination} ({count} converted releases)")
