r"""Sink the TMA-store drain out of the epilogue and behind the next K loop.

`schedule_async` sinks an `await` to just before the first consumer of the
buffer it guards, but it cannot sink one past a `for` op, so every output-store
await lands inside the epilogue that issued the store:

    bar.sync 0; stmatrix(A); wait.read 0; leader_bar; store(A)
    bar.sync 0; stmatrix(B); wait.read 0; leader_bar; store(B)   <- drains A,
    <K loop>                                                        issued ~10
                                                                    instrs ago

The second wait exists to protect `stmatrix(A)` in the *next* epilogue, a whole
K loop later, so it drains a store that has had almost no time to run.  Because
`cp.async.bulk.wait_group.read 0` drains everything, one wait placed at the top
of each epilogue is both sufficient and strictly later:

    <K loop>
    wait.read 0                 <- both stores have had the whole K loop
    bar.sync 0; stmatrix(A); leader_bar; store(A)
    bar.sync 0; stmatrix(B); leader_bar; store(B)

The full `bar.sync 0` right after the wait is what publishes thread 0's drain to
the other 255 threads (only thread 0 ever issues a bulk group, so the wait is a
no-op everywhere else).  The trailing wait before `ret` is left alone.
"""
from __future__ import annotations

import re
import sys

WAIT = re.compile(r"^(\s*)cp\.async\.bulk\.wait_group\.read 0;\s*$")
STORE = re.compile(r"^\s*cp\.async\.bulk\.tensor\.\dd\.global\.")
LOOP_BEGIN = re.compile(r"^\s*// BEGIN: `for ")
FULL_BARRIER = re.compile(r"^\s*bar\.sync 0;\s*$")


def _epilogues(lines: list[str]) -> list[list[int]]:
    """Bulk-store line indices grouped by the K loop that precedes them."""
    groups: list[list[int]] = []
    fresh = True
    for i, line in enumerate(lines):
        if LOOP_BEGIN.match(line):
            fresh = True
        elif STORE.match(line):
            if fresh:
                groups.append([])
                fresh = False
            groups[-1].append(i)
    return groups


def transform(ptx: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    groups = _epilogues(lines)
    assert groups, "no bulk stores found"

    drop: set[int] = set()
    insert: list[tuple[int, str]] = []
    last_store = groups[-1][-1]

    for n, stores in enumerate(groups):
        lo = 0 if n == 0 else groups[n - 1][-1] + 1
        hi = stores[-1]
        # Every wait between the previous epilogue's last store and this
        # epilogue's last store is redundant once one wait covers the top.
        drop.update(i for i in range(lo, hi) if WAIT.match(lines[i]))
        if n == 0:
            continue  # nothing outstanding before the first epilogue
        # Latest safe point: immediately before the barrier that publishes the
        # drain to the warps that run stmatrix.
        anchor = next(
            i for i in range(stores[0], lo, -1) if FULL_BARRIER.match(lines[i])
        )
        insert.append((anchor, lines[anchor][: len(lines[anchor]) - len(lines[anchor].lstrip())]))

    assert insert, "expected more than one epilogue"
    out: list[str] = []
    for i, line in enumerate(lines):
        for anchor, indent in insert:
            if i == anchor:
                out.append(f"{indent}cp.async.bulk.wait_group.read 0;")
                out.append("")
        if i in drop:
            continue
        out.append(line)

    kept = sum(1 for i, l in enumerate(lines) if WAIT.match(l) and i not in drop)
    if verbose:
        print(f"  dropped {len(drop)} in-epilogue store drains, "
              f"inserted {len(insert)} post-loop drains, kept {kept} "
              f"(last store at line {last_store + 1})")
    return "\n".join(out) + "\n", len(drop)


if __name__ == "__main__":
    out, n = transform(open(sys.argv[1]).read(), verbose=True)
    open(sys.argv[2], "w").write(out)
