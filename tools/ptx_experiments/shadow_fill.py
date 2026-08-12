r"""Fill the `wgmma.wait_group 0` drain shadow with the next stage's prologue.

Per k64 stage the kernel emits

    wgmma.commit_group
    wgmma.wait_group 0          <- 64% of stall samples; tensor cores idle after
    <addr calc>
    barrier(block_leader)
    TMA load into the stage just freed
    mbarrier spin for the *next* stage        \  dead tensor time: neither
    two 64-bit GMMA descriptor rebuilds       /  depends on the drain
    wgmma group for the next stage

The mbarrier spin waits on the stage loaded two iterations ago, and the
descriptors derive only from prologue-fixed SMEM slot addresses, so both can
legally execute *before* the drain, i.e. while the tensor cores are still busy
with the group that was just committed.

`hoist_mbar_await` moves the spin. `hoist_descriptors` additionally moves the
`shr/cvt/or` descriptor rebuild (but never the `wgmma.fence`, which must stay
adjacent to its group -- dropping fences makes ptxas inject WARPGROUP.ARRIVE).

Both transforms only move a block when the drain it hops is inside the same
straight-line region: loop headers, loop BEGIN/END markers and backward
branches are hard stops, so nothing is lifted out of a loop body (which is what
made the earlier whole-loop descriptor hoist regress -- see
docs/profiling/sass-stage-boundary-2026-08-05.md).
"""
from __future__ import annotations

import re
import sys

WAIT_WGMMA = re.compile(r"^\s*wgmma\.wait_group\.sync\.aligned 0;\s*$")
LABEL = re.compile(r"^\s*(\$L\d+):\s*$")
BRA = re.compile(r"^\s*(?:@!?%p\d+ )?bra(?:\.uni)? (\$L\d+);\s*$")
MBAR_AWAIT_COMMENT = re.compile(r"^\s*// PTX template `await\(%\d+: lock<mbar>\)`\s*$")
WGMMA_AWAIT_COMMENT = re.compile(r"^\s*// PTX template `await\(%\d+: lock<wgmma>\)`\s*$")
TRY_WAIT = re.compile(r"^\s*mbarrier\.try_wait\.parity\.[\w:.]+ (%p\d+), \[(%r\d+)\], (%r\d+);\s*$")
REGION_BREAK = re.compile(r"^\s*// (BEGIN|END):")


def _loop_head_labels(lines: list[str]) -> set[str]:
    """Labels that some later `bra` jumps back to."""
    seen: dict[str, int] = {}
    heads: set[str] = set()
    for i, line in enumerate(lines):
        m = LABEL.match(line)
        if m:
            seen.setdefault(m.group(1), i)
            continue
        m = BRA.match(line)
        if m and m.group(1) in seen:
            heads.add(m.group(1))
    return heads


def _mbar_await_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """`(start, end)` half-open line ranges of each mbarrier await block.

    A block is the `// PTX template` marker through the parity flip that
    follows the spin loop.
    """
    blocks = []
    for i, line in enumerate(lines):
        if not MBAR_AWAIT_COMMENT.match(line):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not LABEL.match(lines[j]):
            continue
        if j + 1 >= len(lines) or not TRY_WAIT.match(lines[j + 1]):
            continue
        end = j + 2
        while end < len(lines) and (
            lines[end].strip().startswith("@!%p")
            or lines[end].strip().startswith("xor.b32")
        ):
            end += 1
        blocks.append((i, end))
    return blocks


def _drain_above(lines: list[str], start: int, heads: set[str]) -> int | None:
    """Index of the `wgmma.wait_group 0` guarding the stage before `start`.

    Returns `None` if a region boundary, another mbarrier spin, or the top of
    the block is reached first.
    """
    for i in range(start - 1, -1, -1):
        line = lines[i]
        if WAIT_WGMMA.match(line):
            return i
        if REGION_BREAK.match(line):
            return None
        m = LABEL.match(line)
        if m and m.group(1) in heads:
            return None
        if TRY_WAIT.match(line):
            return None
    return None


def _insert_point(lines: list[str], drain: int) -> int:
    """Line to insert before: the drain, or its `await(wgmma)` marker."""
    i = drain - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i >= 0 and WGMMA_AWAIT_COMMENT.match(lines[i]):
        return i
    return drain


def hoist_mbar_await(ptx: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    heads = _loop_head_labels(lines)
    moves = []
    for start, end in _mbar_await_blocks(lines):
        drain = _drain_above(lines, start, heads)
        if drain is None:
            continue
        moves.append((start, end, _insert_point(lines, drain)))

    # Apply back-to-front so earlier indices stay valid.
    for start, end, dest in sorted(moves, reverse=True):
        assert dest < start
        block = lines[start:end]
        del lines[start:end]
        lines[dest:dest] = block + [""]
    if verbose:
        print(f"  hoisted {len(moves)} mbarrier spins above their drain")
    return "\n".join(lines) + "\n", len(moves)


DESC_OPS = (
    re.compile(r"^\s*shr\.b32 (%r\d+), (%r\d+), 4;\s*$"),
    re.compile(r"^\s*cvt\.u64\.u32 (%rd\d+), (%r\d+);\s*$"),
    re.compile(r"^\s*or\.b64 (%rd\d+), (%rd\d+), \d+;\s*$"),
)


def hoist_descriptors(ptx: str, verbose: bool = False) -> tuple[str, int]:
    """Move the two `shr/cvt/or` descriptor rebuilds above the preceding drain.

    Only the six-line run that directly follows a `wgmma.fence` is touched, and
    only when the drain above it is in the same straight-line region.
    """
    lines = ptx.splitlines()
    heads = _loop_head_labels(lines)
    moves = []
    for i, line in enumerate(lines):
        if line.strip() != "wgmma.fence.sync.aligned;":
            continue
        j = i + 1
        run = []
        while j < len(lines):
            stripped = lines[j]
            if any(p.match(stripped) for p in DESC_OPS):
                run.append(j)
                j += 1
                continue
            if stripped.strip().startswith("setp.ne.u32"):
                j += 1  # scale-d predicate: leave in place, it reads the loop var
                continue
            break
        if len(run) != 6 or run != list(range(run[0], run[0] + 6)):
            continue
        drain = _drain_above(lines, run[0], heads)
        if drain is None:
            continue
        moves.append((run[0], run[0] + 6, _insert_point(lines, drain)))

    for start, end, dest in sorted(moves, reverse=True):
        assert dest < start
        block = lines[start:end]
        del lines[start:end]
        lines[dest:dest] = block
    if verbose:
        print(f"  hoisted {len(moves)} descriptor rebuilds above their drain")
    return "\n".join(lines) + "\n", len(moves)


TRANSFORMS = {
    "mbar": hoist_mbar_await,
    "desc": hoist_descriptors,
}


def transform(ptx: str, which: str = "mbar", verbose: bool = False) -> tuple[str, int]:
    total = 0
    for name in which.split("+"):
        ptx, n = TRANSFORMS[name](ptx, verbose=verbose)
        total += n
    return ptx, total


if __name__ == "__main__":
    which = sys.argv[3] if len(sys.argv) > 3 else "mbar"
    out, n = transform(open(sys.argv[1]).read(), which, verbose=True)
    open(sys.argv[2], "w").write(out)
