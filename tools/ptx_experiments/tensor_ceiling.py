r"""Ablate the mainloop down to bare WGMMA to find this chip's real tensor ceiling.

The datasheet peak (132 SM x 4096 FLOP/clk) is an upper bound that no real
WGMMA stream reaches, so "% of peak" over-states how much of the residual gap
is actually addressable.  These ablations strip one mainloop obligation at a
time and time what is left.  Every variant computes **garbage** -- buffers are
never refilled, waits are removed -- but the HGMMA count, the accumulator
dependency chain, the register allocation and the 221 KB SMEM footprint (hence
1 CTA/SM) are all preserved, so the timing is a valid probe of issue rate.

    notma   drop the TMA load blocks (and their expect_tx arrival)
    nomb    drop the mbarrier spins that wait on those loads
    nobar   drop the per-stage block-leader barrier rendezvous
    drainN  relax the per-stage `wgmma.wait_group 0` to depth N; a depth-0
            wait is always kept immediately before each `convert`, which is
            the real consumer of the accumulators

The interesting quantity is the gap between the full kernel and
`notma+nomb+nobar+drain0`: that is everything sync- and memory-related.  What
remains below *that* is hardware, not schedulable.
"""
from __future__ import annotations

import re
import sys

MBAR_AWAIT = re.compile(r"^\s*// PTX template `await\(%\d+: lock<mbar>\)`\s*$")
WGMMA_AWAIT = re.compile(r"^\s*// PTX template `await\(%\d+: lock<wgmma>\)`\s*$")
BARRIER = re.compile(r"^\s*// PTX template `barrier\(\) attrs=\{id: \d+, scope: \"block_leader\"\}`")
TMA_LOAD = re.compile(r"^\s*// PTX template `async\(%\d+: lock<mbar>\) attrs=\{label: \"mm_(lhs|rhs)_")
CONVERT = re.compile(r"^\s*// PTX template `convert\(")
DRAIN = re.compile(r"^(\s*)wgmma\.wait_group\.sync\.aligned 0;\s*$")
LABEL = re.compile(r"^\s*(\$L\d+):\s*$")


def _block_end(lines: list[str], start: int) -> int:
    """End of a `// PTX template` block: up to the next blank-line-separated marker."""
    i = start + 1
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("// PTX template") or s.startswith("// BEGIN:") or s.startswith("// END:"):
            break
        i += 1
    # trim trailing blanks back to the last real statement
    while i > start + 1 and not lines[i - 1].strip():
        i -= 1
    return i


def _drop_blocks(lines: list[str], marker: re.Pattern) -> tuple[list[str], int]:
    out, i, n = [], 0, 0
    while i < len(lines):
        if marker.match(lines[i]):
            i = _block_end(lines, i)
            n += 1
            continue
        out.append(lines[i])
        i += 1
    return out, n


def transform(ptx: str, spec: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    total = 0
    steps = spec.split("+")

    for step in steps:
        if step == "notma":
            lines, n = _drop_blocks(lines, TMA_LOAD)
        elif step == "nomb":
            lines, n = _drop_blocks(lines, MBAR_AWAIT)
        elif step == "nobar":
            lines, n = _drop_blocks(lines, BARRIER)
        elif step.startswith("drain"):
            depth = int(step[5:])
            # Keep a full drain in front of every convert; relax the rest.
            keep = set()
            for i, line in enumerate(lines):
                if not CONVERT.match(line):
                    continue
                j = i - 1
                while j >= 0 and not DRAIN.match(lines[j]):
                    j -= 1
                if j >= 0:
                    keep.add(j)
            n = 0
            for i, line in enumerate(lines):
                m = DRAIN.match(line)
                if m and i not in keep and depth != 0:
                    lines[i] = f"{m.group(1)}wgmma.wait_group.sync.aligned {depth};"
                    n += 1
        else:
            raise ValueError(f"unknown ablation {step!r}")
        total += n
        if verbose:
            print(f"  {step}: {n} sites")

    # Dangling branch targets are legal PTX labels; leave them.
    return "\n".join(lines) + "\n", total


if __name__ == "__main__":
    out, n = transform(open(sys.argv[1]).read(), sys.argv[3], verbose=True)
    open(sys.argv[2], "w").write(out)
