"""Pipeline pairs of Ayla WGMMA stages without early SMEM reuse.

For each generated three-stage body this changes the first two drains to
``wait_group 1``.  A stage's CTA release/refill block is delayed until the next
wait, which proves that the older WGMMA group has completed before its shared
memory is overwritten.  The third stage retains ``wait_group 0`` and drains
both remaining groups before releasing the final two buffers.

This is intentionally a PTX artifact experiment, not a framework pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path


BEGIN = "// BEGIN: `for ("
END = "// END: `for ("
WAIT = "wgmma.wait_group.sync.aligned 0;"


def _is_mbar_wait_comment(line: str) -> bool:
    text = line.strip()
    return text.startswith("// PTX template `await(") and "lock<mbar>" in text


def transform(ptx: str) -> tuple[str, int]:
    lines = ptx.splitlines()
    output: list[str] = []
    cursor = 0
    transformed = 0
    while cursor < len(lines):
        begin = next(
            (
                i for i in range(cursor, len(lines))
                if lines[i].strip().startswith(BEGIN) and " step 3)" in lines[i]
            ),
            None,
        )
        if begin is None:
            output.extend(lines[cursor:])
            break
        end = next(
            i for i in range(begin + 1, len(lines))
            if lines[i].strip().startswith(END)
        )
        waits = [
            i for i in range(begin, end)
            if lines[i].strip() == WAIT
        ]
        if len(waits) != 3:
            raise RuntimeError(
                f"expected three WGMMA drains in loop at line {begin + 1}, "
                f"found {len(waits)}"
            )
        stage_starts = [
            i for i in range(begin, end) if _is_mbar_wait_comment(lines[i])
        ]
        if len(stage_starts) != 3:
            raise RuntimeError(
                f"expected three stage waits in loop at line {begin + 1}, "
                f"found {len(stage_starts)}"
            )
        if not all(stage_starts[i] < waits[i] for i in range(3)):
            raise RuntimeError("unexpected stage/WGMMA ordering")

        preamble_and_compute0 = lines[begin:waits[0] + 1]
        release0 = lines[waits[0] + 1:stage_starts[1]]
        compute1 = lines[stage_starts[1]:waits[1] + 1]
        release1 = lines[waits[1] + 1:stage_starts[2]]
        compute2 = lines[stage_starts[2]:waits[2] + 1]
        release2 = lines[waits[2] + 1:end]

        output.extend(lines[cursor:begin])
        output.extend(preamble_and_compute0[:-1])
        output.append(preamble_and_compute0[-1].replace(" 0;", " 1;"))
        output.extend(compute1[:-1])
        output.append(compute1[-1].replace(" 0;", " 1;"))
        output.extend(release0)
        output.extend(compute2)
        output.extend(release1)
        output.extend(release2)
        output.append(lines[end])
        cursor = end + 1
        transformed += 1
    return "\n".join(output) + ("\n" if ptx.endswith("\n") else ""), transformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result, count = transform(args.input.read_text())
    if count != 4:
        raise RuntimeError(f"expected four output-tile loops, transformed {count}")
    args.output.write_text(result)
    print(f"pipelined {count} three-stage loops in {args.output}")


if __name__ == "__main__":
    main()
