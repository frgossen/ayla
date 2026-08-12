"""Prototype an asymmetric two-warpgroup Ayla operand pipeline.

WG0 releases each consumed operand slot and immediately advances.  WG1 waits
for that release, then issues the refill from an elected lane in warp 4.  This
removes the CTA-wide rendezvous from the hot path and lets WG0 lead WG1 while
retaining the existing three operand buffers and both compute warpgroups.

This is deliberately an artifact-level experiment, not a framework pass.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


_BARRIER = re.compile(
    r'^\s*// PTX template `barrier\(\) attrs=\{id: \d+, scope: "block_leader"\}`'
)
_INPUT_ASYNC = re.compile(
    r'^\s*// PTX template `async\(%0: lock<mbar>\) attrs=\{label: "mm_lhs_smem_(iter[012]|tail0)_'
)
_GUARD = re.compile(r"^(\s*)setp\.ne\.s32 (%p\d+), (%r\d+), 0;\s*$")


def transform(ptx: str, verbose: bool = False) -> tuple[str, int]:
    lines = ptx.splitlines()
    if ".reg .b32  %r<288>;" not in ptx or ".reg .pred %p<137>;" not in ptx:
        raise ValueError("prototype expects the current 288-register PTX numbering")

    lines = [
        line.replace(".reg .b32  %r<288>;", ".reg .b32  %r<294>;")
            .replace(".reg .pred %p<137>;", ".reg .pred %p<141>;")
        for line in lines
    ]

    tid = next(i for i, line in enumerate(lines) if line.strip() == "mov.u32 %r0, %tid.x;")
    init = [
        "",
        "  // Experimental WG0 -> WG1 operand-slot release barriers.",
        "  .shared .align 8 .b64 _pp_done_0[1];",
        "  .shared .align 8 .b64 _pp_done_1[1];",
        "  .shared .align 8 .b64 _pp_done_2[1];",
        "  mov.u32 %r288, _pp_done_0;",
        "  mov.u32 %r289, _pp_done_1;",
        "  mov.u32 %r290, _pp_done_2;",
        "  mov.u32 %r291, 0;",
        "  mov.u32 %r292, 0;",
        "  mov.u32 %r293, 0;",
        "  setp.ne.u32 %p137, %r0, 0;",
        "  @%p137 bra $L72;",
        "  mbarrier.init.shared::cta.b64 [%r288], 1;",
        "  mbarrier.init.shared::cta.b64 [%r289], 1;",
        "  mbarrier.init.shared::cta.b64 [%r290], 1;",
        "  fence.mbarrier_init.release.cluster;",
        "  $L72:",
        "  bar.sync 0;",
    ]
    lines[tid + 1:tid + 1] = init

    output: list[str] = []
    cursor = 0
    changed = 0
    next_label = 73
    while cursor < len(lines):
        match = _BARRIER.match(lines[cursor])
        if match is None:
            output.append(lines[cursor])
            cursor += 1
            continue

        # Only replace barriers whose immediate consumer is an operand TMA
        # reload. Output-store leader barriers must retain their old semantics.
        next_marker = next(
            (i for i in range(cursor + 1, len(lines))
             if lines[i].lstrip().startswith("// PTX template")),
            None,
        )
        if next_marker is None or not _INPUT_ASYNC.match(lines[next_marker]):
            output.append(lines[cursor])
            cursor += 1
            continue

        async_match = _INPUT_ASYNC.match(lines[next_marker])
        assert async_match is not None
        slot_name = async_match.group(1)
        barrier_id = 0 if slot_name == "tail0" else int(slot_name[-1])
        end_barrier = next_marker
        wait_label, spin_label, done_label = range(next_label, next_label + 3)
        next_label += 3
        output.extend([
            f"  // Ping-pong release for operand slot {barrier_id}.",
            "  setp.eq.u32 %p138, %r0, 0;",
            f"  @%p138 mbarrier.arrive.release.cta.shared::cta.b64 _, [%r{288 + barrier_id}];",
            "  setp.eq.u32 %p139, %r0, 128;",
            f"  @!%p139 bra $L{done_label};",
            f"  $L{wait_label}:",
            f"  $L{spin_label}:",
            ("  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 %p140, "
             f"[%r{288 + barrier_id}], %r{291 + barrier_id};"),
            f"  @!%p140 bra $L{spin_label};",
            f"  xor.b32 %r{291 + barrier_id}, %r{291 + barrier_id}, 1;",
            f"  $L{done_label}:",
            "",
        ])

        # Copy the following TMA block, electing lane 0 of WG1 (tid 128)
        # instead of lane 0 of WG0. All address operands are CTA-uniform.
        async_end = next(
            (i for i in range(next_marker + 1, len(lines))
             if lines[i].lstrip().startswith("// PTX template")
             or lines[i].lstrip().startswith("// END:")),
            len(lines),
        )
        async_block = lines[next_marker:async_end]
        guards = 0
        for i, line in enumerate(async_block):
            guard = _GUARD.match(line)
            if guard:
                async_block[i] = f"{guard.group(1)}setp.ne.s32 {guard.group(2)}, {guard.group(3)}, 128;"
                guards += 1
        if guards != 1:
            raise RuntimeError(f"expected one TMA issuer guard, found {guards}")
        output.extend(async_block)
        cursor = async_end
        changed += 1

    if verbose:
        print(f"converted {changed} operand releases to asymmetric WG ping-pong")
    return "\n".join(output) + ("\n" if ptx.endswith("\n") else ""), changed


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:3])
    result, count = transform(source.read_text(), verbose=True)
    destination.write_text(result)
    print(f"wrote {destination} ({count} converted releases)")
