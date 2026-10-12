"""Tests for scheduling asynchronous work around independent operations."""
import textwrap

import pytest

from ayla.gpu_dialect import ModuleOp, gpu_dialect
from ayla.passes.lower_to_async import lower_to_async
from ayla.passes.schedule_async import ScheduleAsync, schedule_async


def test_bulk_await_sinks_until_shared_source_is_overwritten():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @test(%g: gmem<4x4xf32>, %a: rmem<1x1xf32>, %b: rmem<1x1xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %s: smem<4x4xf32> = alloc() attrs={id: 0}
            %r: rmem<4x4xf32> = alloc() attrs={id: 1}
            %acc: rmem<1x1xf32> = alloc() attrs={id: 2, init: 0.0}
            move(%g: gmem<4x4xf32>, %s: smem<4x4xf32>) attrs={label: "store", tma: true}
            matmul(%a: rmem<1x1xf32>, %b: rmem<1x1xf32>, %acc: rmem<1x1xf32>) attrs={label: "next_compute"}
            move(%s: smem<4x4xf32>, %r: rmem<4x4xf32>) attrs={label: "reuse"}
          }
        }"""))
    assert isinstance(module, ModuleOp)
    text = repr(schedule_async(lower_to_async(module)))
    store = text.index('label: "store_async"')
    compute = text.index('label: "next_compute"')
    await_store = text.index('label: "store_await"')
    reuse = text.index('label: "reuse"')
    assert store < compute < await_store < reuse



@pytest.mark.parametrize("pending_groups", [0, 1])
def test_wgmma_pending_groups_keeps_refills_below_newer_issues(pending_groups):
    acc = 'wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">'
    ops = {
        "load0": (
            "async(%load0: lock<mbar>) move(%s0: smem<64x16xbf16>, "
            "%a0: gmem<64x16xbf16>) attrs={tma: True}"
        ),
        "load1": (
            "async(%load1: lock<mbar>) move(%s1: smem<64x16xbf16>, "
            "%a1: gmem<64x16xbf16>) attrs={tma: True}"
        ),
        "mma0": (
            "async(%mma0: lock<wgmma>) matmul(%s0: smem<64x16xbf16>, "
            f"%s0: smem<64x16xbf16>, %acc: {acc})"
        ),
        "mma1": (
            "async(%mma1: lock<wgmma>) matmul(%s1: smem<64x16xbf16>, "
            f"%s1: smem<64x16xbf16>, %acc: {acc})"
        ),
        "wait0": "await(%load0: lock<mbar>)",
        "wait1": "await(%load1: lock<mbar>)",
    }

    def module(*body: str) -> str:
        return textwrap.dedent(f"""\
            module {{
              kernel @k(%a0: gmem<64x16xbf16>, %a1: gmem<64x16xbf16>, %acc: {acc}) {{
                %s0: smem<64x16xbf16> = alloc()
                %s1: smem<64x16xbf16> = alloc()
                %load0: lock<mbar> = alloc()
                %load1: lock<mbar> = alloc()
                %mma0: lock<wgmma> = alloc()
                %mma1: lock<wgmma> = alloc()
                for (%k: index = 0 to 4) {{
            """) + "".join(
            f"      {ops[name]}\n" for name in body
        ) + "    }\n  }\n}"  # noqa: E501

    source = module("wait0", "mma0", "load1", "wait1", "mma1", "load0")
    scheduled = ScheduleAsync(wgmma_pending_groups=pending_groups).run(
        gpu_dialect.parse_module(source),
    )

    # Fully hoisted, each refill directly follows the group it conflicts with
    # (this iteration's or the previous one's), so its wait drains the queue.
    expected = source if pending_groups else module(
        "load1", "wait0", "mma0", "load0", "wait1", "mma1",
    )
    assert repr(scheduled) == expected
