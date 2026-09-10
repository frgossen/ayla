import textwrap

import pytest

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.schedule_output_epilogue import schedule_output_epilogue


MODULE = textwrap.dedent("""\
    module {
      kernel @k(%a: smem<1x1xf32>, %out0: gmem<2x1xbf16>, %out1: gmem<2x1xbf16>) {
        %store0: lock<bulk> = alloc() attrs={id: 0}
        %wg1: lock<wgmma> = alloc() attrs={id: 1}
        %store1: lock<bulk> = alloc() attrs={id: 2}
        %acc0: wrmem<1x4xf32 layout="mma"> = alloc() attrs={id: 3}
        %acc0_head: wrmem<1x3xf32 layout="mma"> = tile(%acc0: wrmem<1x4xf32 layout="mma">, 0, 0)
        %acc0_tail: wrmem<1x1xf32 layout="mma"> = tile(%acc0: wrmem<1x4xf32 layout="mma">, 0, 3)
        %bf0: wrmem<1x4xbf16 layout="mma"> = alloc() attrs={id: 4}
        %bf0_head: wrmem<1x3xbf16 layout="mma"> = tile(%bf0: wrmem<1x4xbf16 layout="mma">, 0, 0)
        %bf0_tail: wrmem<1x1xbf16 layout="mma"> = tile(%bf0: wrmem<1x4xbf16 layout="mma">, 0, 3)
        %smem0: smem<2x4xbf16> = alloc() attrs={id: 5}
        %smem0_wg: smem<1x4xbf16 strides=[4, 1]> = tile(%smem0: smem<2x4xbf16>, 0, 0)
        %smem0_head: smem<1x3xbf16 strides=[4, 1]> = tile(%smem0_wg: smem<1x4xbf16 strides=[4, 1]>, 0, 0)
        %smem0_tail: smem<1x1xbf16 strides=[4, 1]> = tile(%smem0_wg: smem<1x4xbf16 strides=[4, 1]>, 0, 3)
        %smem0_block_tail: smem<2x1xbf16 strides=[4, 1]> = tile(%smem0: smem<2x4xbf16>, 0, 3)
        elementwise(%acc0_head: wrmem<1x3xf32 layout="mma">, %acc0_head: wrmem<1x3xf32 layout="mma">) attrs={fn: "sigmoid", label: "epi"}
        elementwise(%acc0_tail: wrmem<1x1xf32 layout="mma">, %acc0_tail: wrmem<1x1xf32 layout="mma">) attrs={fn: "sigmoid", label: "epi"}
        convert(%bf0_head: wrmem<1x3xbf16 layout="mma">, %acc0_head: wrmem<1x3xf32 layout="mma">)
        convert(%bf0_tail: wrmem<1x1xbf16 layout="mma">, %acc0_tail: wrmem<1x1xf32 layout="mma">)
        move(%smem0_head: smem<1x3xbf16 strides=[4, 1]>, %bf0_head: wrmem<1x3xbf16 layout="mma">)
        move(%smem0_tail: smem<1x1xbf16 strides=[4, 1]>, %bf0_tail: wrmem<1x1xbf16 layout="mma">)
        async(%store0: lock<bulk>) move(%out0: gmem<2x1xbf16>, %smem0_block_tail: smem<2x1xbf16 strides=[4, 1]>)
        await(%store0: lock<bulk>)
        %mma1: wrmem<1x1xf32 layout="mma"> = alloc() attrs={id: 6}
        async(%wg1: lock<wgmma>) matmul(%a: smem<1x1xf32>, %a: smem<1x1xf32>, %mma1: wrmem<1x1xf32 layout="mma">)
        await(%wg1: lock<wgmma>)
        %acc1: wrmem<1x4xf32 layout="mma"> = alloc() attrs={id: 7}
        %acc1_tail: wrmem<1x1xf32 layout="mma"> = tile(%acc1: wrmem<1x4xf32 layout="mma">, 0, 3)
        elementwise(%acc1_tail: wrmem<1x1xf32 layout="mma">, %acc1_tail: wrmem<1x1xf32 layout="mma">) attrs={fn: "sigmoid", label: "epi"}
        %bf1: wrmem<1x4xbf16 layout="mma"> = alloc() attrs={id: 8}
        %bf1_tail: wrmem<1x1xbf16 layout="mma"> = tile(%bf1: wrmem<1x4xbf16 layout="mma">, 0, 3)
        convert(%bf1_tail: wrmem<1x1xbf16 layout="mma">, %acc1_tail: wrmem<1x1xf32 layout="mma">)
        %smem1: smem<2x4xbf16> = alloc() attrs={id: 9}
        %smem1_wg: smem<1x4xbf16 strides=[4, 1]> = tile(%smem1: smem<2x4xbf16>, 0, 0)
        %smem1_tail: smem<1x1xbf16 strides=[4, 1]> = tile(%smem1_wg: smem<1x4xbf16 strides=[4, 1]>, 0, 3)
        %smem1_block_tail: smem<2x1xbf16 strides=[4, 1]> = tile(%smem1: smem<2x4xbf16>, 0, 3)
        move(%smem1_tail: smem<1x1xbf16 strides=[4, 1]>, %bf1_tail: wrmem<1x1xbf16 layout="mma">)
        async(%store1: lock<bulk>) move(%out1: gmem<2x1xbf16>, %smem1_block_tail: smem<2x1xbf16 strides=[4, 1]>)
        await(%store1: lock<bulk>)
      }
    }""")  # noqa: E501


EXPECTED = textwrap.dedent("""\
    module {
      kernel @k(%a: smem<1x1xf32>, %out0: gmem<2x1xbf16>, %out1: gmem<2x1xbf16>) {
        %store0: lock<bulk> = alloc() attrs={id: 0}
        %wg1: lock<wgmma> = alloc() attrs={id: 1}
        %store1: lock<bulk> = alloc() attrs={id: 2}
        %acc0: wrmem<1x4xf32 layout="mma"> = alloc() attrs={id: 3}
        %acc0_head: wrmem<1x3xf32 layout="mma"> = tile(%acc0: wrmem<1x4xf32 layout="mma">, 0, 0)
        %acc0_tail: wrmem<1x1xf32 layout="mma"> = tile(%acc0: wrmem<1x4xf32 layout="mma">, 0, 3)
        %bf0: wrmem<1x4xbf16 layout="mma"> = alloc() attrs={id: 4}
        %bf0_head: wrmem<1x3xbf16 layout="mma"> = tile(%bf0: wrmem<1x4xbf16 layout="mma">, 0, 0)
        %bf0_tail: wrmem<1x1xbf16 layout="mma"> = tile(%bf0: wrmem<1x4xbf16 layout="mma">, 0, 3)
        %smem0: smem<2x4xbf16> = alloc() attrs={id: 5}
        %smem0_wg: smem<1x4xbf16> = tile(%smem0: smem<2x4xbf16>, 0, 0)
        %smem0_head: smem<1x3xbf16 strides=[4, 1]> = tile(%smem0_wg: smem<1x4xbf16>, 0, 0)
        %smem0_tail: smem<1x1xbf16 strides=[4, 1]> = tile(%smem0_wg: smem<1x4xbf16>, 0, 3)
        %smem0_block_tail: smem<2x1xbf16 strides=[4, 1]> = tile(%smem0: smem<2x4xbf16>, 0, 3)
        elementwise(%acc0_head: wrmem<1x3xf32 layout="mma">, %acc0_head: wrmem<1x3xf32 layout="mma">) attrs={fn: "sigmoid", label: "epi"}
        convert(%bf0_head: wrmem<1x3xbf16 layout="mma">, %acc0_head: wrmem<1x3xf32 layout="mma">)
        move(%smem0_head: smem<1x3xbf16 strides=[4, 1]>, %bf0_head: wrmem<1x3xbf16 layout="mma">)
        %mma1: wrmem<1x1xf32 layout="mma"> = alloc() attrs={id: 6}
        async(%wg1: lock<wgmma>) matmul(%a: smem<1x1xf32>, %a: smem<1x1xf32>, %mma1: wrmem<1x1xf32 layout="mma">)
        elementwise(%acc0_tail: wrmem<1x1xf32 layout="mma">, %acc0_tail: wrmem<1x1xf32 layout="mma">) attrs={fn: "sigmoid", label: "epi"}
        convert(%bf0_tail: wrmem<1x1xbf16 layout="mma">, %acc0_tail: wrmem<1x1xf32 layout="mma">)
        move(%smem0_tail: smem<1x1xbf16 strides=[4, 1]>, %bf0_tail: wrmem<1x1xbf16 layout="mma">)
        async(%store0: lock<bulk>) move(%out0: gmem<2x1xbf16>, %smem0_block_tail: smem<2x1xbf16 strides=[4, 1]>)
        await(%store0: lock<bulk>)
        await(%wg1: lock<wgmma>)
        %acc1: wrmem<1x4xf32 layout="mma"> = alloc() attrs={id: 7}
        %acc1_tail: wrmem<1x1xf32 layout="mma"> = tile(%acc1: wrmem<1x4xf32 layout="mma">, 0, 3)
        elementwise(%acc1_tail: wrmem<1x1xf32 layout="mma">, %acc1_tail: wrmem<1x1xf32 layout="mma">) attrs={fn: "sigmoid", label: "epi"}
        %bf1: wrmem<1x4xbf16 layout="mma"> = alloc() attrs={id: 8}
        %bf1_tail: wrmem<1x1xbf16 layout="mma"> = tile(%bf1: wrmem<1x4xbf16 layout="mma">, 0, 3)
        convert(%bf1_tail: wrmem<1x1xbf16 layout="mma">, %acc1_tail: wrmem<1x1xf32 layout="mma">)
        %smem1: smem<2x4xbf16> = alloc() attrs={id: 9}
        %smem1_wg: smem<1x4xbf16> = tile(%smem1: smem<2x4xbf16>, 0, 0)
        %smem1_tail: smem<1x1xbf16 strides=[4, 1]> = tile(%smem1_wg: smem<1x4xbf16>, 0, 3)
        %smem1_block_tail: smem<2x1xbf16 strides=[4, 1]> = tile(%smem1: smem<2x4xbf16>, 0, 3)
        move(%smem1_tail: smem<1x1xbf16 strides=[4, 1]>, %bf1_tail: wrmem<1x1xbf16 layout="mma">)
        async(%store1: lock<bulk>) move(%out1: gmem<2x1xbf16>, %smem1_block_tail: smem<2x1xbf16 strides=[4, 1]>)
        await(%store1: lock<bulk>)
      }
    }""")  # noqa: E501


def test_moves_complete_tail_dataflow_into_next_wgmma_window():
    module = gpu_dialect.parse(MODULE)
    schedule_output_epilogue(module, label_filter="epi")

    assert repr(module) == EXPECTED


def test_rejects_a_wgmma_that_reuses_the_tail_accumulator():
    module = gpu_dialect.parse(MODULE.replace(
        "%mma1: wrmem<1x1xf32 layout=\"mma\">)",
        "%acc0_tail: wrmem<1x1xf32 layout=\"mma\">)",
    ))
    with pytest.raises(ValueError, match="share an accumulator"):
        schedule_output_epilogue(module, label_filter="epi")
