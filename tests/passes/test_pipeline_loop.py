"""Tests for the pipeline_loop pass."""

import textwrap

from ayla.gpu_dialect import gpu_dialect
from ayla.passes.pipeline_loop import pipeline_loop


def test_pipeline_reorders_across_inner_loop():
    """Pipelining a labeled load past an unlabeled inner for-loop.

    The hoisted `load` writes %buf1, while the inner loop only touches
    %buf0/%acc, so the reorder is legal. This exercises memory_effects'
    ForOp modeling: without it the loop's effects are unknown (None),
    treated as a worst-case conflict, and the legality check wrongly
    rejects the reorder with "cannot be proven independent"."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %buf0: smem<4x1xf32> = alloc() attrs={id: 0}
            %buf1: smem<4x1xf32> = alloc() attrs={id: 1}
            %acc: smem<4x1xf32> = alloc() attrs={id: 2}
            %acc_init: rmem<f32> = constant 0.0
            elementwise fill(%acc: smem<4x1xf32>, %acc_init: rmem<f32>)
            for (%k: index = 0 to 4) attrs={label: "k"} {
              for (%l: index = 0 to 2) {
                matmul(%buf0: smem<4x1xf32>, %buf0: smem<4x1xf32>, %acc: smem<4x1xf32>)
              }
              %a_k: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              move(%buf1: smem<4x1xf32>, %a_k: gmem<4x1xf32 strides=[4, 1]>) attrs={label: "load"}
            }
          }
        }""")  # noqa: E501
    module = gpu_dialect.parse_module(module_str)
    pipelined = pipeline_loop(
        module,
        label_filter_loop="k",
        label_filter_op="load",
        pipelining_offset=1,
    )
    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %buf0: smem<4x1xf32> = alloc() attrs={id: 0}
            %buf1: smem<4x1xf32> = alloc() attrs={id: 1}
            %acc: smem<4x1xf32> = alloc() attrs={id: 2}
            %acc_init: rmem<f32> = constant 0.0
            elementwise fill(%acc: smem<4x1xf32>, %acc_init: rmem<f32>)
            %c0: index = index_constant() attrs={value: 0}
            %a_k: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c0: index)
            move(%buf1: smem<4x1xf32>, %a_k: gmem<4x1xf32 strides=[4, 1]>) attrs={label: "load"}
            for (%k: index = 0 to 3) attrs={label: "k"} {
              %a_k1: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              for (%l: index = 0 to 2) {
                matmul(%buf0: smem<4x1xf32>, %buf0: smem<4x1xf32>, %acc: smem<4x1xf32>)
              }
              %c1: index = index_constant() attrs={value: 1}
              %0: index = index_add(%k: index, %c1: index)
              %a_k2: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %0: index)
              move(%buf1: smem<4x1xf32>, %a_k2: gmem<4x1xf32 strides=[4, 1]>) attrs={label: "load"}
            }
            %c3: index = index_constant() attrs={value: 3}
            %a_k3: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c3: index)
            for (%l1: index = 0 to 2) {
              matmul(%buf0: smem<4x1xf32>, %buf0: smem<4x1xf32>, %acc: smem<4x1xf32>)
            }
          }
        }""")  # noqa: E501
    assert repr(pipelined) == expected


def test_pipeline_queued_async_completion():
    module = gpu_dialect.parse_module(
        textwrap.dedent("""\
        module {
          kernel @test(%a: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) {
            %l0: lock<wgmma> = alloc()
            %l1: lock<wgmma> = alloc()
            for (%k: index = 0 to 4) attrs={label: "k"} {
              async(%l0: lock<wgmma>) attrs={label: "compute0"} matmul(%a: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
              await(%l0: lock<wgmma>)
              async(%l1: lock<wgmma>) attrs={label: "compute1"} matmul(%a: smem<64x16xbf16>, %b: smem<64x16xbf16>, %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">)
              await(%l1: lock<wgmma>)
            }
          }
        }""")
    )

    pipeline_loop(
        module,
        label_filter_loop="k",
        label_filter_op="compute.*",
        pipelining_offset=1,
    )

    text = repr(module)
    assert "await(%l1: lock<wgmma>) attrs={pending: 1}" in text
    assert "await(%l0: lock<wgmma>) attrs={pending: 1}" in text
    assert text.rfind("await(%l1: lock<wgmma>)") > text.rfind("for (%k")


def test_pipeline_matmul_smem_skeleton():
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<8x8xf32>, %b: gmem<8x8xf32>, %c: gmem<8x8xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %sa: smem<8x1xf32> = alloc() attrs={id: 0}
            %sb: smem<8x1xf32> = alloc() attrs={id: 1}
            %sa2: smem<8x1xf32> = alloc() attrs={id: 2}
            %sb2: smem<8x1xf32> = alloc() attrs={id: 3}
            %sc: smem<8x1xf32> = alloc() attrs={id: 4}
            %sc_init: rmem<f32> = constant 0.0
            elementwise fill(%sc: smem<8x1xf32>, %sc_init: rmem<f32>)
            for (%k: index = 0 to 4 step 2) {
              %a_k: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %k: index)
              %b_k: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %k: index)
              move(%sa: smem<8x1xf32>, %a_k: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              move(%sb: smem<8x1xf32>, %b_k: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              matmul(%sa: smem<8x1xf32>, %sb: smem<8x1xf32>, %sc: smem<8x1xf32>)
              %c1: index = index_constant() attrs={value: 1}
              %kp1: index = index_add(%k: index, %c1: index)
              %a_kp1: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %kp1: index)
              %b_kp1: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %kp1: index)
              move(%sa2: smem<8x1xf32>, %a_kp1: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              move(%sb2: smem<8x1xf32>, %b_kp1: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              matmul(%sa2: smem<8x1xf32>, %sb2: smem<8x1xf32>, %sc: smem<8x1xf32>)
            }
          }
        }""")  # noqa: E501

    module = gpu_dialect.parse_module(module_str)
    pipelined = pipeline_loop(
        module, label_filter_loop=".*", label_filter_op="load", pipelining_offset=1
    )
    text = repr(pipelined)
    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<8x8xf32>, %b: gmem<8x8xf32>, %c: gmem<8x8xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %sa: smem<8x1xf32> = alloc() attrs={id: 0}
            %sb: smem<8x1xf32> = alloc() attrs={id: 1}
            %sa2: smem<8x1xf32> = alloc() attrs={id: 2}
            %sb2: smem<8x1xf32> = alloc() attrs={id: 3}
            %sc: smem<8x1xf32> = alloc() attrs={id: 4}
            %sc_init: rmem<f32> = constant 0.0
            elementwise fill(%sc: smem<8x1xf32>, %sc_init: rmem<f32>)
            %c0: index = index_constant() attrs={value: 0}
            %a_k: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %c0: index)
            %b_k: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %c0: index)
            %c1: index = index_constant() attrs={value: 1}
            %kp1: index = index_add(%c0: index, %c1: index)
            %a_kp1: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %kp1: index)
            %b_kp1: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %kp1: index)
            move(%sa: smem<8x1xf32>, %a_k: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
            move(%sb: smem<8x1xf32>, %b_k: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
            move(%sa2: smem<8x1xf32>, %a_kp1: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
            move(%sb2: smem<8x1xf32>, %b_kp1: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
            for (%k: index = 0 to 2 step 2) {
              %a_k1: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %k: index)
              %b_k1: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %k: index)
              %c1_1: index = index_constant() attrs={value: 1}
              %kp1_1: index = index_add(%k: index, %c1_1: index)
              %a_kp1_1: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %kp1_1: index)
              %b_kp1_1: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %kp1_1: index)
              matmul(%sa: smem<8x1xf32>, %sb: smem<8x1xf32>, %sc: smem<8x1xf32>)
              matmul(%sa2: smem<8x1xf32>, %sb2: smem<8x1xf32>, %sc: smem<8x1xf32>)
              %c2: index = index_constant() attrs={value: 2}
              %0: index = index_add(%k: index, %c2: index)
              %a_k2: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %0: index)
              %b_k2: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %0: index)
              %c1_2: index = index_constant() attrs={value: 1}
              %kp1_2: index = index_add(%0: index, %c1_2: index)
              %a_kp1_2: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %kp1_2: index)
              %b_kp1_2: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %kp1_2: index)
              move(%sa: smem<8x1xf32>, %a_k2: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              move(%sb: smem<8x1xf32>, %b_k2: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              move(%sa2: smem<8x1xf32>, %a_kp1_2: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
              move(%sb2: smem<8x1xf32>, %b_kp1_2: gmem<8x1xf32 strides=[8, 1]>) attrs={label: "load"}
            }
            %c2_1: index = index_constant() attrs={value: 2}
            %a_k3: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %c2_1: index)
            %b_k3: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %c2_1: index)
            %c1_3: index = index_constant() attrs={value: 1}
            %kp1_3: index = index_add(%c2_1: index, %c1_3: index)
            %a_kp1_3: gmem<8x1xf32 strides=[8, 1]> = tile(%a: gmem<8x8xf32>, 0, %kp1_3: index)
            %b_kp1_3: gmem<8x1xf32 strides=[8, 1]> = tile(%b: gmem<8x8xf32>, 0, %kp1_3: index)
            matmul(%sa: smem<8x1xf32>, %sb: smem<8x1xf32>, %sc: smem<8x1xf32>)
            matmul(%sa2: smem<8x1xf32>, %sb2: smem<8x1xf32>, %sc: smem<8x1xf32>)
          }
        }""")  # noqa: E501
    assert text == expected


def test_pipeline_clones_matched_pure_ops_for_unlabeled_consumers():
    """Pipelining everything but a store, including unlabeled pure ops.

    The filter also matches the store's unlabeled ``tile`` (effective label
    ""). Being pure, it is cloned into both sections like any pure
    dependency, so the store keeps its own iteration's tile instead of being
    rejected for consuming a labeled result."""
    module_str = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>, %c: gmem<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %buf: smem<4x1xf32> = alloc() attrs={id: 0}
            for (%k: index = 0 to 4) attrs={label: "k"} {
              %a_k: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              move(%buf: smem<4x1xf32>, %a_k: gmem<4x1xf32 strides=[4, 1]>) attrs={label: "load"}
              %c_k: gmem<4x1xf32 strides=[4, 1]> = tile(%c: gmem<4x4xf32>, 0, %k: index)
              move(%c_k: gmem<4x1xf32 strides=[4, 1]>, %buf: smem<4x1xf32>) attrs={label: "store"}
            }
          }
        }""")  # noqa: E501
    module = gpu_dialect.parse_module(module_str)
    pipelined = pipeline_loop(
        module,
        label_filter_loop="k",
        label_filter_op="(?!store).*",
        pipelining_offset=1,
    )
    expected = textwrap.dedent("""\
        module {
          kernel @test(%a: gmem<4x4xf32>, %c: gmem<4x4xf32>) grid_dim=[1, 1, 1] block_dim=[1, 1, 1] {
            %buf: smem<4x1xf32> = alloc() attrs={id: 0}
            %c0: index = index_constant() attrs={value: 0}
            %a_k: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c0: index)
            %c_k: gmem<4x1xf32 strides=[4, 1]> = tile(%c: gmem<4x4xf32>, 0, %c0: index)
            move(%buf: smem<4x1xf32>, %a_k: gmem<4x1xf32 strides=[4, 1]>) attrs={label: "load"}
            for (%k: index = 0 to 3) attrs={label: "k"} {
              %a_k1: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %k: index)
              %c_k1: gmem<4x1xf32 strides=[4, 1]> = tile(%c: gmem<4x4xf32>, 0, %k: index)
              move(%c_k1: gmem<4x1xf32 strides=[4, 1]>, %buf: smem<4x1xf32>) attrs={label: "store"}
              %c1: index = index_constant() attrs={value: 1}
              %0: index = index_add(%k: index, %c1: index)
              %a_k2: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %0: index)
              %c_k2: gmem<4x1xf32 strides=[4, 1]> = tile(%c: gmem<4x4xf32>, 0, %0: index)
              move(%buf: smem<4x1xf32>, %a_k2: gmem<4x1xf32 strides=[4, 1]>) attrs={label: "load"}
            }
            %c3: index = index_constant() attrs={value: 3}
            %a_k3: gmem<4x1xf32 strides=[4, 1]> = tile(%a: gmem<4x4xf32>, 0, %c3: index)
            %c_k3: gmem<4x1xf32 strides=[4, 1]> = tile(%c: gmem<4x4xf32>, 0, %c3: index)
            move(%c_k3: gmem<4x1xf32 strides=[4, 1]>, %buf: smem<4x1xf32>) attrs={label: "store"}
          }
        }""")  # noqa: E501
    assert repr(pipelined) == expected
