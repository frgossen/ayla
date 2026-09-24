"""Tests for reusable GEMM scheduling pipelines."""

import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.gemm import schedule_register_matmul
from ayla.gpu_dialect import (
    AsyncOp,
    AwaitOp,
    ForOp,
    MatmulOp,
    MoveOp,
    SmemTensorType,
    gpu_dialect,
)


def _side_by_side_module():
    return gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @pair(%a: gmem<64x64xbf16>, %first_w: gmem<64x64xbf16>, %second_w: gmem<64x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %first_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            %second_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101, init: 0.0}
            matmul(%a: gmem<64x64xbf16>, %first_w: gmem<64x64xbf16>, %first_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "first"}
            matmul(%a: gmem<64x64xbf16>, %second_w: gmem<64x64xbf16>, %second_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "second"}
          }
        }"""))  # noqa: E501


def test_register_matmul_fuses_independent_selected_heroes():
    module = _side_by_side_module()

    schedule_register_matmul(
        label="first|second",
        operand_stages=1,
        operand_k_tile=16,
    ).run(module)

    k_loops = [
        op for op in find_ops_of_type(module, ForOp)
        if op.attrs.get("label") == "k"
    ]
    assert len(k_loops) == 1
    matmuls = find_ops_of_type(k_loops[0], MatmulOp)
    assert len(matmuls) == 2
    assert matmuls[0].lhs is matmuls[1].lhs
    assert matmuls[0].rhs is not matmuls[1].rhs
    assert matmuls[0].dst is not matmuls[1].dst

    loads = [
        op for op in find_ops_of_type(k_loops[0], MoveOp)
        if isinstance(op.dst.type, SmemTensorType)
    ]
    assert len(loads) == 3
    assert {op.attrs.get("label") for op in loads} == {
        "first_lhs_smem",
        "first_rhs_smem",
        "second_rhs_smem",
    }
    asyncs = find_ops_of_type(k_loops[0], AsyncOp)
    assert sorted(len(op.body.ops) for op in asyncs) == [2, 3]
    assert len(find_ops_of_type(k_loops[0], AwaitOp)) == 2


def test_register_matmul_pipeline_keeps_epilogue_before_unequal_tail():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @tail(%a: gmem<64x96xbf16>, %short_w: gmem<64x64xbf16>, %long_w: gmem<64x96xbf16>) block_dim=[128, 1, 1] {
            %prefix: gmem<64x64xbf16 strides=[96, 1]> = tile(%a: gmem<64x96xbf16>, 0, 0)
            %short_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            %long_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101, init: 0.0}
            matmul(%prefix: gmem<64x64xbf16 strides=[96, 1]>, %short_w: gmem<64x64xbf16>, %short_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "short"}
            elementwise sigmoid(%short_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %short_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "short_epilogue"}
            matmul(%a: gmem<64x96xbf16>, %long_w: gmem<64x96xbf16>, %long_out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "long"}
          }
        }"""))  # noqa: E501

    schedule_register_matmul(
        label="short|long",
        operand_stages=1,
        operand_k_tile=16,
        epilogue_label_filter="short_epilogue",
    ).run(module)

    loops = [
        op for op in find_ops_of_type(module, ForOp)
        if op.attrs.get("label") == "k"
    ]
    assert [(loop.lb, loop.ub) for loop in loops] == [(0, 4), (4, 6)]
    [epilogue] = [
        op for op in module.body.ops[0].body.ops
        if op.attrs.get("label") == "short_epilogue"
    ]
    kernel_ops = module.body.ops[0].body.ops
    assert kernel_ops.index(loops[0]) < kernel_ops.index(epilogue)
    assert kernel_ops.index(epilogue) < kernel_ops.index(loops[1])


def test_register_matmul_schedules_side_by_side_gemms():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @dual(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %gate: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            %up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101, init: 0.0}
            matmul(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %gate: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "gate"}
            matmul(%a: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>, %up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "up"}
            elementwise mul(%up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "product"}
          }
        }"""))  # noqa: E501

    for label in ("gate", "up"):
        schedule_register_matmul(
            label=label,
            operand_stages=1,
            operand_k_tile=16,
            loop_label_prefix=label,
        ).run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @dual(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %gate: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            %up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101, init: 0.0}
            %0: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 0, label: "gate_lhs_smem"}
            %1: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 1, label: "gate_rhs_smem"}
            %2: lock<mbar> = alloc() attrs={id: 2}
            %3: lock<wgmma> = alloc() attrs={id: 4}
            async(%2: lock<mbar>) collective=[x.wg, yz.grid] attrs={label: "gate_lhs_smem_async"} {
              move(%0: smem<64x16xbf16 swizzle="128B">, %a: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "gate_lhs_smem", tma: True}
              move(%1: smem<64x16xbf16 swizzle="128B">, %gate_w: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "gate_rhs_smem", tma: True}
            }
            await(%2: lock<mbar>)
            %4: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 5, label: "up_lhs_smem"}
            %5: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 6, label: "up_rhs_smem"}
            %6: lock<mbar> = alloc() attrs={id: 7}
            %7: lock<wgmma> = alloc() attrs={id: 9}
            async(%3: lock<wgmma>) collective=[x.wg, yz.grid] attrs={label: "gate_async"} matmul(%0: smem<64x16xbf16 swizzle="128B">, %1: smem<64x16xbf16 swizzle="128B">, %gate: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "gate"}
            await(%3: lock<wgmma>)
            async(%6: lock<mbar>) collective=[x.wg, yz.grid] attrs={label: "up_lhs_smem_async"} {
              move(%4: smem<64x16xbf16 swizzle="128B">, %a: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "up_lhs_smem", tma: True}
              move(%5: smem<64x16xbf16 swizzle="128B">, %up_w: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "up_rhs_smem", tma: True}
            }
            await(%6: lock<mbar>)
            async(%7: lock<wgmma>) collective=[x.wg, yz.grid] attrs={label: "up_async"} matmul(%4: smem<64x16xbf16 swizzle="128B">, %5: smem<64x16xbf16 swizzle="128B">, %up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "up"}
            await(%7: lock<wgmma>)
            elementwise mul(%up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate: wrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "product"}
          }
        }""")  # noqa: E501
