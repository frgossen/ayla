"""Tests for reusable GEMM scheduling pipelines."""

import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.gemm import schedule_register_matmul
from ayla.gpu_dialect import AsyncOp, AwaitOp, ModuleOp, MoveOp, gpu_dialect


def test_register_matmul_schedules_side_by_side_gemms():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @dual(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101, init: 0.0}
            matmul(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "gate"}
            matmul(%a: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "up"}
            elementwise mul(%up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "product"}
          }
        }"""))  # noqa: E501
    assert isinstance(module, ModuleOp)

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
            %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101, init: 0.0}
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
            async(%3: lock<wgmma>) collective=[x.wg, yz.grid] attrs={label: "gate_async"} matmul(%0: smem<64x16xbf16 swizzle="128B">, %1: smem<64x16xbf16 swizzle="128B">, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "gate"}
            await(%3: lock<wgmma>)
            async(%6: lock<mbar>) collective=[x.wg, yz.grid] attrs={label: "up_lhs_smem_async"} {
              move(%4: smem<64x16xbf16 swizzle="128B">, %a: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "up_lhs_smem", tma: True}
              move(%5: smem<64x16xbf16 swizzle="128B">, %up_w: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "up_rhs_smem", tma: True}
            }
            await(%6: lock<mbar>)
            async(%7: lock<wgmma>) collective=[x.wg, yz.grid] attrs={label: "up_async"} matmul(%4: smem<64x16xbf16 swizzle="128B">, %5: smem<64x16xbf16 swizzle="128B">, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "up"}
            await(%7: lock<wgmma>)
            elementwise mul(%up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "product"}
          }
        }""")  # noqa: E501


def test_register_matmul_keeps_multicast_and_local_loads_separate():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x16xbf16>, %b: gmem<64x16xbf16>) grid_dim=[1, 2, 1] block_dim=[128, 1, 1] cluster_dim=[1, 2, 1] {
            %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100, init: 0.0}
            matmul(%a: gmem<64x16xbf16>, %b: gmem<64x16xbf16>, %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=xyz.block attrs={label: "mm"}
          }
        }"""))

    schedule_register_matmul(
        label="mm",
        operand_stages=1,
        operand_k_tile=16,
        operand_move_attrs={"lhs": {"multicast": True}},
    ).run(module)

    load_groups = [
        op for op in find_ops_of_type(module, AsyncOp)
        if op.body.ops and isinstance(op.body.ops[0], MoveOp)
    ]
    assert len(load_groups) == 2
    assert [group.is_tma_multicast for group in load_groups] == [True, False]
    assert load_groups[0].collective != load_groups[1].collective


def test_second_hero_ignores_an_already_pipelined_singleton():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @second_hero(%prior: gmem<64x16xbf16>, %lhs: gmem<64x16xbf16>, %rhs: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %prior_smem: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 0}
            %prior_lock: lock<mbar> = alloc() attrs={id: 1}
            await(%prior_lock: lock<mbar>)
            async(%prior_lock: lock<mbar>) collective=xyz.block attrs={label: "prior_async"} move(%prior_smem: smem<64x16xbf16 swizzle="128B">, %prior: gmem<64x16xbf16>) collective=xyz.block attrs={label: "prior", tma: True}
            %out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 2, init: 0.0}
            matmul(%lhs: gmem<64x16xbf16>, %rhs: gmem<64x16xbf16>, %out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=xyz.block attrs={label: "second"}
          }
        }"""))

    schedule_register_matmul(
        label="second", operand_stages=1, operand_k_tile=16,
    ).run(module)

    asyncs = find_ops_of_type(module, AsyncOp)
    [prior] = [op for op in asyncs if op.attrs.get("label") == "prior_async"]
    assert len(prior.body.ops) == 1
    [prior_await] = [
        op for op in find_ops_of_type(module, AwaitOp)
        if op.lock is prior.lock
    ]
    kernel = module.body.ops[0]
    assert kernel.body.ops.index(prior_await) < kernel.body.ops.index(prior)
    assert any(
        len(op.body.ops) == 2
        and op.attrs.get("label") == "second_lhs_smem_async"
        for op in asyncs
    )
