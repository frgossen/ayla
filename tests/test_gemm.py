"""Tests for reusable GEMM scheduling pipelines."""

import textwrap

from ayla.base.ir import find_ops_of_type
from ayla.gemm_collection.gemm import schedule_matmul, schedule_register_matmul
from ayla.gpu_dialect import AsyncOp, ModuleOp, MoveOp, gpu_dialect
from ayla.ptx_codegen import make_ptx_kernel


def test_tiled_schedule_maps_two_output_dimensions_to_warp_groups():
    pipeline = schedule_tiled_matmul(
        label="mm",
        output_m_tile=128,
        output_n_tile=160,
        warpgroup_m_tile=64,
        warpgroup_n_tile=80,
        operand_stages=1,
    )

    warpgroup_tiling = pipeline.passes[0]
    assert isinstance(warpgroup_tiling, TileMatmul)
    assert warpgroup_tiling.op_dims == ["M", "N"]
    assert warpgroup_tiling.tile_sizes == [64, 80]


def test_schedule_matmul_initializes_accumulator_from_broadcast_memory():
    module = gpu_dialect.parse(textwrap.dedent("""\
        module {
          kernel @addmm(%bias: gmem<128x64xbf16 strides=[0, 1]>, %a: gmem<128x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<128x64xbf16>) {
            matmul(%a: gmem<128x64xbf16>, %b: gmem<64x64xbf16>, %c: gmem<128x64xbf16>, %bias: gmem<128x64xbf16 strides=[0, 1]>) collective=xyz.grid attrs={label: "mm"}
          }
        }"""))

    module = schedule_matmul(
        module,
        output_n_tile=64,
        warpgroup_n_tile=64,
        operand_stages=1,
        output_tma_inner_size=64,
    )
    ptx = make_ptx_kernel(module).ptx

    assert ptx.count("ld.global.b32") == 16
    assert "add.f32" not in ptx
    assert ptx.index("ld.global.b32") < ptx.index("wgmma.mma_async")


def test_register_matmul_schedules_side_by_side_gemms():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @dual(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100}
            %gate_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate_init: rmem<f32>)
            %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101}
            %up_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up_init: rmem<f32>)
            matmul(%a: gmem<64x16xbf16>, %gate_w: gmem<64x16xbf16>, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "gate"}
            matmul(%a: gmem<64x16xbf16>, %up_w: gmem<64x16xbf16>, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={label: "up"}
            elementwise mul(%up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "product"}
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
            %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100}
            %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 101}
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
            async(%3: lock<wgmma>) collective=[x.wg, yz.grid] attrs={label: "gate_async"} matmul(%0: smem<64x16xbf16 swizzle="128B">, %1: smem<64x16xbf16 swizzle="128B">, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={accumulate: False, label: "gate"}
            await(%3: lock<wgmma>)
            async(%6: lock<mbar>) collective=[x.wg, yz.grid] attrs={label: "up_lhs_smem_async"} {
              move(%4: smem<64x16xbf16 swizzle="128B">, %a: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "up_lhs_smem", tma: True}
              move(%5: smem<64x16xbf16 swizzle="128B">, %up_w: gmem<64x16xbf16>) collective=[x.wg, yz.grid] attrs={label: "up_rhs_smem", tma: True}
            }
            await(%6: lock<mbar>)
            async(%7: lock<wgmma>) collective=[x.wg, yz.grid] attrs={label: "up_async"} matmul(%4: smem<64x16xbf16 swizzle="128B">, %5: smem<64x16xbf16 swizzle="128B">, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=[x.wg, yz.grid] attrs={accumulate: False, label: "up"}
            await(%7: lock<wgmma>)
            elementwise mul(%up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %up: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %gate: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "product"}
          }
        }""")  # noqa: E501


def test_register_matmul_fuses_multicast_and_local_load_completion():
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @mm(%a: gmem<64x16xbf16>, %b: gmem<64x16xbf16>) grid_dim=[1, 2, 1] block_dim=[128, 1, 1] cluster_dim=[1, 2, 1] {
            %c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 100}
            %c_init: rmem<f32> = constant() attrs={value: 0.0}
            elementwise fill(%c: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">, %c_init: rmem<f32>)
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
    assert len(load_groups) == 1
    [group] = load_groups
    assert group.is_tma_multicast
    assert len(group.body.ops) == 2
    lhs_move, rhs_move = group.body.ops
    assert isinstance(lhs_move, MoveOp)
    assert isinstance(rhs_move, MoveOp)
    assert [lhs_move.multicast, rhs_move.multicast] == [True, False]
    assert group.collective == lhs_move.collective
    assert group.collective != rhs_move.collective


def test_second_hero_ignores_an_already_pipelined_singleton():
    # The prior load was pipelined as a singleton, so its await precedes it.
    # Scheduling the second GEMM leaves it as is and fuses only the new loads.
    module = gpu_dialect.parse_module(textwrap.dedent("""\
        module {
          kernel @second_hero(%prior: gmem<64x16xbf16>, %lhs: gmem<64x16xbf16>, %rhs: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %prior_smem: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 100}
            %prior_lock: lock<mbar> = alloc() attrs={id: 101}
            await(%prior_lock: lock<mbar>)
            async(%prior_lock: lock<mbar>) collective=xyz.block attrs={label: "prior_async"} move(%prior_smem: smem<64x16xbf16 swizzle="128B">, %prior: gmem<64x16xbf16>) collective=xyz.block attrs={label: "prior", tma: True}
            %out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 102, init: 0.0}
            matmul(%lhs: gmem<64x16xbf16>, %rhs: gmem<64x16xbf16>, %out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=xyz.block attrs={label: "second"}
          }
        }"""))  # noqa: E501

    schedule_register_matmul(
        label="second", operand_stages=1, operand_k_tile=16,
    ).run(module)

    assert repr(module) == textwrap.dedent("""\
        module {
          kernel @second_hero(%prior: gmem<64x16xbf16>, %lhs: gmem<64x16xbf16>, %rhs: gmem<64x16xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %prior_smem: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 100}
            %prior_lock: lock<mbar> = alloc() attrs={id: 101}
            await(%prior_lock: lock<mbar>)
            async(%prior_lock: lock<mbar>) collective=xyz.block attrs={label: "prior_async"} move(%prior_smem: smem<64x16xbf16 swizzle="128B">, %prior: gmem<64x16xbf16>) collective=xyz.block attrs={label: "prior", tma: True}
            %out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={id: 102, init: 0.0}
            %0: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 0, label: "second_lhs_smem"}
            %1: smem<64x16xbf16 swizzle="128B"> = alloc() attrs={id: 1, label: "second_rhs_smem"}
            %2: lock<mbar> = alloc() attrs={id: 2}
            %3: lock<wgmma> = alloc() attrs={id: 4}
            async(%2: lock<mbar>) collective=xyz.block attrs={label: "second_lhs_smem_async"} {
              move(%0: smem<64x16xbf16 swizzle="128B">, %lhs: gmem<64x16xbf16>) collective=xyz.block attrs={label: "second_lhs_smem", tma: True}
              move(%1: smem<64x16xbf16 swizzle="128B">, %rhs: gmem<64x16xbf16>) collective=xyz.block attrs={label: "second_rhs_smem", tma: True}
            }
            await(%2: lock<mbar>)
            async(%3: lock<wgmma>) collective=xyz.block attrs={label: "second_async"} matmul(%0: smem<64x16xbf16 swizzle="128B">, %1: smem<64x16xbf16 swizzle="128B">, %out: wgrmem<64x64xf32 warps=4 layout="wgmma_layout_0">) collective=xyz.block attrs={label: "second"}
            await(%3: lock<wgmma>)
          }
        }""")  # noqa: E501
