"""Tests for reusable GEMM scheduling pipelines."""

import textwrap

from ayla.gemm_collection.gemm import schedule_register_matmul
from ayla.gpu_dialect import ModuleOp, gpu_dialect


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
