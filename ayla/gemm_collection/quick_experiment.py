"""Quick-experiment kernel: a frozen snapshot of the IR that
``auto_matmul_4096_wgmma256_wg_module`` produces, parsed as-is with no passes.

Edit ``QUICK_EXPERIMENT_IR`` below for quick-and-dirty IR tweaks and see the
perf effect without re-running the build pipeline. To regenerate the baseline::

    from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg import \
        auto_matmul_4096_wgmma256_wg_module as f
    print(repr(f()).replace("@auto_matmul_4096_wgmma256_wg", "@quick_experiment"))
"""
from ..gpu_dialect import gpu_dialect
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call

QUICK_EXPERIMENT_IR = r'''
module {
  kernel @quick_experiment(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[128, 1, 1] num_warp_groups=2 {
    %c0: index = index_constant() attrs={value: 0}
    %0: smem_tensor<256x64xbf16 swizzle="128B"> = alloc()
    %1: lock<mbar> = alloc()
    %2: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
    %c3: index = index_constant() attrs={value: 3}
    %3: smem_tensor<256x64xbf16 swizzle="128B"> = alloc()
    %4: lock<mbar> = alloc()
    %5: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
    %c2: index = index_constant() attrs={value: 2}
    %6: smem_tensor<256x64xbf16 swizzle="128B"> = alloc()
    %7: lock<mbar> = alloc()
    %8: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
    %c1: index = index_constant() attrs={value: 1}
    %9: smem_tensor<256x64xbf16 swizzle="128B"> = alloc()
    %10: lock<mbar> = alloc()
    %11: smem_tensor<128x64xbf16 swizzle="128B"> = alloc()
    %bx: index, %by: index, %bz: index = block_ids()
    %12: gmem_tensor<256x4096xbf16> = tile(%b: gmem_tensor<4096x4096xbf16>, %by: index, 0)
    %13: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %c3: index)
    %14: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %c2: index)
    %15: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %c1: index)
    %16: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %c0: index)
    %17: gmem_tensor<128x4096xbf16> = tile(%a: gmem_tensor<4096x4096xbf16>, %bx: index, 0)
    %18: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %c3: index)
    async(%1: lock<mbar>) attrs={label: "mm_lhs_smem_iter3_async", sync: "none"} {
      move(%2: smem_tensor<128x64xbf16 swizzle="128B">, %18: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter3", nc: True, tma: True}
      move(%0: smem_tensor<256x64xbf16 swizzle="128B">, %13: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter3", nc: True, tma: True}
    }
    %19: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %c2: index)
    async(%4: lock<mbar>) attrs={label: "mm_lhs_smem_iter2_async", sync: "none"} {
      move(%5: smem_tensor<128x64xbf16 swizzle="128B">, %19: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter2", nc: True, tma: True}
      move(%3: smem_tensor<256x64xbf16 swizzle="128B">, %14: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter2", nc: True, tma: True}
    }
    %20: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %c1: index)
    async(%7: lock<mbar>) attrs={label: "mm_lhs_smem_iter1_async", sync: "none"} {
      move(%8: smem_tensor<128x64xbf16 swizzle="128B">, %20: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter1", nc: True, tma: True}
      move(%6: smem_tensor<256x64xbf16 swizzle="128B">, %15: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter1", nc: True, tma: True}
    }
    %21: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %c0: index)
    async(%10: lock<mbar>) attrs={label: "mm_lhs_smem_iter0_async", sync: "none"} {
      move(%11: smem_tensor<128x64xbf16 swizzle="128B">, %21: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter0", nc: True, tma: True}
      move(%9: smem_tensor<256x64xbf16 swizzle="128B">, %16: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter0", nc: True, tma: True}
    }
    %22: gmem_tensor<128x4096xbf16> = tile(%c: gmem_tensor<4096x4096xbf16>, %bx: index, 0)
    %23: gmem_tensor<128x256xbf16 strides=[4096, 1]> = tile(%22: gmem_tensor<128x4096xbf16>, 0, %by: index)
    %wgx: index, %wgy: index, %wgz: index = warp_group_id()
    %24: gmem_tensor<64x256xbf16 strides=[4096, 1]> = tile(%23: gmem_tensor<128x256xbf16 strides=[4096, 1]>, %wgx: index, 0)
    %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0"> = alloc() attrs={init: 0.0}
    %26: smem_tensor<64x64xbf16 swizzle="128B"> = tile(%11: smem_tensor<128x64xbf16 swizzle="128B">, %wgx: index, 0)
    %27: smem_tensor<64x64xbf16 swizzle="128B"> = tile(%8: smem_tensor<128x64xbf16 swizzle="128B">, %wgx: index, 0)
    %28: smem_tensor<64x64xbf16 swizzle="128B"> = tile(%5: smem_tensor<128x64xbf16 swizzle="128B">, %wgx: index, 0)
    %29: smem_tensor<64x64xbf16 swizzle="128B"> = tile(%2: smem_tensor<128x64xbf16 swizzle="128B">, %wgx: index, 0)
    %30: lock<wgmma> = alloc()
    %31: lock<wgmma> = alloc()
    %32: lock<wgmma> = alloc()
    %33: lock<wgmma> = alloc()
    %c4: index = index_constant() attrs={value: 4}
    %34: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c0: index)
    %35: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c0: index)
    %36: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c1: index)
    %37: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c1: index)
    %38: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c2: index)
    %39: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c2: index)
    %40: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%26: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c3: index)
    %41: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%9: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c3: index)
    %42: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c0: index)
    %43: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c0: index)
    %44: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c1: index)
    %45: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c1: index)
    %46: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c2: index)
    %47: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c2: index)
    %48: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%27: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c3: index)
    %49: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%6: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c3: index)
    %50: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c0: index)
    %51: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c0: index)
    %52: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c1: index)
    %53: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c1: index)
    %54: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c2: index)
    %55: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c2: index)
    %56: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%28: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c3: index)
    %57: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%3: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c3: index)
    %58: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c0: index)
    %59: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c0: index)
    %60: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c1: index)
    %61: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c1: index)
    %62: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c2: index)
    %63: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c2: index)
    %64: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%29: smem_tensor<64x64xbf16 swizzle="128B">, 0, %c3: index)
    %65: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%0: smem_tensor<256x64xbf16 swizzle="128B">, 0, %c3: index)
    %c5: index = index_constant() attrs={value: 5}
    %c6: index = index_constant() attrs={value: 6}
    %c7: index = index_constant() attrs={value: 7}
    for (%k: index = 0 to 60 step 4) attrs={label: "k"} {
      %66: index = index_add(%k: index, %c7: index)
      %67: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %66: index)
      %68: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %66: index)
      %69: index = index_add(%k: index, %c6: index)
      %70: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %69: index)
      %71: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %69: index)
      %72: index = index_add(%k: index, %c5: index)
      %73: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %72: index)
      %74: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %72: index)
      %75: index = index_add(%k: index, %c4: index)
      %76: gmem_tensor<128x64xbf16 strides=[4096, 1]> = tile(%17: gmem_tensor<128x4096xbf16>, 0, %75: index)
      %77: gmem_tensor<256x64xbf16 strides=[4096, 1]> = tile(%12: gmem_tensor<256x4096xbf16>, 0, %75: index)
      await(%10: lock<mbar>)
      async(%30: lock<wgmma>) attrs={label: "mm_iter0_async_iter0"} {
        matmul(%34: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %35: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter0"}
        matmul(%36: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %37: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter1"}
        matmul(%38: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %39: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter2"}
        matmul(%40: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %41: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter3"}
      }
      await(%30: lock<wgmma>) attrs={arrive: 1}
      async(%10: lock<mbar>) attrs={label: "mm_lhs_smem_iter0_async", sync: "named", bar: 1} {
        move(%11: smem_tensor<128x64xbf16 swizzle="128B">, %76: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter0", nc: True, tma: True}
        move(%9: smem_tensor<256x64xbf16 swizzle="128B">, %77: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter0", nc: True, tma: True}
      }
      await(%7: lock<mbar>)
      async(%31: lock<wgmma>) attrs={label: "mm_iter1_async_iter0"} {
        matmul(%42: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %43: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter0"}
        matmul(%44: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %45: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter1"}
        matmul(%46: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %47: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter2"}
        matmul(%48: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %49: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter3"}
      }
      await(%31: lock<wgmma>) attrs={arrive: 2}
      async(%7: lock<mbar>) attrs={label: "mm_lhs_smem_iter1_async", sync: "named", bar: 2} {
        move(%8: smem_tensor<128x64xbf16 swizzle="128B">, %73: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter1", nc: True, tma: True}
        move(%6: smem_tensor<256x64xbf16 swizzle="128B">, %74: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter1", nc: True, tma: True}
      }
      await(%4: lock<mbar>)
      async(%32: lock<wgmma>) attrs={label: "mm_iter2_async_iter0"} {
        matmul(%50: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %51: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter0"}
        matmul(%52: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %53: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter1"}
        matmul(%54: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %55: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter2"}
        matmul(%56: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %57: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter3"}
      }
      await(%32: lock<wgmma>) attrs={arrive: 3}
      async(%4: lock<mbar>) attrs={label: "mm_lhs_smem_iter2_async", sync: "named", bar: 3} {
        move(%5: smem_tensor<128x64xbf16 swizzle="128B">, %70: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter2", nc: True, tma: True}
        move(%3: smem_tensor<256x64xbf16 swizzle="128B">, %71: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter2", nc: True, tma: True}
      }
      await(%1: lock<mbar>)
      async(%33: lock<wgmma>) attrs={label: "mm_iter3_async_iter0"} {
        matmul(%58: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %59: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter0"}
        matmul(%60: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %61: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter1"}
        matmul(%62: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %63: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter2"}
        matmul(%64: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %65: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter3"}
      }
      await(%33: lock<wgmma>) attrs={arrive: 4}
      async(%1: lock<mbar>) attrs={label: "mm_lhs_smem_iter3_async", sync: "named", bar: 4} {
        move(%2: smem_tensor<128x64xbf16 swizzle="128B">, %67: gmem_tensor<128x64xbf16 strides=[4096, 1]>) attrs={label: "mm_lhs_smem_iter3", nc: True, tma: True}
        move(%0: smem_tensor<256x64xbf16 swizzle="128B">, %68: gmem_tensor<256x64xbf16 strides=[4096, 1]>) attrs={label: "mm_rhs_smem_iter3", nc: True, tma: True}
      }
    }
    await(%10: lock<mbar>)
    async(%30: lock<wgmma>) attrs={label: "mm_iter0_async_iter0"} {
      matmul(%34: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %35: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter0"}
      matmul(%36: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %37: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter1"}
      matmul(%38: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %39: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter2"}
      matmul(%40: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %41: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter0_iter3"}
    }
    await(%30: lock<wgmma>)
    await(%7: lock<mbar>)
    async(%31: lock<wgmma>) attrs={label: "mm_iter1_async_iter0"} {
      matmul(%42: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %43: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter0"}
      matmul(%44: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %45: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter1"}
      matmul(%46: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %47: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter2"}
      matmul(%48: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %49: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter1_iter3"}
    }
    await(%31: lock<wgmma>)
    await(%4: lock<mbar>)
    async(%32: lock<wgmma>) attrs={label: "mm_iter2_async_iter0"} {
      matmul(%50: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %51: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter0"}
      matmul(%52: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %53: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter1"}
      matmul(%54: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %55: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter2"}
      matmul(%56: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %57: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter2_iter3"}
    }
    await(%32: lock<wgmma>)
    await(%1: lock<mbar>)
    async(%33: lock<wgmma>) attrs={label: "mm_iter3_async_iter0"} {
      matmul(%58: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %59: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter0"}
      matmul(%60: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %61: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter1"}
      matmul(%62: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %63: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter2"}
      matmul(%64: smem_tensor<64x16xbf16 strides=[64, 1] swizzle="128B">, %65: smem_tensor<256x16xbf16 strides=[64, 1] swizzle="128B">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_iter3_iter3"}
    }
    %78: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0"> = alloc()
    await(%33: lock<wgmma>)
    convert(%78: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">, %25: wrmem_tensor<64x256xf32 warps=4 layout="wgmma_layout_0">)
    move(%24: gmem_tensor<64x256xbf16 strides=[4096, 1]>, %78: wrmem_tensor<64x256xbf16 warps=4 layout="wgmma_layout_0">) attrs={label: "mm_dst_wrmem"}
  }
}
'''


def quick_experiment_module():
    return gpu_dialect.parse(QUICK_EXPERIMENT_IR.strip())


def _register_quick_experiment() -> None:
    module = quick_experiment_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(
        launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )


_register_quick_experiment()
