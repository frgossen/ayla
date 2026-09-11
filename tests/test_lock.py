"""Tests for the lock-based async model."""
import textwrap

from ayla.gpu_dialect import gpu_dialect


def test_lock_based_gemm_inner_loop():
    """Lock-based async matmul: multiple ops share one lock inside a loop."""
    module_str = textwrap.dedent("""\
        module {
          kernel @mm(%a: smem<128x64xbf16 swizzle="128B">, %b: smem<128x64xbf16 swizzle="128B">, %c: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %lock: lock<?> = alloc() attrs={id: 0}
            for (%ki: index = 0 to 4) {
              %a_ki: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%a: smem<128x64xbf16 swizzle="128B">, 0, %ki: index)
              %b_ki: smem<128x16xbf16 strides=[64, 1] swizzle="128B"> = tile(%b: smem<128x64xbf16 swizzle="128B">, 0, %ki: index)
              async(%lock: lock<?>) matmul(%a_ki: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %b_ki: smem<128x16xbf16 strides=[64, 1] swizzle="128B">, %c: wrmem<128x128xf32 warps=4 layout="wgmma_layout_0">) collective=[xy.block, z.thread]
            }
            await(%lock: lock<?>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    text = repr(module)
    assert text == module_str


def test_lock_based_tma_load():
    """Lock-based async move: two TMA loads share one lock."""
    module_str = textwrap.dedent("""\
        module {
          kernel @load(%a: gmem<128x64xbf16>, %sa: smem<128x64xbf16>, %sb: smem<128x64xbf16>) grid_dim=[1, 1, 1] block_dim=[128, 1, 1] {
            %lock: lock<?> = alloc() attrs={id: 0}
            async(%lock: lock<?>) move(%sa: smem<128x64xbf16>, %a: gmem<128x64xbf16>) attrs={tma: True}
            async(%lock: lock<?>) move(%sb: smem<128x64xbf16>, %a: gmem<128x64xbf16>) attrs={tma: True}
            await(%lock: lock<?>)
          }
        }""")

    module = gpu_dialect.parse(module_str)
    text = repr(module)
    assert text == module_str
