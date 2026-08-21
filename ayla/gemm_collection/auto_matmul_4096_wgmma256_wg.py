"""The auto_matmul_4096_wgmma256_wg GEMM kernel: builder + torch registration."""
import textwrap

from ..gpu_dialect import gpu_dialect
from ..passes.canonicalize import canonicalize
from ..passes.fuse_async import fuse_async
from ..passes.licm import licm
from ..passes.lower_to_async import lower_to_async
from ..passes.multi_buffer import multi_buffer
from ..passes.pipeline_loop import pipeline_loop
from ..passes.promote_buffers import promote_buffers
from ..passes.schedule_async import schedule_async
from ..passes.tile_matmul import tile_matmul
from ..passes.unroll_loop import unroll_loop
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call


def auto_matmul_4096_wgmma256_wg_module():
    module_str = textwrap.dedent("""\
        module {
          kernel @auto_matmul_4096_wgmma256_wg(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[32, 16, 1] block_dim=[128, 1, 1] {
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={init: 0.0, label: "mm"}
          }
        }""")
    module = gpu_dialect.parse(module_str)
    module = tile_matmul(
        module,
        tile_dims=["M", "N"], tile_sizes=[128, 256],
        tile_levels=["block.x", "block.y"], tile_labels=[None, None],
        label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["M"], tile_sizes=[64],
        tile_levels=["wg.x"], tile_labels=[None],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"],
        swizzle=None, layout="wgmma_layout_0", warps=4,
        tma=False, nc=False, target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"],
        label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None,
        tma=True, nc=True, label_filter="mm",
    )
    module = tile_matmul(
        module,
        tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["kk"],
        label_filter="mm",
    )
    module = multi_buffer(module, depth=4, label_filter="mm_lhs_smem")
    module = multi_buffer(module, depth=4, label_filter="mm_rhs_smem")
    module = unroll_loop(module, unroll_factor=4, label_filter="k", clone_allocs=False)
    module = canonicalize(module)
    module = lower_to_async(module)
    # Fuse each stage's lhs+rhs TMA loads onto one mbarrier BEFORE pipelining,
    # while load and await are still adjacent (post-pipeline the await moves to
    # the loop top and can't be remapped to the fused lock).
    module = fuse_async(module)
    for i in range(4):
      module = licm(module)
      module = canonicalize(module)

    module = pipeline_loop(module, label_filter_loop="k", label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async", pipelining_offset=1)

    module = unroll_loop(module, unroll_factor=4, label_filter=r"kk_iter\d")
    module = fuse_async(module)


    for i in range(4):
      module = licm(module)
      module = canonicalize(module)


    module = schedule_async(module)
    module = canonicalize(module)

    return module


def _register_auto_matmul_4096_wgmma256_wg() -> None:
    module = auto_matmul_4096_wgmma256_wg_module()
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT])


_register_auto_matmul_4096_wgmma256_wg()
