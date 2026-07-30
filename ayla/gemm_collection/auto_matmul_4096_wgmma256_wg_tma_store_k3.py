"""Three-stage wgmma256 GEMM with pipelined TMA output stores."""
import textwrap

from ..gpu_dialect import gpu_dialect
from ..passes.canonicalize import canonicalize
from ..passes.fuse_async import fuse_async
from ..passes.licm import licm
from ..passes.lower_to_async import lower_to_async
from ..passes.pipeline_loop import pipeline_loop
from ..passes.promote_buffers import promote_buffers
from ..passes.schedule_async import schedule_async
from ..passes.tile_matmul import tile_matmul
from ..passes.unroll_loop import unroll_loop
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call


def auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
    *, input_stages: int = 3, output_tiles_per_cta: int = 1,
):
    if input_stages not in (2, 3):
        raise ValueError("input_stages must be 2 or 3")
    if output_tiles_per_cta not in (1, 2):
        raise ValueError("output_tiles_per_cta must be 1 or 2")
    name = f"auto_matmul_4096_wgmma256_wg_tma_store_k{input_stages}"
    if output_tiles_per_cta == 2:
        name += "_2tiles"
    grid_x = 32 // output_tiles_per_cta
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[{grid_x}, 16, 1] block_dim=[128, 1, 1] {{
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    if output_tiles_per_cta == 1:
        module = tile_matmul(
            module, tile_dims=["M", "N"], tile_sizes=[128, 256],
            tile_levels=["block.x", "block.y"], tile_labels=[None, None],
            label_filter="mm",
        )
    else:
        module = tile_matmul(
            module, tile_dims=["M", "M", "N"],
            tile_sizes=[2048, 128, 256],
            tile_levels=["for", "block.x", "block.y"],
            tile_labels=["outm", None, None], label_filter="mm",
        )
    module = tile_matmul(
        module, tile_dims=["M"], tile_sizes=[64],
        tile_levels=["wg.x"], tile_labels=[None], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="wrmem", operands=["dst"], swizzle=None,
        layout="wgmma_layout_0", warps=4, tma=True, nc=False,
        target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[64],
        tile_levels=["for"], tile_labels=["k"], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None, tma=True, nc=True,
        label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[16],
        tile_levels=["for"], tile_labels=["kk"], label_filter="mm",
    )
    module = unroll_loop(module, unroll_factor=input_stages, label_filter="k")
    module = lower_to_async(module)
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    module = pipeline_loop(
        module, label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async",
        pipelining_offset=1,
    )
    module = unroll_loop(
        module, unroll_factor=4, label_filter=r"kk_iter\d"
    )
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    if output_tiles_per_cta == 2:
        module = unroll_loop(
            module, unroll_factor=2, label_filter="outm"
        )
    module = schedule_async(module)
    return canonicalize(module)


def _register_auto_matmul_4096_wgmma256_wg_tma_store_k3() -> None:
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_k3_module()
    )
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(
        launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    two_tile_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
            output_tiles_per_cta=2
        )
    )
    two_tile_launcher = make_ptx_launcher(two_tile_kernel)
    make_torch_custom_call(
        two_tile_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    k2_two_tile_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
            input_stages=2, output_tiles_per_cta=2
        )
    )
    k2_two_tile_launcher = make_ptx_launcher(k2_two_tile_kernel)
    make_torch_custom_call(
        k2_two_tile_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )


_register_auto_matmul_4096_wgmma256_wg_tma_store_k3()
