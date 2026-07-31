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
    *, output_tiles_per_cta: int = 1,
    operand_stages: int = 3,
    output_slices: int = 1,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Eight shared-only elements of padding per dimension make the physical
    # row stride 272 BF16s, reducing STMatrix's bank expansion from 8x to 2x.
    output_tma_inner_size: int | None = 128,
    output_tma_inner_padding: int = 8,
    output_store: str = "stmatrix",
):
    if output_tiles_per_cta not in (1, 2):
        raise ValueError("output_tiles_per_cta must be 1 or 2")
    if output_store not in ("scalar", "stmatrix"):
        raise ValueError("output_store must be 'scalar' or 'stmatrix'")
    if output_store == "scalar":
        output_tma_inner_size = None
        output_tma_inner_padding = 0
    name = "auto_matmul_4096_wgmma256_wg_tma_store_k3"
    if output_tiles_per_cta == 2:
        name += "_2tiles"
    if (operand_stages, output_slices) != (3, 1):
        name += f"_k{operand_stages}_split{output_slices}"
    if output_store == "scalar":
        name += "_pre_stsm"
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
        physical_layout=(
            {"kind": "blocked_padding", "block": output_tma_inner_size,
             "padding": output_tma_inner_padding}
            if output_tma_inner_size is not None else None
        ),
        writeback_slices=output_slices,
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
    module = unroll_loop(module, unroll_factor=operand_stages, label_filter="k")
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

    k4_split2_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
            output_tiles_per_cta=2, operand_stages=4, output_slices=2,
        )
    )
    k4_split2_launcher = make_ptx_launcher(k4_split2_kernel)
    make_torch_custom_call(
        k4_split2_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    pre_stsm_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_k3_module(
            output_tiles_per_cta=2, output_store="scalar",
        )
    )
    pre_stsm_launcher = make_ptx_launcher(pre_stsm_kernel)
    make_torch_custom_call(
        pre_stsm_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )


_register_auto_matmul_4096_wgmma256_wg_tma_store_k3()
