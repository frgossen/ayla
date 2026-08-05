"""Configurable wgmma256 GEMM with pipelined TMA output stores."""
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


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, output_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    # Split each 256-column output row into two 128-element TMA dimensions.
    # Shared-only padding controls the physical row stride seen by the TMA
    # store; the best overall stride may trade some STMatrix bank expansion
    # for faster TMA reads of the arena.
    output_tma_inner_size: int | None = 128,
    output_tma_inner_padding: int = 8,
    output_store: str = "stmatrix",
):
    if output_tiles_per_cta not in (1, 2):
        raise ValueError("output_tiles_per_cta must be 1 or 2")
    if output_m_tile not in (128, 192):
        raise ValueError("output_m_tile must be 128 or 192")
    if output_m_tile == 192 and output_tiles_per_cta != 1:
        raise ValueError("M=192 currently requires one output tile per CTA")
    if output_n_tile not in (256, 384):
        raise ValueError("output_n_tile must be 256 or 384")
    if operand_k_tile not in (64, 128):
        raise ValueError("operand_k_tile must be 64 or 128")
    if warpgroup_n_tile is None:
        warpgroup_n_tile = output_n_tile if output_n_tile == 256 else 128
    if output_n_tile % warpgroup_n_tile:
        raise ValueError("output_n_tile must divide evenly by warpgroup_n_tile")
    if output_store not in ("scalar", "stmatrix"):
        raise ValueError("output_store must be 'scalar' or 'stmatrix'")
    if output_store == "scalar":
        output_tma_inner_size = None
        output_tma_inner_padding = 0
    name = f"auto_matmul_4096_wgmma256_wg_tma_store_k{operand_stages}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if warpgroup_n_tile not in (128, 256):
        name += f"_wg{warpgroup_n_tile}"
    if output_tiles_per_cta == 2:
        name += "_2tiles"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"
    m_blocks = (4096 + output_m_tile - 1) // output_m_tile
    grid_x = (m_blocks + output_tiles_per_cta - 1) // output_tiles_per_cta
    grid_y = (4096 + output_n_tile - 1) // output_n_tile
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[{grid_x}, {grid_y}, 1] block_dim=[128, 1, 1] {{
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    if output_tiles_per_cta == 1:
        module = tile_matmul(
            module, tile_dims=["M", "N"],
            tile_sizes=[output_m_tile, output_n_tile],
            tile_levels=["block.x", "block.y"], tile_labels=[None, None],
            label_filter="mm",
            allow_partial=(output_m_tile == 192 or output_n_tile == 384),
        )
    else:
        module = tile_matmul(
            module, tile_dims=["M", "M", "N"],
            tile_sizes=[grid_x * output_m_tile, output_m_tile, 256],
            tile_levels=["for", "block.x", "block.y"],
            tile_labels=["outm", None, None], label_filter="mm",
            allow_partial=output_m_tile == 192,
        )
    wg_dim, wg_size = (
        ("M", 64) if output_n_tile == 256
        else ("N", warpgroup_n_tile)
    )
    module = tile_matmul(
        module, tile_dims=[wg_dim], tile_sizes=[wg_size],
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
        writeback_buffers=output_store_buffers,
        target_dtype="f32", label_filter="mm",
    )
    module = tile_matmul(
        module, tile_dims=["K"], tile_sizes=[operand_k_tile],
        tile_levels=["for"], tile_labels=["k"], label_filter="mm",
    )
    module = promote_buffers(
        module, target_type="smem", operands=["lhs", "rhs"],
        swizzle="128B", layout=None, warps=None, tma=True, nc=True,
        physical_layout=(
            {"kind": "blocked_padding", "block": 64, "padding": 0,
             "order": "block_major"}
            if operand_k_tile > 64 else None
        ),
        tma_load_slices=(
            {"rhs": output_n_tile // warpgroup_n_tile}
            if output_n_tile == 384 else None
        ),
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
        module, unroll_factor=operand_k_tile // 16,
        label_filter=r"kk(?:_iter\d+)?",
    )
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    if output_tiles_per_cta == 2:
        module = unroll_loop(
            module, unroll_factor=2,
            label_filter="outm",
        )
    module = schedule_async(module)
    return canonicalize(module)


def _register_auto_matmul_4096_wgmma256_wg_tma_store() -> None:
    kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module()
    )
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(
        launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    two_tile_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_tiles_per_cta=2
        )
    )
    two_tile_launcher = make_ptx_launcher(two_tile_kernel)
    make_torch_custom_call(
        two_tile_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    k3_split2_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_tiles_per_cta=2, operand_stages=3, output_slices=2,
            output_tma_inner_padding=16, output_store_buffers=2,
        )
    )
    make_torch_custom_call(
        make_ptx_launcher(k3_split2_kernel),
        [OperandKind.R, OperandKind.R, OperandKind.RESULT],
    )

    k128_split2_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_tiles_per_cta=2, operand_stages=2, operand_k_tile=128,
            output_slices=2,
        )
    )
    make_torch_custom_call(
        make_ptx_launcher(k128_split2_kernel),
        [OperandKind.R, OperandKind.R, OperandKind.RESULT],
    )

    k4_split2_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_tiles_per_cta=2, operand_stages=4, output_slices=2,
        )
    )
    k4_split2_launcher = make_ptx_launcher(k4_split2_kernel)
    make_torch_custom_call(
        k4_split2_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    n384_k2_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_n_tile=384, operand_stages=2,
            output_slices=3,
        )
    )
    n384_k2_launcher = make_ptx_launcher(n384_k2_kernel)
    make_torch_custom_call(
        n384_k2_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )

    m192_k3_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_m_tile=192, operand_stages=3, output_slices=2,
        )
    )
    m192_k3_launcher = make_ptx_launcher(m192_k3_kernel)
    make_torch_custom_call(
        m192_k3_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )


    pre_stsm_kernel = make_ptx_kernel(
        auto_matmul_4096_wgmma256_wg_tma_store_module(
            output_tiles_per_cta=2, output_store="scalar",
        )
    )
    pre_stsm_launcher = make_ptx_launcher(pre_stsm_kernel)
    make_torch_custom_call(
        pre_stsm_launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )


_register_auto_matmul_4096_wgmma256_wg_tma_store()
