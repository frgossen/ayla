"""Three-stage wgmma256 GEMM with pipelined TMA output stores."""
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
from .registry import register


def _tma_store_k3_module(*, output_tiles_per_cta: int):
    if output_tiles_per_cta not in (1, 2):
        raise ValueError("output_tiles_per_cta must be 1 or 2")
    name = "auto_matmul_4096_wgmma256_wg_tma_store_k3"
    if output_tiles_per_cta == 2:
        name += "_2tiles"
    grid_x = 32 // output_tiles_per_cta
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) grid_dim=[{grid_x}, 16, 1] {{
            matmul(%a: gmem<4096x4096xbf16>, %b: gmem<4096x4096xbf16>, %c: gmem<4096x4096xbf16>) attrs={{init: 0.0, label: "mm"}}
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
        physical_layout={
            "swizzle": "128B",
            "block_width": 64,
            "order": "block_major",
        },
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
    module = multi_buffer(module, depth=3, label_filter="mm_lhs_smem")
    module = multi_buffer(module, depth=3, label_filter="mm_rhs_smem")
    module = licm(module)
    module = unroll_loop(module, unroll_factor=3, label_filter="k")
    module = canonicalize(module)
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


@register
def auto_matmul_4096_wgmma256_wg_tma_store_k3_module():
    return _tma_store_k3_module(output_tiles_per_cta=1)


@register
def auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles_module():
    return _tma_store_k3_module(output_tiles_per_cta=2)
