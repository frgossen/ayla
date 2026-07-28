"""Store-pipelined ``auto_matmul_4096_wgmma256_wg`` variant.

Each block computes multiple 128x256 output tiles sequentially. Completed
accumulators are staged through four compact shared-memory stripes and stored
with TMA; each store wait is delayed across the following tile's WGMMA work.
"""
import textwrap

from ..gpu_dialect import gpu_dialect
from ..passes.canonicalize import canonicalize
from ..passes.fuse_async import fuse_async
from ..passes.licm import licm
from ..passes.lower_to_async import lower_to_async
from ..passes.pipeline_loop import pipeline_loop
from ..passes.pipeline_store import pipeline_store
from ..passes.promote_buffers import promote_buffers
from ..passes.schedule_async import schedule_async
from ..passes.tile_matmul import tile_matmul
from ..passes.unroll_loop import unroll_loop
from ..ptx_codegen import make_ptx_kernel
from ..ptx_runtime import make_ptx_launcher
from ..torch_custom_call import OperandKind, make_torch_custom_call


def auto_matmul_4096_wgmma256_wg_storepipe_module(
    reduction_factor: int = 1,
    reduction_m: int = 2,
    k_stages: int = 2,
    output_buffers: int = 1,
    kernel_name: str = "auto_matmul_4096_wgmma256_wg_storepipe",
):
    """Build a two-dimensional sequentially tiled, store-pipelined GEMM.

    Args:
        reduction_factor: ``Rn`` — number of 256-wide N output tiles each
            block computes sequentially via the ``"outn"`` loop. Grid.y
            shrinks to ``16 / Rn``. Must divide 16.
        reduction_m: ``Rm`` — number of 128-tall M output tiles each block
            computes sequentially via the ``"outm"`` loop. Grid.x shrinks to
            ``32 / Rm``. Must divide 32. Each block computes ``Rm * Rn``
            output tiles; total blocks = ``(32/Rm) * (16/Rn)``.
        k_stages: operand-buffer depth = number of pipelined K-stages (the
            ``"k"`` unroll factor).
        output_buffers: one or two round-robin 64 KB output staging banks.
    """
    Rn = reduction_factor
    Rm = reduction_m
    if 16 % Rn != 0:
        raise ValueError(f"reduction_factor must divide 16; got {Rn}")
    if 32 % Rm != 0:
        raise ValueError(f"reduction_m must divide 32; got {Rm}")
    if 64 % k_stages != 0:
        raise ValueError(f"k_stages must divide 64; got {k_stages}")
    if Rm * Rn <= 1:
        raise ValueError("store pipelining requires more than one output tile per block")
    if k_stages > 2:
        raise ValueError("store pipelining requires k_stages <= 2 (smem budget)")
    if output_buffers not in (1, 2):
        raise ValueError(f"output_buffers must be 1 or 2; got {output_buffers}")

    module_str = textwrap.dedent(f"""\
        module {{
          kernel @{kernel_name}(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) grid_dim=[{32 // Rm}, {16 // Rn}, 1] block_dim=[128, 1, 1] {{
            matmul(%a: gmem_tensor<4096x4096xbf16>, %b: gmem_tensor<4096x4096xbf16>, %c: gmem_tensor<4096x4096xbf16>) attrs={{init: 0.0, label: "mm"}}
          }}
        }}""")
    module = gpu_dialect.parse(module_str)
    # Grid-reduction tiling: block.x tiles M to 128*Rm, block.y tiles N to
    # 256*Rn, then per-block sequential "outm"/"outn" for-loops peel single
    # 128x256 output tiles. Everything below nests inside those loops, so the
    # accumulator init + K-loop run per output tile. Rm=Rn=1 reproduces the
    # baseline (one 128x256 tile per block, no sequential loop).
    tile_dims = ["M", "N"]
    tile_sizes = [128 * Rm, 256 * Rn]
    tile_levels = ["block.x", "block.y"]
    tile_labels = [None, None]
    if Rm > 1:
        tile_dims.append("M"); tile_sizes.append(128)
        tile_levels.append("for"); tile_labels.append("outm")
    if Rn > 1:
        tile_dims.append("N"); tile_sizes.append(256)
        tile_levels.append("for"); tile_labels.append("outn")
    module = tile_matmul(
        module,
        tile_dims=tile_dims, tile_sizes=tile_sizes,
        tile_levels=tile_levels, tile_labels=tile_labels,
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
    module = unroll_loop(module, unroll_factor=k_stages, label_filter="k")
    module = lower_to_async(module)
    module = fuse_async(module)
    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    module = pipeline_loop(
        module, label_filter_loop="k",
        label_filter_op=r"mm_(lhs|rhs)_smem_iter\d+_async", pipelining_offset=1,
    )

    module = unroll_loop(module, unroll_factor=4, label_filter=r"kk_iter\d")
    module = fuse_async(module)

    for _ in range(4):
        module = licm(module)
        module = canonicalize(module)

    module = schedule_async(module)
    module = canonicalize(module)

    # Do this only after LICM has hoisted the K-pipeline operand buffers out of
    # the output loop. Unrolling earlier would clone those large allocations.
    module = unroll_loop(module, unroll_factor=Rn, label_filter="outn")
    module = unroll_loop(module, unroll_factor=Rm, label_filter="outm")
    module = canonicalize(module)
    module = pipeline_store(
        module,
        label_filter=r"mm_dst_wrmem(?:_iter\d+)+",
        num_buffers=output_buffers,
    )

    return module


def _register() -> None:
    module = auto_matmul_4096_wgmma256_wg_storepipe_module(
        kernel_name="auto_matmul_4096_wgmma256_wg_storepipe",
    )
    kernel = make_ptx_kernel(module)
    launcher = make_ptx_launcher(kernel)
    make_torch_custom_call(
        launcher, [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    )


_register()
