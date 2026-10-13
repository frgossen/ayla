"""Configurable GEMM module builders used by collection entries."""
import textwrap

from ..gpu_dialect import ModuleOp, gpu_dialect
from ..pipelines.tile_matmul_pipeline import TileMatmulPipeline


def auto_matmul_4096_wgmma256_wg_tma_store_module(
    *, problem_size: int = 4096,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    output_m_tiles_per_cta: int = 1,
    output_n_tiles_per_cta: int = 1,
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_n_tile: int | None = None,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    operand_move_attrs: dict[str, dict[str, object]] | None = None,
    ptx_optimization_level: int | None = None,
    dtype: str = "bf16",
) -> ModuleOp:
    """Build and schedule a static BF16 or FP16 GEMM module."""
    m = problem_size if m is None else m
    n = problem_size if n is None else n
    k = problem_size if k is None else k
    if min(m, n, k) <= 0:
        raise ValueError("problem dimensions must be positive")
    if dtype not in ("bf16", "f16"):
        raise ValueError("dtype must be 'bf16' or 'f16'")
    problem_name = (
        str(problem_size)
        if m == n == k == problem_size
        else f"m{m}_n{n}_k{k}"
    )
    resolved_warpgroup_n_tile = warpgroup_n_tile
    if resolved_warpgroup_n_tile is None:
        resolved_warpgroup_n_tile = (
            output_n_tile if output_n_tile <= 256 else 128
        )
    name = f"auto_matmul_{problem_name}_wgmma256_wg_tma_store_k{operand_stages}"
    if dtype != "bf16":
        name += f"_{dtype}"
    if operand_k_tile != 64:
        name += f"_ktile{operand_k_tile}"
    if output_m_tile != 128:
        name += f"_m{output_m_tile}"
    if output_n_tile != 256:
        name += f"_n{output_n_tile}"
    if resolved_warpgroup_n_tile not in (128, 256):
        name += f"_wg{resolved_warpgroup_n_tile}"
    if output_m_tiles_per_cta != 1:
        name += f"_{output_m_tiles_per_cta}tiles"
    if output_n_tiles_per_cta != 1:
        name += f"_n{output_n_tiles_per_cta}tiles"
    if output_slices != 1:
        name += f"_split{output_slices}"
    if output_store_buffers != 1:
        name += f"_dbuf{output_store_buffers}"
    if output_store == "scalar":
        name += "_pre_stsm"

    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>) attrs={{ptx_optimization_level: {ptx_optimization_level}}} {{
            matmul(%a: gmem<{m}x{k}x{dtype}>, %b: gmem<{n}x{k}x{dtype}>, %c: gmem<{m}x{n}x{dtype}>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return TileMatmulPipeline(
        m=m, n=n, k=k,
        output_m_tiles_per_cta=output_m_tiles_per_cta,
        output_n_tiles_per_cta=output_n_tiles_per_cta,
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        warpgroup_n_tile=resolved_warpgroup_n_tile,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_tma_inner_size=output_tma_inner_size,
        output_store=output_store,
        operand_move_attrs=operand_move_attrs,
    ).run(module)
