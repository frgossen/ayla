"""Dynamic-M BF16 GEMM specialization and Torch integration."""
from __future__ import annotations

import textwrap
from typing import NotRequired, TypedDict

from .. import ptx_runtime, torch_custom_call
from ..gpu_dialect import GmemTensorType, KernelOp, ModuleOp, gpu_dialect
from ..jit_compiler import ShapeSpecializingJITCompiler
from ..ptx_codegen import make_ptx_kernel
from ..torch_custom_call import DEFAULT_NAMESPACE, OperandKind
from ..pipelines.tile_matmul_pipeline import TileMatmulPipeline


def _cache_policy_heuristic(
    *, keep_lhs: bool,
) -> dict[str, dict[str, object]]:
    return {
        "lhs": {"l2_cache_hint": "evict_last" if keep_lhs else "evict_first"},
        "rhs": {"l2_cache_hint": "evict_first"},
        "dst": {"l2_cache_hint": "evict_first"},
    }


class _DynamicMSchedule(TypedDict):
    output_m_tile: int
    output_n_tile: int
    operand_k_tile: int
    operand_stages: int
    operand_move_attrs: dict[str, dict[str, object]]
    output_tma_inner_size: NotRequired[int | None]


def _dynamic_m_schedule_heuristic(
    m: int, n: int, k: int,
) -> _DynamicMSchedule:
    """Broad schedule regimes distilled from the Muse cold-cache sweep."""
    square = n == k

    if square and (m <= 128 or m >= 4096):
        return {
            "output_m_tile": 128,
            "output_n_tile": 256,
            "operand_k_tile": 64,
            "operand_stages": 3,
            "operand_move_attrs": _cache_policy_heuristic(keep_lhs=False),
        }
    if m <= 128 and not square:
        return {
            "output_m_tile": 128,
            "output_n_tile": 64,
            "operand_k_tile": 64,
            "operand_stages": 4,
            "output_tma_inner_size": None,
            "operand_move_attrs": _cache_policy_heuristic(keep_lhs=True),
        }
    return {
        "output_m_tile": 128,
        "output_n_tile": 208,
        "operand_k_tile": 64,
        "operand_stages": 4,
        "output_tma_inner_size": None,
        "operand_move_attrs": _cache_policy_heuristic(
            keep_lhs=not square and 512 <= m <= 2048,
        ),
    }


def dynamic_m_matmul_template(
    n: int,
    k: int,
    *,
    name: str | None = None,
) -> ModuleOp:
    """Build the high-level BF16 GEMM template with dynamic M."""
    if n <= 0 or k <= 0:
        raise ValueError("problem dimensions must be positive")
    name = name or f"dynamic_m_matmul_n{n}_k{k}"
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<?x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<?x{n}xbf16>) {{
            matmul(%a: gmem<?x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<?x{n}xbf16>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
          }}
        }}"""))
    assert isinstance(module, ModuleOp)
    return module


def make_dynamic_m_matmul(
    n: int,
    k: int,
    *,
    name: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Create an exact-shape cached ``op(lhs, rhs) -> out`` GEMM."""
    module = dynamic_m_matmul_template(n, k, name=name)

    def compile_specialization(module: ModuleOp):
        [kernel] = module.body.ops
        assert isinstance(kernel, KernelOp)
        lhs_type = kernel.signature[0]
        assert isinstance(lhs_type, GmemTensorType)
        m = lhs_type.sizes[0]
        assert isinstance(m, int)
        scheduled = TileMatmulPipeline(
            m=m, n=n, k=k,
            **_dynamic_m_schedule_heuristic(m, n, k),
        ).run(module)
        kernel = make_ptx_kernel(scheduled)
        return ptx_runtime.make_ptx_launcher(kernel)

    compiler = ShapeSpecializingJITCompiler(module, compile_specialization)
    operands = [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    return torch_custom_call.make_torch_custom_call(
        compiler,
        operands,
        namespace=namespace,
        infer_result_shapes=lambda lhs, _rhs: [(lhs.shape[0], n)],
    )
