"""Dynamic-M BF16 GEMM specialization and Torch integration."""
from __future__ import annotations

import textwrap

from .. import ptx_runtime, torch_custom_call
from ..gpu_dialect import ModuleOp, gpu_dialect
from ..jit_compiler import ShapeSpecializingJITCompiler
from ..ptx_codegen import make_ptx_kernel
from ..torch_custom_call import DEFAULT_NAMESPACE, OperandKind
from .gemm import schedule_matmul


def _cache_policy_heuristic(
    *, keep_lhs: bool,
) -> dict[str, dict[str, object]]:
    return {
        "lhs": {"l2_cache_hint": "evict_last" if keep_lhs else "evict_first"},
        "rhs": {"l2_cache_hint": "evict_first"},
        "dst": {"l2_cache_hint": "evict_first"},
    }


def _dynamic_m_schedule_heuristic(
    m: int, n: int, k: int,
) -> dict[str, object]:
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
    epilogue: str | None = None,
) -> ModuleOp:
    """Build the high-level BF16 GEMM template with dynamic M."""
    if n <= 0 or k <= 0:
        raise ValueError("problem dimensions must be positive")
    if epilogue not in (None, "sigmoid", "silu", "sigmoid_mul"):
        raise ValueError(f"unsupported epilogue: {epilogue}")
    name = name or (
        f"dynamic_m_matmul_n{n}_k{k}"
        f"{f'_{epilogue}' if epilogue is not None else ''}"
    )
    attention_arg = (
        f", %attention: gmem<?x{n}xbf16>"
        if epilogue == "sigmoid_mul" else ""
    )
    if epilogue == "sigmoid_mul":
        epilogue_ops = (
            f"\n            elementwise sigmoid"
            f"(%c: gmem<?x{n}xbf16>, %c: gmem<?x{n}xbf16>)"
            ' attrs={label: "epi"}'
            f"\n            elementwise mul"
            f"(%c: gmem<?x{n}xbf16>, %c: gmem<?x{n}xbf16>, "
            f"%attention: gmem<?x{n}xbf16>)"
            ' attrs={label: "epi"}'
        )
    elif epilogue is None:
        epilogue_ops = ""
    else:
        epilogue_ops = (
            f"\n            elementwise {epilogue}"
            f"(%c: gmem<?x{n}xbf16>, %c: gmem<?x{n}xbf16>)"
            ' attrs={label: "epi"}'
        )
    module = gpu_dialect.parse(textwrap.dedent(f"""\
        module {{
          kernel @{name}(%a: gmem<?x{k}xbf16>, %b: gmem<{n}x{k}xbf16>{attention_arg}, %c: gmem<?x{n}xbf16>) {{
            matmul(%a: gmem<?x{k}xbf16>, %b: gmem<{n}x{k}xbf16>, %c: gmem<?x{n}xbf16>) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}{epilogue_ops}
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
    compiler = _make_dynamic_m_matmul_compiler(n, k, name=name)

    operands = [OperandKind.R, OperandKind.R, OperandKind.RESULT]
    return torch_custom_call.make_torch_custom_call(
        compiler,
        operands,
        namespace=namespace,
        infer_result_shapes=lambda lhs, _rhs: [(lhs.shape[0], n)],
    )


def _make_dynamic_m_matmul_compiler(
    n: int,
    k: int,
    *,
    name: str | None = None,
    epilogue: str | None = None,
) -> ShapeSpecializingJITCompiler:
    """Create the cached compiler shared by functional and out calls."""
    module = dynamic_m_matmul_template(
        n, k, name=name, epilogue=epilogue,
    )

    def compile_specialization(module: ModuleOp):
        m = int(module.body.ops[0].signature[0].sizes[0])
        scheduled = schedule_matmul(
            module,
            epilogue_label_filter=(None if epilogue is None else "epi"),
            **_dynamic_m_schedule_heuristic(m, n, k),
        )
        kernel = make_ptx_kernel(scheduled)
        return ptx_runtime.make_ptx_launcher(kernel)

    return ShapeSpecializingJITCompiler(module, compile_specialization)


def make_dynamic_m_matmul_out(
    n: int,
    k: int,
    *,
    name: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    epilogue: str | None = None,
):
    """Create an exact-shape cached ``op(lhs, rhs, out)`` GEMM."""
    op_name = name or (
        f"dynamic_m_matmul_n{n}_k{k}"
        f"{f'_{epilogue}' if epilogue is not None else ''}_out"
    )
    compiler = _make_dynamic_m_matmul_compiler(
        n, k, name=op_name, epilogue=epilogue,
    )
    operands = [OperandKind.R, OperandKind.R]
    if epilogue == "sigmoid_mul":
        operands.append(OperandKind.R)
    operands.append(OperandKind.RW)
    return torch_custom_call.make_torch_custom_call(
        compiler,
        operands,
        namespace=namespace,
        name=op_name,
    )
