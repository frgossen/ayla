#!/usr/bin/env python3
"""Benchmark the Muse MLP ``GEMM -> SiLU -> GEMM`` chain at small M."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ayla.base.ir import Block, Value
from ayla.gemm_collection.gemm import schedule_persistent_matmul
from ayla.gemm_collection.muse_mlp_gemm_chain import (
    muse_mlp_gemm_chain_module,
)
from ayla.gpu_dialect import (
    BF16Type,
    ElementwiseOp,
    GmemTensorType,
    KernelOp,
    MatmulOp,
    ModuleOp,
)
from ayla.passes.canonicalize import Canonicalize
from ayla.passes.schedule_async import schedule_async
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from common import BenchConfig, error_metrics, run_bench, write_csv


M_VALUES = (32, 64)
HIDDEN = 6_656
INTERMEDIATE = 19_968
DEFAULT_CONFIGS = {
    32: (2, 48, 4),
    64: (None, None, None),
}


def _standalone_gemm(
    m: int,
    n: int,
    k: int,
    *,
    epilogue: str | None = None,
    dst_cache_hint: str = "evict_first",
):
    bf16 = BF16Type()
    lhs = Value("lhs", GmemTensorType.build([m, k], bf16))
    rhs = Value("rhs", GmemTensorType.build([n, k], bf16))
    dst = Value("dst", GmemTensorType.build([m, n], bf16))
    ops = [MatmulOp.build(lhs, rhs, dst, init=0.0, attrs={"label": "mm"})]
    if epilogue is not None:
        ops.append(ElementwiseOp.build(
            dst, dst, epilogue, attrs={"label": "mm_epi"},
        ))
    kernel = KernelOp.build(
        f"standalone_m{m}_n{n}_k{k}_{epilogue or 'none'}",
        Block(arguments=[lhs, rhs, dst], ops=ops),
        grid_dim=(312, 1, 1),
        ptx_optimization_level=4,
    )
    module = ModuleOp.build(Block(ops=[kernel]))
    module = schedule_persistent_matmul(
        module,
        label="mm",
        epilogue=epilogue,
        m_tile=64,
        n_tile=64,
        resident_blocks=312,
        operand_k_tile=64,
        operand_stages=4,
        dst_cache_hint=dst_cache_hint,
    )
    module = schedule_async(module)
    module = Canonicalize().run(module)
    return make_ptx_launcher(make_ptx_kernel(module))


def _run_case(
    m: int,
    split_k: int | None,
    down_n_tile: int | None,
    down_operand_stages: int | None,
) -> dict[str, object]:
    torch.manual_seed(20260902 + m)
    a = torch.randn((m, HIDDEN), device="cuda", dtype=torch.bfloat16)
    w_up = (
        torch.randn(
            (INTERMEDIATE, HIDDEN), device="cuda", dtype=torch.bfloat16,
        ) * .01
    )
    w_down = (
        torch.randn(
            (HIDDEN, INTERMEDIATE), device="cuda", dtype=torch.bfloat16,
        ) * .01
    )
    intermediate = torch.empty(
        (m, INTERMEDIATE), device="cuda", dtype=torch.bfloat16,
    )
    output = torch.empty((m, HIDDEN), device="cuda", dtype=torch.bfloat16)
    sync = torch.zeros(2, device="cuda", dtype=torch.float32)

    fused = make_ptx_launcher(make_ptx_kernel(
        muse_mlp_gemm_chain_module(
            m=m,
            split_k=split_k,
            down_n_tile=down_n_tile,
            down_operand_stages=down_operand_stages,
        ),
    ))
    baseline_fused = (
        make_ptx_launcher(make_ptx_kernel(muse_mlp_gemm_chain_module(m=m)))
        if split_k is not None else None
    )
    up = _standalone_gemm(
        m, INTERMEDIATE, HIDDEN,
        epilogue="silu", dst_cache_hint="evict_last",
    )
    down = _standalone_gemm(m, HIDDEN, INTERMEDIATE)

    def torch_chain(a, w_up, w_down):
        return torch.nn.functional.silu(a @ w_up.t()) @ w_down.t()

    compiled_torch = torch.compile(torch_chain, fullgraph=True)
    torch_output = None

    def run_fused() -> None:
        fused(a, w_up, w_down, intermediate, output, sync)

    def run_unfused() -> None:
        up(a, w_up, intermediate)
        down(intermediate, w_down, output)

    def run_baseline_fused() -> None:
        assert baseline_fused is not None
        baseline_fused(a, w_up, w_down, intermediate, output, sync)

    def run_torch() -> None:
        nonlocal torch_output
        torch_output = compiled_torch(a, w_up, w_down)

    expected = torch_chain(a, w_up, w_down)
    run_fused()
    torch.cuda.synchronize()
    max_abs, relative_rms = error_metrics(output, expected)
    if max_abs > .125 or relative_rms > .006:
        raise RuntimeError(
            f"M={m}: max abs {max_abs}, relative RMS {relative_rms}"
        )

    configs = [
        BenchConfig(
            name="Ayla unfused", make_inputs=lambda: [], compute=run_unfused,
        ),
        BenchConfig(
            name="Ayla fused", make_inputs=lambda: [], compute=run_fused,
        ),
    ]
    if baseline_fused is not None:
        configs.append(BenchConfig(
            name="Ayla fused baseline", make_inputs=lambda: [],
            compute=run_baseline_fused,
        ))
    configs.append(
        BenchConfig(
            name="Torch compiled", make_inputs=lambda: [], compute=run_torch,
        )
    )
    timings = run_bench(configs)
    by_label = {timing.name: timing for timing in timings}
    unfused_ms = by_label["Ayla unfused"].median_ms
    fused_ms = by_label["Ayla fused"].median_ms
    torch_ms = by_label["Torch compiled"].median_ms
    baseline_fused_ms = (
        by_label["Ayla fused baseline"].median_ms
        if baseline_fused is not None else fused_ms
    )
    print(
        f"M={m:2d}  unfused={unfused_ms:.6f} ms  fused={fused_ms:.6f} ms  "
        f"fusion={unfused_ms / fused_ms:.4f}x  torch={torch_ms:.6f} ms  "
        f"vs_torch={torch_ms / fused_ms:.4f}x  "
        f"vs_baseline={baseline_fused_ms / fused_ms:.4f}x"
    )
    return {
        "m": m,
        "split_k": split_k,
        "down_n_tile": down_n_tile,
        "down_operand_stages": down_operand_stages,
        "unfused_ayla_ms": unfused_ms,
        "fused_ayla_ms": fused_ms,
        "baseline_fused_ayla_ms": baseline_fused_ms,
        "split_k_speedup": baseline_fused_ms / fused_ms,
        "fusion_speedup": unfused_ms / fused_ms,
        "torch_ms": torch_ms,
        "fused_vs_torch": torch_ms / fused_ms,
        "fused_stdev_ms": by_label["Ayla fused"].stdev_ms,
        "correctness_max_abs": max_abs,
        "correctness_relative_rms": relative_rms,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", nargs="+", type=int, default=list(M_VALUES))
    parser.add_argument("--split-k", type=int)
    parser.add_argument("--down-n-tile", type=int)
    parser.add_argument("--down-operand-stages", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(m not in M_VALUES for m in args.ms):
        parser.error(f"--ms must be selected from {M_VALUES}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    use_selected = (
        args.split_k is None
        and args.down_n_tile is None
        and args.down_operand_stages is None
    )
    rows = []
    for m in args.ms:
        config = (
            DEFAULT_CONFIGS[m]
            if use_selected else (
                args.split_k, args.down_n_tile, args.down_operand_stages,
            )
        )
        rows.append(_run_case(m, *config))
    if args.output is not None:
        write_csv(rows, args.output)
        print(f"Raw results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
