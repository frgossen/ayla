#!/usr/bin/env python3
"""Benchmark and plot Muse projection GEMMs across dynamic token counts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from ayla.gemm_collection.gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from benchmarks.common import (
    GemmBenchConfig,
    gemm_bench_config,
    assert_against_torch,
    out_parameter_op,
    plot_tflops_curve,
    print_results,
    run,
    torch_mm_nt,
    write_csv,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".benchmarks"
CSV_PATH = OUTPUT_DIR / "bf16_muse_dynamic_m.csv"
PLOT_PATH = OUTPUT_DIR / "bf16_muse_dynamic_m.png"
M_VALUES = (128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class MatmulConfig:
    label: str
    m: int
    n: int
    k: int


PROJECTION_DIMS = {
    "up": (19968, 6656),
    "intermediate": (19968, 19968),
    "down": (6656, 19968),
}

CONFIGS = {
    "one_tile_k3": {
        "output_m_tiles_per_cta": 1,
        "operand_stages": 3,
    },
    "two_tile_k4_split2": {
        "output_m_tiles_per_cta": 2,
        "operand_stages": 4,
        "output_slices": 2,
    },
    "four_tile_k3_split2": {
        "output_m_tiles_per_cta": 4,
        "operand_stages": 3,
        "output_slices": 2,
        "output_store_buffers": 2,
        "operand_move_attrs": {"lhs": {"l2_cache_hint": "evict_first"}},
    },
}


def _default_config(m: int) -> str:
    if m < 256:
        return "one_tile_k3"
    if m < 512:
        return "two_tile_k4_split2"
    return "four_tile_k3_split2"


def _ayla_config(matmul: MatmulConfig) -> tuple[GemmBenchConfig, str]:
    config_name = _default_config(matmul.m)
    module = auto_matmul_4096_wgmma256_wg_tma_store_module(
        m=matmul.m,
        n=matmul.n,
        k=matmul.k,
        **CONFIGS[config_name],
    )
    op = out_parameter_op(make_ptx_launcher(make_ptx_kernel(module)))
    return (
        gemm_bench_config(
            name=f"Ayla {matmul.label}", m=matmul.m, n=matmul.n,
            k=matmul.k, compute=op, dtype=torch.bfloat16,
        ),
        config_name,
    )


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    workloads = [
        name for name in PROJECTION_DIMS
        if any(row["workload"] == name for row in rows)
    ]
    figure, axes = plt.subplots(
        1, len(workloads), figsize=(6.4 * len(workloads), 5.6),
        sharey=True, squeeze=False,
    )
    for axis, workload_name in zip(axes[0], workloads, strict=True):
        selected = sorted(
            (row for row in rows if row["workload"] == workload_name),
            key=lambda row: int(row["m"]),
        )
        ms = [int(row["m"]) for row in selected]
        ayla = [float(row["ayla_tflops"]) for row in selected]
        torch_values = [float(row["torch_tflops"]) for row in selected]
        plot_tflops_curve(
            axis, ms, ayla, torch_values, annotation_fontsize=6.5,
        )
        axis.set_title(
            f"{workload_name.capitalize()}: "
            f"N={int(selected[0]['n']):,}, K={int(selected[0]['k']):,}"
        )
        axis.set_xlabel("M = flattened tokens (B×S)")
    axes[0][0].set_ylabel("Useful TFLOP/s")
    axes[0][-1].legend()
    figure.suptitle("H100 BF16 Muse projections — Ayla advantage vs cuBLAS")
    figure.tight_layout(rect=(0, 0, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    # Every config in this file wants tf32 matmuls; it used to be a
    # per-config field that nothing ever set to False.
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads", nargs="+", choices=PROJECTION_DIMS,
        default=list(PROJECTION_DIMS),
    )
    parser.add_argument("--ms", nargs="+", type=int, default=list(M_VALUES))
    parser.add_argument("--output", type=Path, default=CSV_PATH)
    parser.add_argument("--plot-output", type=Path, default=PLOT_PATH)
    args = parser.parse_args()
    if any(m not in M_VALUES for m in args.ms):
        parser.error(f"--ms must be selected from {M_VALUES}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    rows = []
    for workload_name in args.workloads:
        n, k = PROJECTION_DIMS[workload_name]
        for m in args.ms:
            matmul = MatmulConfig(workload_name, m, n, k)
            ayla_config, config_name = _ayla_config(matmul)
            max_abs, relative_rms = assert_against_torch(
                ayla_config, seed=m, max_abs_error=4, max_relative_rms_error=1e-3,
            )
            configs = [
                ayla_config,
                gemm_bench_config(
                    name="cuBLAS", m=matmul.m, n=matmul.n, k=matmul.k,
                    compute=torch_mm_nt, dtype=torch.bfloat16,
                ),
            ]
            ayla_result, torch_result = run(configs)
            print_results([ayla_result, torch_result])
            rows.append({
                "workload": matmul.label,
                "m": matmul.m,
                "n": matmul.n,
                "k": matmul.k,
                "config": config_name,
                "ayla_ms": ayla_result.median_ms,
                "ayla_tflops": ayla_result.tflops,
                "torch_ms": torch_result.median_ms,
                "torch_tflops": torch_result.tflops,
                "ayla_throughput_delta_pct": 100 * (
                    ayla_result.tflops / torch_result.tflops - 1
                ),
                "correctness_max_abs": max_abs,
                "correctness_relative_rms": relative_rms,
                "gpu": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            })
            del configs, ayla_config
            torch.cuda.empty_cache()

    write_csv(rows, args.output)
    _plot(rows, args.plot_output)
    print(f"\nRaw results: {args.output.resolve()}")
    print(f"View plot:   {args.plot_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
