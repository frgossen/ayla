#!/usr/bin/env python3
"""Benchmark and plot retained Ayla square BF16 GEMMs against cuBLAS."""

from __future__ import annotations

from pathlib import Path

import torch

import ayla.ptx_runtime as runtime
import ayla.torch_custom_call as custom_call


# Import the configurable builder without registering its default kernels.
_make_launcher = runtime.make_ptx_launcher
_make_custom_call = custom_call.make_torch_custom_call
runtime.make_ptx_launcher = lambda kernel, **kwargs: None
custom_call.make_torch_custom_call = lambda *args, **kwargs: None
from ayla.gemm_collection.auto_matmul_4096_wgmma256_wg_tma_store import (  # noqa: E402
    auto_matmul_4096_wgmma256_wg_tma_store_module,
)
runtime.make_ptx_launcher = _make_launcher
custom_call.make_torch_custom_call = _make_custom_call

from ayla.ptx_codegen import make_ptx_kernel  # noqa: E402
from common import (  # noqa: E402
    BenchGemmConfig,
    check_against_torch,
    out_parameter_op,
    plot_tflops_curve,
    print_results,
    run,
    torch_mm_nt,
    write_csv,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".benchmarks"
CSV_PATH = OUTPUT_DIR / "bf16_square_curve.csv"
PLOT_PATH = OUTPUT_DIR / "bf16_square_curve.png"


def _square(size: int, **config):
    return lambda: auto_matmul_4096_wgmma256_wg_tma_store_module(
        problem_size=size, **config,
    )


CONFIGS = {
    512: (_square(512, operand_stages=2), "m128n256_k64_s2"),
    1024: (
        _square(
            1024, output_tiles_per_cta=2, operand_stages=4, output_slices=2,
        ),
        "m2x128n256_k64_s4_split2",
    ),
    2048: (
        _square(
            2048,
            output_tiles_per_cta=4,
            operand_stages=3,
            output_slices=2,
            output_store_buffers=2,
            operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
        ),
        "m4x128n256_k64_s3_split2",
    ),
    4096: (
        _square(
            4096,
            output_tiles_per_cta=4,
            operand_stages=3,
            output_slices=2,
            output_store_buffers=2,
            operand_move_attrs={"lhs": {"l2_cache_hint": "evict_first"}},
        ),
        "four_stage_split_output",
    ),
    8192: (
        _square(
            8192, output_tiles_per_cta=4, operand_stages=4, output_slices=2,
        ),
        "m4x128n256_k64_s4_split2",
    ),
    16384: (
        _square(
            16384, output_tiles_per_cta=4, operand_stages=4, output_slices=2,
        ),
        "m4x128n256_k64_s4_split2",
    ),
}


def _plot(rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    sizes = [int(row["size"]) for row in rows]
    ayla = [float(row["ayla_tflops"]) for row in rows]
    torch_values = [float(row["torch_tflops"]) for row in rows]

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    plot_tflops_curve(
        axis, sizes, ayla, torch_values, annotation_prefix="Ayla ",
    )
    axis.set_xlabel("Square matrix size")
    axis.set_ylabel("TFLOP/s")
    axis.set_title(
        "H100 square BF16 GEMM — graph-captured, cache-hot\n"
        "labels show throughput advantage vs cuBLAS"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    rows = []
    for size, (build, config_name) in CONFIGS.items():
        print(f"size={size}: build {config_name}", flush=True)
        ayla_op = out_parameter_op(_make_launcher(make_ptx_kernel(build())))
        ayla_config = BenchGemmConfig(
            "Ayla", size, size, size, ayla_op, torch.bfloat16,
        )
        check_against_torch(ayla_config)
        configs = [
            ayla_config,
            BenchGemmConfig(
                "cuBLAS", size, size, size, torch_mm_nt, torch.bfloat16,
            ),
        ]
        ayla_result, torch_result = run(configs)
        print_results([ayla_result, torch_result])
        rows.append({
            "size": size,
            "config": config_name,
            "ayla_ms": ayla_result.median_ms,
            "ayla_tflops": ayla_result.tflops,
            "torch_ms": torch_result.median_ms,
            "torch_tflops": torch_result.tflops,
            "ayla_throughput_delta_pct": 100 * (
                ayla_result.tflops / torch_result.tflops - 1
            ),
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        })
        del configs, ayla_config, ayla_op
        torch.cuda.empty_cache()

    write_csv(rows, CSV_PATH)
    _plot(rows)
    print(f"\nRaw results: {CSV_PATH.resolve()}")
    print(f"View plot:   {PLOT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
