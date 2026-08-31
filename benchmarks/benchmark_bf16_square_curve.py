#!/usr/bin/env python3
"""Benchmark and plot retained Ayla square BF16 GEMMs against cuBLAS."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

import common as benchmark_common
from ayla import ModuleOp
from ayla.gemm import (
    auto_matmul_4096_wgmma256_wg_tma_store_module,
    blackwell_tcgen05_gemm_module,
)
from ayla.gemm_collection.blackwell_tcgen05_4096 import (
    blackwell_tcgen05_4096_module,
)
from ayla.ptx_codegen import make_ptx_kernel
from ayla.ptx_runtime import make_ptx_launcher
from common import (
    BenchGemmConfig,
    assert_against_torch,
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


def _square_module(size: int, **config: object) -> ModuleOp:
    return auto_matmul_4096_wgmma256_wg_tma_store_module(
        problem_size=size, **config,
    )


HOPPER_CONFIGS: dict[int, tuple[Callable[[], ModuleOp], str]] = {
    512: (partial(_square_module, 512, operand_stages=2), "m128n256_k64_s2"),
    1024: (
        partial(
            _square_module,
            1024, output_tiles_per_cta=2, operand_stages=4, output_slices=2,
        ),
        "m2x128n256_k64_s4_split2",
    ),
    2048: (
        partial(
            _square_module,
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
        partial(
            _square_module,
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
        partial(
            _square_module,
            8192, output_tiles_per_cta=4, operand_stages=4, output_slices=2,
        ),
        "m4x128n256_k64_s4_split2",
    ),
    16384: (
        partial(
            _square_module,
            16384, output_tiles_per_cta=4, operand_stages=4, output_slices=2,
        ),
        "m4x128n256_k64_s4_split2",
    ),
}


def _blackwell_module(size: int) -> ModuleOp:
    common = {
        "m": size, "n": size, "k": size,
        "transpose_cta_grid": True,
        "lhs_l2_cache_hint": "evict_last",
        "rhs_l2_cache_hint": "evict_last",
        "lhs_l2_promotion": "256B",
        "rhs_l2_promotion": "256B",
        "output_l1_cache_policy": "no_allocate",
        "output_l2_eviction_priority": "evict_first",
    }
    if size == 512:
        return blackwell_tcgen05_gemm_module(
            output_m_tile=64, output_n_tile=32,
            operand_stages=32, derive_tcgen05_descriptors=True,
            ptx_optimization_level=2, **common,
            overlap_initial_tma_setup=True,
        )
    if size == 1024:
        return blackwell_tcgen05_gemm_module(
            output_m_tile=64, output_n_tile=64,
            operand_stages=16, derive_tcgen05_descriptors=True,
            prefetch_next_banks=1, ptx_optimization_level=2,
            overwrite_first_mma=True,
            batch_lock_initialization=True,
            overlap_tmem_deallocation=True,
            output_tmem_load_shape="16x32bx2",
            **{
                **common,
                "lhs_l2_cache_hint": None,
                "rhs_l2_cache_hint": None,
            },
        )
    if size == 2048:
        return blackwell_tcgen05_gemm_module(
            output_m_tile=128, output_n_tile=128,
            operand_stages=8, ptx_optimization_level=3,
            split_tma_barriers=True,
            split_rhs_tma=True, prefetch_lhs_next=True,
            prefetch_rhs_next=True, lean_bookkeeping=True,
            output_tmem_load_shape="16x32bx2",
            **{
                **common,
                "lhs_l2_cache_hint": None,
                "rhs_l2_cache_hint": None,
            },
        )
    if size == 4096:
        return blackwell_tcgen05_4096_module()
    if size == 16384:
        return blackwell_tcgen05_gemm_module(
            output_m_tile=128, output_n_tile=256,
            operand_stages=8, split_tma_barriers=True,
            split_rhs_tma=True, prefetch_lhs_next=True,
            prefetch_rhs_next=True, lean_bookkeeping=True,
            output_tmem_load_shape="16x32bx2",
            overwrite_first_mma=True,
            cluster_multicast_b=True,
            cluster_multicast_b_ctas=8,
            cluster_multicast_b_issuers=4,
            ptx_optimization_level=1,
            **common,
        )
    return blackwell_tcgen05_gemm_module(
        output_m_tile=128, output_n_tile=232, balanced_n_tile=224,
        operand_stages=8, split_tma_barriers=True,
        split_rhs_tma=True, prefetch_lhs_next=True,
        prefetch_rhs_next=True, lean_bookkeeping=True,
        output_tmem_load_shape="16x32bx2",
        persistent_n_pairs=True,
        persistent_m_pairs=size == 8192,
        reuse_persistent_lhs=size == 8192,
        **common,
    )


BLACKWELL_CONFIG_NAMES = {
    512: "tcgen05_m64n32_k512_early_tma_o2",
    1024: "tcgen05_m64n64_k256_prefetch_bank_paired_o2",
    2048: "tcgen05_m128n128_k128_prefetch_no_l2_hints_o3",
    4096: "tcgen05_balanced_persistent_phase_split",
    8192: "tcgen05_balanced_persistent_2x2_lhs_reuse",
    16384: "tcgen05_m128n256_k128_single_cluster_b8_i4_o1",
}


def configs_for_device() -> dict[int, tuple[Callable[[], ModuleOp], str]]:
    if torch.cuda.get_device_capability()[0] >= 10:
        return {
            size: (partial(_blackwell_module, size), name)
            for size, name in BLACKWELL_CONFIG_NAMES.items()
        }
    return HOPPER_CONFIGS


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
        f"{torch.cuda.get_device_name()} square BF16 GEMM — "
        "graph-captured, cache-hot\n"
        "labels show throughput advantage vs cuBLAS"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--reverse", action="store_true",
        help="benchmark cuBLAS before Ayla at each size",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        help="run only the listed configured square sizes",
    )
    parser.add_argument(
        "--graph-launches", type=int,
        help="launches captured per timed graph replay",
    )
    args = parser.parse_args()
    if args.graph_launches is not None:
        if args.graph_launches < 1:
            raise SystemExit("--graph-launches must be positive")
        benchmark_common._GRAPH_LAUNCHES = args.graph_launches
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    blackwell = torch.cuda.get_device_capability()[0] >= 10
    rows = []
    selected = configs_for_device()
    if args.sizes is not None:
        unknown = sorted(set(args.sizes) - set(selected))
        if unknown:
            raise SystemExit(f"unconfigured square sizes: {unknown}")
        selected = {size: selected[size] for size in args.sizes}
    for size, (build, config_name) in selected.items():
        print(f"size={size}: build {config_name}", flush=True)
        ayla_op = out_parameter_op(make_ptx_launcher(make_ptx_kernel(build())))
        ayla_config = BenchGemmConfig(
            "Ayla", size, size, size, ayla_op, torch.bfloat16,
        )
        if blackwell:
            for seed in (42, 43, 44):
                max_abs, relative_rms = assert_against_torch(
                    ayla_config,
                    seed=seed,
                    max_abs_error=2.0,
                    max_relative_rms_error=0.001,
                )
                print(
                    f"size={size} seed={seed} max_abs={max_abs:.6f} "
                    f"relative_rms={relative_rms:.9f}",
                    flush=True,
                )
        else:
            assert_against_torch(ayla_config)
        configs = [
            ayla_config,
            BenchGemmConfig(
                "cuBLAS", size, size, size, torch_mm_nt, torch.bfloat16,
            ),
        ]
        if args.reverse:
            configs.reverse()
        results = run(configs)
        by_label = {result.label.split(" ", 1)[0]: result for result in results}
        ayla_result = by_label["Ayla"]
        torch_result = by_label["cuBLAS"]
        print_results(results)
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
