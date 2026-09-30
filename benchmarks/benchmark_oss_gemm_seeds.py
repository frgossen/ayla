"""Benchmark the compact seed set selected for one FP16 GEMM shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from ayla import GemmScheduleSeed, select_gemm_schedules
from benchmarks.common import (
    assert_against_torch,
    gemm_bench_config,
    out_parameter_op,
    print_results,
    run,
    torch_mm_nt,
    write_csv,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--frequency", type=int, default=1)
    parser.add_argument("--show-only", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    return parser


def _ayla_compute(m: int, n: int, k: int, config: dict[str, object]):
    from ayla.gemm_collection.gemm import (
        auto_matmul_4096_wgmma256_wg_tma_store_module,
    )
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher

    builder = cast(Any, auto_matmul_4096_wgmma256_wg_tma_store_module)
    module = builder(
        m=m,
        n=n,
        k=k,
        dtype="f16",
        **config,
    )
    return out_parameter_op(make_ptx_launcher(make_ptx_kernel(module)))


def _write_json(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    args = _parser().parse_args()
    selection = select_gemm_schedules(args.m, args.n, args.k)
    print(f"leaf: {selection.leaf}")
    print("path: " + " -> ".join(selection.decision_path))
    for index, seed in enumerate(selection.seeds, 1):
        source_score = (
            "unmeasured" if seed.source_percentage_of_torch is None
            else f"{seed.source_percentage_of_torch:.1f}% of torch"
        )
        print(
            f"seed {index} from {seed.source_shape} "
            f"({source_score}): "
            f"{json.dumps(seed.captured_config, sort_keys=True)}"
        )
        for adaptation in seed.adaptations:
            print(f"  adaptation: {adaptation}")
        if seed.builder_config is not None:
            print(
                "  builder: "
                f"{json.dumps(seed.builder_config, sort_keys=True)}"
            )
    if args.show_only:
        return 0
    if not any(seed.builder_config is not None for seed in selection.seeds):
        raise RuntimeError("the heuristic produced no legal schedule seeds")

    configs = [gemm_bench_config(
        name="torch.mm",
        m=args.m,
        n=args.n,
        k=args.k,
        compute=torch_mm_nt,
        dtype=torch.float16,
    )]
    measured_seeds: list[GemmScheduleSeed | None] = [None]
    for index, seed in enumerate(selection.seeds, 1):
        if seed.builder_config is None:
            print(f"skipping unsafe seed {index}")
            continue
        bench = gemm_bench_config(
            name=f"seed-{index}",
            m=args.m,
            n=args.n,
            k=args.k,
            compute=_ayla_compute(
                args.m, args.n, args.k, seed.builder_config,
            ),
            dtype=torch.float16,
        )
        assert_against_torch(
            bench,
            max_abs_error=4,
            max_relative_rms_error=2e-3,
        )
        configs.append(bench)
        measured_seeds.append(seed)

    results = run(configs)
    print_results(cast(Any, results))
    rows: list[dict[str, object]] = [
        {
            "status": "ok",
            "workload": "gemm",
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": "float16",
            "frequency": args.frequency,
            "heuristic_group": (
                f"float16/{'k128' if args.k % 128 == 0 else 'k64'}"
            ),
            "heuristic_features": {
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "aspect_m_over_n": args.m / args.n,
                "output_tiles_64": (
                    ((args.m + 63) // 64) * ((args.n + 63) // 64)
                ),
                "reduction_tiles_64": (args.k + 63) // 64,
            },
            "heuristic_source": {"m": args.m, "n": args.n, "k": args.k},
            "configuration": (
                None if seed is None else seed.builder_config
            ),
            "captured_configuration": (
                None if seed is None else seed.captured_config
            ),
            "adaptations": (
                [] if seed is None else list(seed.adaptations)
            ),
            "latency_ms": result.median_ms,
            "tflops": result.tflops,
            "percentage_of_torch": result.torch_mm_pct,
        }
        for seed, result in zip(measured_seeds, results, strict=True)
    ]
    if args.csv is not None:
        csv_rows: list[dict[str, object]] = []
        for row in rows:
            csv_row = dict(row)
            csv_row["configuration"] = json.dumps(row["configuration"])
            csv_row["captured_configuration"] = json.dumps(
                row["captured_configuration"]
            )
            csv_row["adaptations"] = json.dumps(row["adaptations"])
            csv_rows.append(csv_row)
        write_csv(csv_rows, args.csv)
    if args.json is not None:
        _write_json(rows, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
