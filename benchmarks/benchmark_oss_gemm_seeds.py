"""Benchmark the compact seed set selected for one FP16 or BF16 GEMM shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import Any, cast

import torch

from ayla import (
    GemmScheduleSeed,
    apply_gemm_schedule_seed_post_build,
    select_gemm_schedules,
)
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
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--frequency", type=int, default=1)
    parser.add_argument("--show-only", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    return parser


def _ayla_compute(
    m: int, n: int, k: int, seed: GemmScheduleSeed, dtype: str = "float16",
):
    from ayla.gemm_collection.gemm import (
        auto_matmul_4096_wgmma256_wg_tma_store_module,
        schedule_matmul,
    )
    from ayla.gpu_dialect import gpu_dialect
    from ayla.ptx_codegen import make_ptx_kernel
    from ayla.ptx_runtime import make_ptx_launcher

    if seed.builder_config is None:
        raise ValueError("cannot build an unsupported GEMM schedule seed")
    ir_dtype = {"float16": "f16", "bfloat16": "bf16"}[dtype]
    if seed.transpose_output or "output_grid_size" in seed.builder_config:
        config = dict(seed.builder_config)
        ptx_level = config.pop("ptx_optimization_level", None)
        if seed.transpose_output:
            c_type = f"gmem<{n}x{m}x{ir_dtype} strides=[1, {n}]>"
            operands = (
                f"%b: gmem<{n}x{k}x{ir_dtype}>, "
                f"%a: gmem<{m}x{k}x{ir_dtype}>, %c: {c_type}"
            )
        else:
            c_type = f"gmem<{m}x{n}x{ir_dtype}>"
            operands = (
                f"%a: gmem<{m}x{k}x{ir_dtype}>, "
                f"%b: gmem<{n}x{k}x{ir_dtype}>, %c: {c_type}"
            )
        module = gpu_dialect.parse(textwrap.dedent(f"""\
            module {{
              kernel @mm(%a: gmem<{m}x{k}x{ir_dtype}>, %b: gmem<{n}x{k}x{ir_dtype}>, %c: {c_type}) attrs={{ptx_optimization_level: {ptx_level}}} {{
                matmul({operands}) collective=xyz.grid attrs={{init: 0.0, label: "mm"}}
              }}
            }}"""))
        module = schedule_matmul(module, **config)
    else:
        builder = cast(Any, auto_matmul_4096_wgmma256_wg_tma_store_module)
        module = builder(
            m=m,
            n=n,
            k=k,
            dtype=ir_dtype,
            **seed.builder_config,
        )
    module = apply_gemm_schedule_seed_post_build(module, seed)
    return out_parameter_op(make_ptx_launcher(make_ptx_kernel(module)))


def _write_json(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    args = _parser().parse_args()
    selection = select_gemm_schedules(args.m, args.n, args.k, dtype=args.dtype)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"dtype: {args.dtype}")
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
        dtype=torch_dtype,
    )]
    measured_seeds: list[GemmScheduleSeed | None] = [None]
    for index, seed in enumerate(selection.seeds, 1):
        if seed.builder_config is None:
            print(f"skipping unsafe seed {index}")
            continue
        try:
            bench = gemm_bench_config(
                name=f"seed-{index}",
                m=args.m,
                n=args.n,
                k=args.k,
                compute=_ayla_compute(
                    args.m, args.n, args.k, seed, args.dtype,
                ),
                dtype=torch_dtype,
            )
            assert_against_torch(
                bench,
                max_abs_error=32 if args.dtype == "bfloat16" else 4,
                max_relative_rms_error=2e-3,
            )
        except Exception as error:
            print(
                f"skipping seed {index}: "
                f"{type(error).__name__}: {error}",
            )
            continue
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
            "dtype": args.dtype,
            "frequency": args.frequency,
            "heuristic_group": (
                f"mm/{args.dtype}/{'k128' if args.k % 128 == 0 else 'k64'}"
            ),
            "heuristic_features": {
                "batch": 1,
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
