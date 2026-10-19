"""Benchmark the compact seed set selected for one FP16 GEMM shape."""

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
import benchmarks.common as benchmark_common
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
    parser.add_argument(
        "--max-timing-rounds", type=int, default=20,
        help=(
            "cap rotated CUDA-event samples per configuration; twenty "
            "already measure far more precisely than a shape's result "
            "carries over to a model's forward graph"
        ),
    )
    parser.add_argument(
        "--compile-workers", type=int,
        default=benchmark_common.COMPILE_WORKERS,
        help=(
            "CPU-only processes that build and assemble the seeds' kernels "
            "before timing; 0 builds and JIT-compiles them one by one"
        ),
    )
    return parser


def _build_ptx_kernel(m: int, n: int, k: int, seed: GemmScheduleSeed):
    """Schedule one seed and lower it to PTX; needs no GPU."""
    from ayla.gemm_collection.gemm import (
        auto_matmul_4096_wgmma256_wg_tma_store_module,
        schedule_matmul,
    )
    from ayla.gpu_dialect import gpu_dialect
    from ayla.ptx_codegen import make_ptx_kernel

    if seed.builder_config is None:
        raise ValueError("cannot build an unsupported GEMM schedule seed")
    if seed.transpose_output or "output_grid_size" in seed.builder_config:
        config = dict(seed.builder_config)
        ptx_level = config.pop("ptx_optimization_level", None)
        if seed.transpose_output:
            c_type = f"gmem<{n}x{m}xf16 strides=[1, {n}]>"
            operands = (
                f"%b: gmem<{n}x{k}xf16>, "
                f"%a: gmem<{m}x{k}xf16>, %c: {c_type}"
            )
        else:
            c_type = f"gmem<{m}x{n}xf16>"
            operands = (
                f"%a: gmem<{m}x{k}xf16>, "
                f"%b: gmem<{n}x{k}xf16>, %c: {c_type}"
            )
        module = gpu_dialect.parse(textwrap.dedent(f"""\
            module {{
              kernel @mm(%a: gmem<{m}x{k}xf16>, %b: gmem<{n}x{k}xf16>, %c: {c_type}) attrs={{ptx_optimization_level: {ptx_level}}} {{
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
            dtype="f16",
            **seed.builder_config,
        )
    module = apply_gemm_schedule_seed_post_build(module, seed)
    return make_ptx_kernel(module)


def _ayla_compute(
    m: int, n: int, k: int, seed: GemmScheduleSeed, prebuilt: tuple | None,
):
    """The seed's launcher, from its ``(PtxKernel, cubin)`` when prebuilt."""
    from ayla.ptx_runtime import make_ptx_launcher

    kernel, image = (
        (_build_ptx_kernel(m, n, k, seed), None) if prebuilt is None
        else prebuilt
    )
    return out_parameter_op(make_ptx_launcher(kernel, image))


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
    benchmark_common._MAX_ROUNDS = args.max_timing_rounds
    buildable = [seed for seed in selection.seeds if seed.builder_config is not None]
    prebuilt = dict(zip(map(id, buildable), benchmark_common.assemble_kernels(
        _build_ptx_kernel, [(args.m, args.n, args.k, seed) for seed in buildable],
        min(args.compile_workers, len(buildable)), None,
    ))) if args.compile_workers else {}
    for index, seed in enumerate(selection.seeds, 1):
        if seed.builder_config is None:
            print(f"skipping unsafe seed {index}")
            continue
        entry = prebuilt.get(id(seed))
        if entry is not None and isinstance(entry[1], str):
            print(f"skipping seed {index}: {entry[0]}: {entry[1]}")
            continue
        try:
            bench = gemm_bench_config(
                name=f"seed-{index}",
                m=args.m,
                n=args.n,
                k=args.k,
                compute=_ayla_compute(
                    args.m, args.n, args.k, seed, entry,
                ),
                dtype=torch.float16,
            )
            assert_against_torch(
                bench,
                max_abs_error=4,
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
