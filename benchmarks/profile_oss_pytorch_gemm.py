#!/usr/bin/env python3
"""Profile one measured Torch or Ayla GEMM inside a CUDA profiler range."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks.common import out_parameter_op, torch_mm_nt
from benchmarks.oss_gemm_corpus import RAW_JSON, ScheduleConfig


def _measured_config(
    rows: list[dict[str, Any]], dimensions: tuple[int, int, int], name: str,
) -> ScheduleConfig:
    matches = [
        row for row in rows
        if (int(row["m"]), int(row["n"]), int(row["k"])) == dimensions
        and row.get("implementation") == "ayla"
        and row.get("status") == "ok"
        and row.get("configuration", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one successful {dimensions} / {name!r} row; "
            f"found {len(matches)}"
        )
    config = matches[0]["configuration"]
    values: dict[str, object] = {}
    for field in dataclasses.fields(ScheduleConfig):
        if field.name in config:
            values[field.name] = config[field.name]
        elif field.default is not dataclasses.MISSING:
            values[field.name] = field.default
        else:
            raise ValueError(f"configuration is missing {field.name!r}")
    return ScheduleConfig(**values)  # type: ignore[arg-type]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("torch", "ayla"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--configuration")
    parser.add_argument("--results-json", type=Path, default=RAW_JSON)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    dimensions = (args.m, args.n, args.k)
    dtype = torch.float16
    a = torch.randn((args.m, args.k), dtype=dtype, device="cuda")
    bt = torch.randn((args.n, args.k), dtype=dtype, device="cuda")

    compute: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if args.implementation == "torch":
        if args.configuration is not None:
            parser.error("--configuration applies only to --implementation ayla")
        compute = torch_mm_nt
        print(f"torch.mm NT {args.m}x{args.n}x{args.k}", flush=True)
    else:
        if args.configuration is None:
            parser.error("--implementation ayla requires --configuration")
        rows = json.loads(args.results_json.read_text())
        schedule = _measured_config(rows, dimensions, args.configuration)
        from ayla.gemm_collection.gemm import (
            auto_matmul_4096_wgmma256_wg_tma_store_module,
        )
        from ayla.ptx_codegen import make_ptx_kernel
        from ayla.ptx_runtime import make_ptx_launcher

        module = auto_matmul_4096_wgmma256_wg_tma_store_module(
            m=args.m, n=args.n, k=args.k, dtype="f16",
            **schedule.builder_kwargs(),  # pyright: ignore[reportArgumentType]
        )
        kernel = make_ptx_kernel(module)
        print(
            f"{kernel.name}: grid={kernel.grid_dim}, block={kernel.block_dim}, "
            f"cluster={kernel.cluster_dim}, smem={kernel.shared_mem_bytes}; "
            f"config={dataclasses.asdict(schedule)}",
            flush=True,
        )
        compute = out_parameter_op(make_ptx_launcher(kernel))

    for _ in range(args.warmup):
        compute(a, bt)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    compute(a, bt)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
