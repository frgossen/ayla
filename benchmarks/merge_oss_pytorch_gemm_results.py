#!/usr/bin/env python3
"""Merge disjoint OSS GEMM benchmark shards into canonical raw CSV/JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.oss_gemm_corpus import (
    RAW_CSV,
    RAW_JSON,
    benchmark_shapes,
    load_shapes,
    validate_shapes,
    write_json,
    write_rows_csv,
)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    config = row.get("configuration", {})
    name = config.get("name", "") if isinstance(config, dict) else ""
    return str(row.get("shape_id", "")), str(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=RAW_JSON)
    parser.add_argument("--output-csv", type=Path, default=RAW_CSV)
    parser.add_argument(
        "--prefer-last", action="store_true",
        help="replace repeated non-Torch measurements with the last input",
    )
    args = parser.parse_args()

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    supported = {
        shape.shape_id
        for shape in benchmark_shapes(validate_shapes(load_shapes()))
    }
    for path in args.inputs:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON list")
        for row in rows:
            if str(row.get("shape_id", "")) not in supported:
                continue
            key = _key(row)
            if key in by_key and by_key[key] != row:
                # Follow-up passes remeasure Torch beside their added Ayla
                # candidate. Preserve the first baseline so one shape still
                # has exactly one denominator in the merged dataset.
                if key[1] == "torch.mm":
                    continue
                if args.prefer_last:
                    by_key[key] = row
                    continue
                raise ValueError(f"conflicting duplicate result for {key}")
            by_key[key] = row

    merged = sorted(
        by_key.values(),
        key=lambda row: (
            -int(row["frequency"]), str(row["shape_id"]),
            str(row["implementation"]), _key(row)[1],
        ),
    )
    write_json(merged, args.output_json)
    write_rows_csv(merged, args.output_csv)
    print(f"merged {len(merged)} rows from {len(args.inputs)} shards")
    print(f"raw JSON: {args.output_json.resolve()}")
    print(f"raw CSV:  {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
