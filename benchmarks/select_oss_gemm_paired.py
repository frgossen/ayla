#!/usr/bin/env python3
"""Select only thresholded improvements from paired OSS GEMM ablations."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.oss_gemm_corpus import require_same_timing, write_json


def _config_key(row: dict[str, Any]) -> str:
    config = {
        key: value
        for key, value in row["configuration"].items()
        if key != "name"
    }
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--ablation", action="append", default=[], required=True,
        metavar="JSON:PREFIX",
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument(
        "--allow-cross-run-base", action="store_true",
        help="compare candidate torch_pct to the fixed baseline when the "
        "ablation does not contain an exact base row",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_speedup <= 1:
        parser.error("--minimum-speedup must exceed one")

    baseline_rows = json.loads(args.baseline.read_text())
    torch_rows = {
        str(row["shape_id"]): row
        for row in baseline_rows
        if row.get("implementation") == "torch"
    }
    baseline = {
        str(row["shape_id"]): row
        for row in baseline_rows
        if row.get("implementation") == "ayla"
        and row.get("status") == "ok"
        and not row.get("configuration", {}).get("unsafe_reuse_smem")
    }

    winners: dict[str, tuple[float, dict[str, Any]]] = {}
    for spec in args.ablation:
        path_text, separator, prefix = spec.rpartition(":")
        if not separator:
            parser.error(f"ablation must be JSON:PREFIX, got {spec!r}")
        prefixes = tuple(prefix.split("|"))
        rows_by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ablation_rows = json.loads(Path(path_text).read_text())
        if args.allow_cross_run_base:
            require_same_timing(
                (args.baseline, baseline_rows), (path_text, ablation_rows),
            )
        for row in ablation_rows:
            if (
                row.get("implementation") == "ayla"
                and row.get("status") == "ok"
                and str(row.get("configuration", {}).get("name", ""))
                .startswith(prefixes)
                and not row.get("configuration", {}).get("unsafe_reuse_smem")
            ):
                rows_by_shape[str(row["shape_id"])].append(row)
        for shape_id, rows in rows_by_shape.items():
            if shape_id not in baseline:
                continue
            baseline_key = _config_key(baseline[shape_id])
            paired_bases = [
                row for row in rows if _config_key(row) == baseline_key
            ]
            if paired_bases:
                paired_bases = [
                    row for row in paired_bases
                    if row.get("torch_pct") is not None
                ]
                candidates = [
                    row for row in rows if row.get("torch_pct") is not None
                ]
                if not paired_bases or not candidates:
                    continue
                base_pct = max(
                    float(row["torch_pct"]) for row in paired_bases
                )
                alternative = max(
                    candidates, key=lambda row: float(row["torch_pct"]),
                )
                speedup = float(alternative["torch_pct"]) / base_pct
            elif args.allow_cross_run_base:
                candidates = [
                    row for row in rows if row.get("torch_pct") is not None
                ]
                if not candidates:
                    continue
                alternative = max(
                    candidates, key=lambda row: float(row["torch_pct"]),
                )
                base_pct = baseline[shape_id].get("torch_pct")
                if base_pct is None:
                    continue
                speedup = float(alternative["torch_pct"]) / float(base_pct)
            else:
                continue
            if _config_key(alternative) == baseline_key:
                continue
            if speedup >= args.minimum_speedup and (
                shape_id not in winners or speedup > winners[shape_id][0]
            ):
                winners[shape_id] = (speedup, alternative)

    output = []
    for shape_id in sorted(baseline):
        if shape_id not in torch_rows:
            continue
        selected = copy.deepcopy(
            winners.get(shape_id, (1.0, baseline[shape_id]))[1]
        )
        selected["paired_selection_speedup"] = winners.get(
            shape_id, (1.0, selected),
        )[0]
        output.extend((torch_rows[shape_id], selected))
    write_json(output, args.output)
    print(
        f"selected {len(winners)} thresholded changes across "
        f"{len(output) // 2} shapes; ranking JSON: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
