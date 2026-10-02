"""Expand, collapse, or update a compact measured-schedule heuristic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ayla.gemm_heuristics import (
    collapse_gemm_heuristic,
    expand_gemm_heuristic,
)


def _leaf(tree: dict, features: dict[str, float]) -> dict:
    while "feature" in tree:
        branch = "left" if features[tree["feature"]] <= tree["threshold"] else "right"
        tree = tree[branch]
    return tree


def _without_names(value):
    if isinstance(value, dict):
        return {
            key: _without_names(item)
            for key, item in value.items() if key != "name"
        }
    if isinstance(value, list):
        return [_without_names(item) for item in value]
    return value


def _update(expanded: dict, paths: list[Path]) -> dict:
    touched: dict[int, dict] = {}
    for path in paths:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON list")
        for row in rows:
            config = row.get("configuration")
            if row.get("status", "ok") != "ok" or not isinstance(config, dict):
                continue
            if row.get("workload") != expanded["workload"]:
                raise ValueError(f"{path}: result belongs to another workload")
            features = row.get("heuristic_features")
            group = row.get("heuristic_group")
            source = row.get("heuristic_source")
            score = row.get("percentage_of_torch")
            if not isinstance(features, dict) or not isinstance(group, str):
                raise ValueError(f"{path}: result lacks heuristic routing metadata")
            if not isinstance(source, dict) or not isinstance(score, (int, float)):
                raise ValueError(f"{path}: result lacks heuristic source or score")
            leaf = _leaf(expanded["groups"][group], features)
            leaf["seeds"].append({
                "configuration": config,
                "source": source,
                "source_percentage_of_torch": score,
            })
            touched[id(leaf)] = leaf

    for leaf in touched.values():
        best = {}
        for seed in leaf["seeds"]:
            family = json.dumps(
                _without_names(seed["configuration"]),
                sort_keys=True,
                separators=(",", ":"),
            )
            incumbent = best.get(family)
            if incumbent is None or seed["source_percentage_of_torch"] > (
                incumbent["source_percentage_of_torch"]
            ):
                best[family] = seed
        leaf["seeds"] = sorted(
            best.values(),
            key=lambda seed: (
                seed["configuration"].get("unsafe_reuse_smem") is True,
                -seed["source_percentage_of_torch"],
            ),
        )[:3]
    return expanded


def _from_mapper(mapper: dict) -> dict:
    """Translate the measured-seed mapper into the production schema."""
    def node(value: dict) -> dict:
        if value["kind"] == "split":
            return {
                "feature": value["feature"],
                "threshold": value["threshold"],
                "left": node(value["left"]),
                "right": node(value["right"]),
            }
        return {
            "leaf": value["id"],
            "seeds": [
                {
                    "configuration": seed["configuration"],
                    "source": seed["source_shape"],
                    "source_percentage_of_torch": (
                        seed["source_percentage_of_torch"]
                    ),
                }
                for seed in value["seeds"]
            ],
        }

    return {
        "version": mapper["version"],
        "workload": "gemm",
        "source_shape_count": mapper["source_shape_count"],
        "layout": mapper["layout"],
        "features": mapper["features"],
        "source_fields": ["m", "n", "k"],
        "fallback": mapper["fallback_seeds"][0]["configuration"],
        "groups": {
            name: node(tree) for name, tree in mapper["groups"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("expand", "collapse", "update", "from-mapper"),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", type=Path, nargs="*")
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    if args.mode == "expand":
        result = expand_gemm_heuristic(source)
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    elif args.mode == "collapse":
        result = collapse_gemm_heuristic(source)
        text = json.dumps(result, separators=(",", ":")) + "\n"
    elif args.mode == "update":
        if not args.results:
            parser.error("update requires at least one result file")
        result = collapse_gemm_heuristic(
            _update(expand_gemm_heuristic(source), args.results),
        )
        text = json.dumps(result, separators=(",", ":")) + "\n"
    else:
        result = collapse_gemm_heuristic(_from_mapper(source))
        text = json.dumps(result, separators=(",", ":")) + "\n"
    args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
