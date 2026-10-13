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


def _update(
    expanded: dict,
    paths: list[Path],
    *,
    max_seeds_per_leaf: int = 3,
) -> dict:
    def family(seed: dict) -> str:
        return json.dumps(
            _without_names(seed["configuration"]),
            sort_keys=True,
            separators=(",", ":"),
        )

    touched: dict[int, tuple[dict, set[str]]] = {}
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
            new_seed = {
                "configuration": config,
                "source": source,
                "source_percentage_of_torch": score,
            }
            leaf["seeds"].append(new_seed)
            _, updated_families = touched.setdefault(
                id(leaf), (leaf, set()),
            )
            updated_families.add(family(new_seed))

    for leaf, updated_families in touched.values():
        best = {}
        for seed in leaf["seeds"]:
            key = family(seed)
            incumbent = best.get(key)
            if incumbent is None or seed["source_percentage_of_torch"] > (
                incumbent["source_percentage_of_torch"]
            ):
                best[key] = seed
        leaf["seeds"] = sorted(
            best.values(),
            key=lambda seed: (
                seed["configuration"].get("unsafe_reuse_smem") is True,
                family(seed) not in updated_families,
                -seed["source_percentage_of_torch"],
            ),
        )[:max_seeds_per_leaf]
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
        "exact": [
            {
                **entry["shape"],
                "seed": {
                    "configuration": entry["seed"]["configuration"],
                    "source": entry["seed"]["source_shape"],
                    "source_percentage_of_torch": (
                        entry["seed"]["source_percentage_of_torch"]
                    ),
                },
            }
            for entry in mapper.get("exact", [])
        ],
        "groups": {
            name: node(tree) for name, tree in mapper["groups"].items()
        },
    }


def _merge_mapper(expanded: dict, mapper: dict) -> dict:
    """Add disjoint operation/dtype groups and their measured exact winners."""
    addition = _from_mapper(mapper)
    for field in ("version", "workload", "layout", "source_fields"):
        if addition[field] != expanded[field]:
            raise ValueError(
                f"incompatible heuristic {field}: "
                f"{addition[field]!r} != {expanded[field]!r}"
            )
    for name in addition["groups"]:
        if name in expanded["groups"]:
            raise ValueError(f"heuristic group already exists: {name}")
    exact = [*expanded.get("exact", []), *addition.get("exact", [])]
    keys = set()
    for entry in exact:
        key = tuple(entry[field] for field in (
            "operation", "dtype", "batch", "m", "n", "k",
        ))
        if key in keys:
            raise ValueError(f"exact GEMM winner already exists: {key}")
        keys.add(key)
    for field in addition["features"]:
        if field not in expanded["features"]:
            expanded["features"].append(field)
    for name, tree in addition["groups"].items():
        expanded["groups"][name] = tree
    expanded["exact"] = exact
    expanded["source_shape_count"] += addition["source_shape_count"]
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("expand", "collapse", "update", "from-mapper", "merge-mapper"),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", type=Path, nargs="*")
    parser.add_argument("--max-seeds-per-leaf", type=int, default=3)
    args = parser.parse_args()
    if args.max_seeds_per_leaf < 1:
        parser.error("--max-seeds-per-leaf must be positive")

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
            _update(
                expand_gemm_heuristic(source),
                args.results,
                max_seeds_per_leaf=args.max_seeds_per_leaf,
            ),
        )
        text = json.dumps(result, separators=(",", ":")) + "\n"
    elif args.mode == "from-mapper":
        result = collapse_gemm_heuristic(_from_mapper(source))
        text = json.dumps(result, separators=(",", ":")) + "\n"
    else:
        if len(args.results) != 1:
            parser.error("merge-mapper requires exactly one mapper file")
        mapper = json.loads(args.results[0].read_text())
        result = collapse_gemm_heuristic(
            _merge_mapper(expand_gemm_heuristic(source), mapper),
        )
        text = json.dumps(result, separators=(",", ":")) + "\n"
    args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
