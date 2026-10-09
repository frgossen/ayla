#!/usr/bin/env python3
"""Build a lossless measured OSS GEMM winner mapper with a compact fallback.

Every measured shape is dispatched to its exact winning configuration.  A
shallow tree of nearby exemplars is retained only for shapes outside the
measured corpus.  Querying the mapper reports whether exact dispatch or the
fallback tree was used.

Examples::

    pixi run python -m benchmarks.generate_oss_gemm_mapper
    pixi run python -m benchmarks.generate_oss_gemm_mapper \
        --query 8192 768 768
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.oss_gemm_corpus import (
    OUTPUT_DIR,
    ScheduleConfig,
    fallback_config,
    write_json,
)


DEFAULT_INPUT = OUTPUT_DIR / "shape_improvements.json"
DEFAULT_OUTPUT = Path(__file__).with_name("oss_gemm_minimal_heuristics.json")


@dataclass(frozen=True)
class Point:
    shape_id: str
    op: str
    m: int
    n: int
    k: int
    dtype: str
    frequency: int
    percentage: float
    config: ScheduleConfig
    batch: int = 1


Feature = tuple[str, Callable[[Point], float]]
FEATURES: tuple[Feature, ...] = (
    ("batch", lambda point: float(point.batch)),
    ("m", lambda point: float(point.m)),
    ("n", lambda point: float(point.n)),
    ("k", lambda point: float(point.k)),
    ("aspect_m_over_n", lambda point: point.m / point.n),
    (
        "output_tiles_64",
        lambda point: float(
            point.batch
            * ((point.m + 63) // 64)
            * ((point.n + 63) // 64)
        ),
    ),
    ("reduction_tiles_64", lambda point: float((point.k + 63) // 64)),
)
FEATURE_BY_NAME = dict(FEATURES)

_LEGACY_CONFIG_DEFAULTS: dict[str, object] = {
    "output_tiles_per_cta": 1,
    "output_m_tile": 128,
    "output_n_tile": 256,
    "operand_stages": 3,
    "operand_k_tile": 64,
    "output_slices": 1,
    "output_store_buffers": 1,
    "output_tma_inner_size": 128,
    "output_store": "stmatrix",
    "lhs_l2_cache_hint": "evict_first",
    "rhs_l2_cache_hint": "evict_first",
    "dst_l2_cache_hint": "evict_first",
}

# Historical result rows retain experimental controls that the simplified
# library deliberately does not expose.  They must neither split mapper
# families nor leak back into the production heuristic.
_REDUNDANT_CONFIG_FIELDS = {
    "output_tile_fast_axis",
    "rhs_tma_load_slices",
    "unsafe_reuse_smem",
    "warpgroup_fast_axis",
    "wgmma_epilogue_pipeline",
    "wgmma_group_size",
    "wgmma_pending_groups",
}


def _production_config(config: ScheduleConfig) -> dict[str, object]:
    return {
        key: value
        for key, value in dataclasses.asdict(config).items()
        if key not in _REDUNDANT_CONFIG_FIELDS
    }


def _exact_entry(point: Point) -> dict[str, object]:
    config = _production_config(point.config)
    # Names are diagnostic only.  A common name lets the compact serializer
    # deduplicate genuinely identical winning configurations.
    config["name"] = "heuristic_exact_winner"
    return {
        "shape": {
            "operation": point.op,
            "dtype": point.dtype,
            "batch": point.batch,
            "m": point.m,
            "n": point.n,
            "k": point.k,
        },
        "seed": {
            "rank": 1,
            "family_support": 1,
            "source_shape_id": point.shape_id,
            "source_shape": {
                "batch": point.batch,
                "m": point.m,
                "n": point.n,
                "k": point.k,
            },
            "source_percentage_of_torch": point.percentage,
            "source_configuration_name": point.config.name,
            "configuration": config,
        },
    }


def _config_from_mapping(mapping: dict[str, Any]) -> ScheduleConfig:
    values: dict[str, object] = {}
    for field in dataclasses.fields(ScheduleConfig):
        if field.name in mapping:
            values[field.name] = mapping[field.name]
        elif field.default is not dataclasses.MISSING:
            values[field.name] = field.default
        elif field.name == "warpgroup_n_tile":
            values[field.name] = int(mapping.get("output_n_tile", 256))
        elif field.name in _LEGACY_CONFIG_DEFAULTS:
            values[field.name] = _LEGACY_CONFIG_DEFAULTS[field.name]
        else:
            raise ValueError(f"configuration is missing {field.name!r}")
    return ScheduleConfig(**values)  # type: ignore[arg-type]


def _family(config: ScheduleConfig) -> tuple[object, ...]:
    """A reusable family; exact CTA count is adapted independently."""
    return tuple(
        (field.name, getattr(config, field.name))
        for field in dataclasses.fields(ScheduleConfig)
        if field.name not in {
            "name", "output_cta_count", *_REDUNDANT_CONFIG_FIELDS,
        }
    ) + (("uses_exact_ctas", config.output_cta_count is not None),)


def _alignment(k: int) -> str:
    return "k128" if k % 128 == 0 else "k64"


def _group_key(op: str, dtype: str, k: int) -> str:
    return f"{op}/{dtype}/{_alignment(k)}"


def _load_points(path: Path) -> list[Point]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list")
    points = []
    for row in rows:
        config = row.get("configuration")
        if not isinstance(config, dict):
            raise ValueError(f"{row.get('shape_id')}: missing configuration")
        points.append(Point(
            shape_id=str(row["shape_id"]),
            op=str(row.get("op", "mm")),
            batch=int(row.get("batch") or 1),
            m=int(row["m"]), n=int(row["n"]), k=int(row["k"]),
            dtype=str(row["dtype"]), frequency=int(row["frequency"]),
            percentage=float(row["percentage_of_torch"]),
            config=_config_from_mapping(config),
        ))
    return points


def _gini(points: list[Point]) -> float:
    counts = Counter(_family(point.config) for point in points)
    total = len(points)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _best_split(
    points: list[Point], min_leaf_shapes: int,
) -> tuple[str, float, list[Point], list[Point], float] | None:
    parent_impurity = _gini(points)
    best: tuple[str, float, list[Point], list[Point], float] | None = None
    for name, getter in FEATURES:
        values = sorted({getter(point) for point in points})
        for lower, upper in zip(values, values[1:]):
            threshold = (lower + upper) / 2
            left = [point for point in points if getter(point) <= threshold]
            if len(left) < min_leaf_shapes:
                continue
            right = [point for point in points if getter(point) > threshold]
            if len(right) < min_leaf_shapes:
                continue
            weighted = (
                len(left) * _gini(left) + len(right) * _gini(right)
            ) / len(points)
            gain = parent_impurity - weighted
            if best is None or gain > best[4]:
                best = name, threshold, left, right, gain
    return best


def _feature_vector(point: Point) -> tuple[float, float, float, float, float]:
    return (
        math.log2(point.batch), math.log2(point.m), math.log2(point.n),
        math.log2(point.k),
        math.log2(point.m / point.n),
    )


def _distance(lhs: Point, rhs: Point) -> float:
    return math.sqrt(sum(
        (left - right) ** 2
        for left, right in zip(
            _feature_vector(lhs), _feature_vector(rhs), strict=True,
        )
    ))


def _leaf(
    points: list[Point], seeds_per_leaf: int, leaf_id: str,
) -> dict[str, object]:
    families = Counter(_family(point.config) for point in points)
    seeds = []
    selected_families = []
    required = [
        lambda config: config.transpose_output,
        lambda config: config.rhs_l2_promotion is not None,
        lambda config: (
            config.output_tiles_per_cta == 5
            or config.output_n_tiles_per_cta == 5
        ),
        lambda config: config.warpgroup_m_tile is not None,
    ]
    uncovered = {
        index for index, predicate in enumerate(required)
        if any(predicate(point.config) for point in points)
    }
    family_config = {
        _family(point.config): point.config for point in points
    }
    while uncovered and len(selected_families) < seeds_per_leaf:
        family, support = max(
            families.items(),
            key=lambda item: (
                sum(
                    required[index](family_config[item[0]])
                    for index in uncovered
                ),
                item[1],
            ),
        )
        covered = {
            index for index in uncovered
            if required[index](family_config[family])
        }
        if not covered:
            break
        selected_families.append((family, support))
        uncovered -= covered
    for candidate in families.most_common():
        if len(selected_families) >= seeds_per_leaf:
            break
        if candidate[0] not in {family for family, _ in selected_families}:
            selected_families.append(candidate)
    for rank, (family, support) in enumerate(selected_families, 1):
        members = [point for point in points if _family(point.config) == family]
        exemplar = min(
            members,
            key=lambda candidate: sum(
                _distance(candidate, other) for other in members
            ),
        )
        config = _production_config(exemplar.config)
        config["name"] = f"heuristic_{leaf_id}_seed{rank}"
        seeds.append({
            "rank": rank,
            "family_support": support,
            "source_shape_id": exemplar.shape_id,
            "source_shape": {
                "batch": exemplar.batch,
                "m": exemplar.m, "n": exemplar.n, "k": exemplar.k,
            },
            "source_percentage_of_torch": exemplar.percentage,
            "source_configuration_name": exemplar.config.name,
            "configuration": config,
        })
    return {
        "kind": "leaf",
        "id": leaf_id,
        "training_shapes": len(points),
        "ranges": {
            axis: [min(getattr(point, axis) for point in points),
                   max(getattr(point, axis) for point in points)]
            for axis in ("m", "n", "k")
        },
        "seeds": seeds,
    }


def _build_tree(
    points: list[Point],
    *,
    leaf_budget: int,
    min_leaf_shapes: int,
    seeds_per_leaf: int,
    leaf_prefix: str,
) -> dict[str, object]:
    if leaf_budget <= 1:
        return _leaf(points, seeds_per_leaf, leaf_prefix)
    split = _best_split(points, min_leaf_shapes)
    if split is None or split[4] <= 0:
        return _leaf(points, seeds_per_leaf, leaf_prefix)
    feature, threshold, left, right, gain = split
    left_budget = max(1, round(leaf_budget * len(left) / len(points)))
    left_budget = min(left_budget, leaf_budget - 1)
    right_budget = leaf_budget - left_budget
    return {
        "kind": "split",
        "feature": feature,
        "threshold": threshold,
        "information_gain": gain,
        "training_shapes": len(points),
        "left": _build_tree(
            left,
            leaf_budget=left_budget,
            min_leaf_shapes=min_leaf_shapes,
            seeds_per_leaf=seeds_per_leaf,
            leaf_prefix=f"{leaf_prefix}L",
        ),
        "right": _build_tree(
            right,
            leaf_budget=right_budget,
            min_leaf_shapes=min_leaf_shapes,
            seeds_per_leaf=seeds_per_leaf,
            leaf_prefix=f"{leaf_prefix}R",
        ),
    }


def _allocate_leaf_budgets(
    groups: dict[str, list[Point]], total_leaves: int,
) -> dict[str, int]:
    if total_leaves < len(groups):
        raise ValueError("max leaves must be at least the number of groups")
    budgets = {key: 1 for key in groups}
    remaining = total_leaves - len(groups)
    while remaining:
        key = max(
            groups,
            key=lambda name: len(groups[name]) / (budgets[name] + 1),
        )
        budgets[key] += 1
        remaining -= 1
    return budgets


def _walk_leaves(node: dict[str, Any]):
    if node["kind"] == "leaf":
        yield node
        return
    yield from _walk_leaves(node["left"])
    yield from _walk_leaves(node["right"])


def build_mapper(
    points: list[Point],
    *,
    max_leaves: int = 16,
    min_leaf_shapes: int = 6,
    seeds_per_leaf: int = 3,
) -> dict[str, object]:
    exact = [_exact_entry(point) for point in points]
    exact_keys = {
        (
            point.op, point.dtype, point.batch,
            point.m, point.n, point.k,
        )
        for point in points
    }
    if len(exact_keys) != len(points):
        raise ValueError(
            "measured winners do not have unique dispatch dimensions"
        )
    groups: dict[str, list[Point]] = {}
    for point in points:
        groups.setdefault(
            _group_key(point.op, point.dtype, point.k), [],
        ).append(point)
    budgets = _allocate_leaf_budgets(groups, max_leaves)
    trees = {
        key: _build_tree(
            group,
            leaf_budget=budgets[key],
            min_leaf_shapes=min_leaf_shapes,
            seeds_per_leaf=seeds_per_leaf,
            leaf_prefix=key.replace("/", "_") + "_",
        )
        for key, group in sorted(groups.items())
    }
    leaves = [leaf for tree in trees.values() for leaf in _walk_leaves(tree)]
    fallback = _production_config(dataclasses.replace(
        fallback_config(), output_m_tile=64,
    ))
    fallback["name"] = "heuristic_coverage_fallback"
    mapper: dict[str, Any] = {
        "version": 2,
        "purpose": "exact production winners with fallback seeds for unseen shapes",
        "source_shape_count": len(points),
        "layout": "canonical_nt",
        "features": [name for name, _ in FEATURES],
        "max_leaves": max_leaves,
        "actual_leaves": len(leaves),
        "seeds_per_leaf": seeds_per_leaf,
        "stored_seed_count": 1 + sum(len(leaf["seeds"]) for leaf in leaves),
        "exact_winner_count": len(exact),
        "exact": exact,
        "fallback_seeds": [{
            "rank": 1,
            "family_support": 0,
            "source_shape_id": None,
            "source_shape": None,
            "source_percentage_of_torch": None,
            "source_configuration_name": "tune_m64_n64_scalar_k64_s3",
            "configuration": fallback,
        }],
        "groups": trees,
    }
    queries = [
        query_mapper(
            mapper, point.m, point.n, point.k, point.dtype, point.op,
            point.batch,
        )
        for point in points
    ]
    covered = sum(bool(query["seeds"]) for query in queries)
    recalled = sum(
        _family(point.config) in {
            _family(_config_from_mapping(seed["configuration"]))
            for seed in query["seeds"]
        }
        for point, query in zip(points, queries, strict=True)
    )
    exact_recalled = sum(
        {
            key: value for key, value in query["seeds"][0]["configuration"].items()
            if key != "name"
        } == {
            key: value for key, value in _production_config(point.config).items()
            if key != "name"
        }
        for point, query in zip(points, queries, strict=True)
    )
    mapper["training_legal_seed_coverage_pct"] = 100 * covered / len(points)
    mapper["training_winner_family_recall_pct"] = 100 * recalled / len(points)
    mapper["training_winner_config_recall_pct"] = (
        100 * exact_recalled / len(points)
    )
    return mapper


def _query_features(
    m: int, n: int, k: int, batch: int = 1,
) -> dict[str, float]:
    return {
        "batch": float(batch),
        "m": float(m), "n": float(n), "k": float(k),
        "aspect_m_over_n": m / n,
        "output_tiles_64": float(
            batch * ((m + 63) // 64) * ((n + 63) // 64)
        ),
        "reduction_tiles_64": float((k + 63) // 64),
    }


def _adapt_query_config(
    config: dict[str, Any], m: int, n: int, k: int, *,
    allow_partial: bool = False,
    preserve_output_grid: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    changes = []
    m_tile = int(config["output_m_tile"])
    n_tile = int(config["output_n_tile"])
    k_tile = int(config["operand_k_tile"])
    if (not allow_partial and (m < m_tile or n < n_tile)) or k % k_tile:
        return None, ["tile does not fit or K tile does not divide K"]
    if (
        config.get("output_store") != "scalar"
        and config.get("output_tma_inner_size") is not None
        and (n % 64 or n_tile % 64)
    ):
        return None, ["blocked output TMA requires 64-column divisibility"]
    m_tiles = (m + m_tile - 1) // m_tile
    n_tiles = (n + n_tile - 1) // n_tile
    if (
        m_tiles % int(config["cluster_m_ctas"])
        or n_tiles % int(config["cluster_n_ctas"])
    ):
        return None, ["cluster shape does not divide the output grid"]
    preferred = config.get("output_cta_count")
    if preferred is not None and not preserve_output_grid:
        logical_tiles = m_tiles * n_tiles
        divisors = [
            count for count in range(64, 265)
            if logical_tiles % count == 0
            and logical_tiles // count <= 8
            and not (k > 2048 and logical_tiles // count > 2)
        ]
        if not divisors:
            return None, ["no valid exact CTA divisor near the seed"]
        adapted = min(divisors, key=lambda count: abs(count - int(preferred)))
        if adapted != preferred:
            changes.append(f"output_cta_count {preferred} -> {adapted}")
        config["output_cta_count"] = adapted
    return config, changes


def query_mapper(
    mapper: dict[str, Any], m: int, n: int, k: int, dtype: str,
    op: str = "mm", batch: int = 1,
) -> dict[str, object]:
    exact = next((
        entry for entry in mapper.get("exact", [])
        if entry["shape"] == {
            "operation": op, "dtype": dtype, "batch": batch,
            "m": m, "n": n, "k": k,
        }
    ), None)
    group = _group_key(op, dtype, k)
    if group not in mapper["groups"]:
        raise ValueError(f"no heuristic group for {group}")
    if exact is not None:
        seed = exact["seed"]
        return {
            "shape": {
                "op": op, "batch": batch,
                "m": m, "n": n, "k": k, "dtype": dtype,
            },
            "group": group,
            "decision_path": [],
            "leaf": {
                "id": f"exact:{op}:{batch}:{m}:{n}:{k}",
                "training_shapes": 1,
                "ranges": {"m": [m, m], "n": [n, n], "k": [k, k]},
            },
            "fallback_used": False,
            "exact_match": True,
            "seeds": [{
                **seed,
                "distance": 0.0,
                "adaptations": [],
                "configuration": dict(seed["configuration"]),
            }],
        }
    features = _query_features(m, n, k, batch)
    node = mapper["groups"][group]
    path = []
    while node["kind"] == "split":
        feature = str(node["feature"])
        value = features[feature]
        threshold = float(node["threshold"])
        branch = "left" if value <= threshold else "right"
        path.append({
            "feature": feature,
            "value": value,
            "comparison": "<=",
            "threshold": threshold,
            "branch": branch,
        })
        node = node[branch]

    seeds = []
    for seed in node["seeds"]:
        config, changes = _adapt_query_config(
            dict(seed["configuration"]), m, n, k,
            allow_partial=op == "bmm",
        )
        if config is None:
            continue
        source = seed["source_shape"]
        distance = math.sqrt(
            (math.log2(m / int(source["m"]))) ** 2
            + (math.log2(n / int(source["n"]))) ** 2
            + (math.log2(k / int(source["k"]))) ** 2
        )
        seeds.append({
            **{key: value for key, value in seed.items() if key != "configuration"},
            "distance": distance,
            "adaptations": changes,
            "configuration": config,
        })
    fallback_used = False
    if not seeds:
        fallback_used = True
        for seed in mapper.get("fallback_seeds", []):
            config, changes = _adapt_query_config(
                dict(seed["configuration"]), m, n, k,
                allow_partial=op == "bmm",
            )
            if config is None:
                continue
            seeds.append({
                **{
                    key: value for key, value in seed.items()
                    if key != "configuration"
                },
                "distance": None,
                "adaptations": ["leaf had no legal seed", *changes],
                "configuration": config,
            })
    seeds.sort(key=lambda seed: (
        math.inf if seed["distance"] is None else float(seed["distance"])
    ))
    return {
        "shape": {
            "op": op, "batch": batch,
            "m": m, "n": n, "k": k, "dtype": dtype,
        },
        "group": group,
        "decision_path": path,
        "leaf": {
            "id": node["id"],
            "training_shapes": node["training_shapes"],
            "ranges": node["ranges"],
        },
        "fallback_used": fallback_used,
        "exact_match": False,
        "seeds": seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-leaves", type=int, default=16)
    parser.add_argument("--min-leaf-shapes", type=int, default=6)
    parser.add_argument("--seeds-per-leaf", type=int, default=3)
    parser.add_argument(
        "--query", nargs=3, type=int, metavar=("M", "N", "K"),
        help="print the decision trace and seeds for one shape",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--op", choices=("mm", "addmm", "bmm"), default="mm")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if args.max_leaves < 1 or args.min_leaf_shapes < 1:
        parser.error("leaf limits must be positive")
    if not 1 <= args.seeds_per_leaf <= 3:
        parser.error("--seeds-per-leaf must be in [1, 3]")

    if args.query is None:
        mapper = build_mapper(
            _load_points(args.input),
            max_leaves=args.max_leaves,
            min_leaf_shapes=args.min_leaf_shapes,
            seeds_per_leaf=args.seeds_per_leaf,
        )
        mapper["source_artifact"] = str(args.input.resolve())
        write_json(mapper, args.output)
        print(
            f"wrote {mapper['exact_winner_count']} exact winners, "
            f"{mapper['actual_leaves']} fallback leaves, and "
            f"{mapper['stored_seed_count']} fallback seeds to "
            f"{args.output.resolve()}"
        )
        return 0

    if args.output.exists():
        mapper = json.loads(args.output.read_text())
    else:
        mapper = build_mapper(_load_points(args.input))
    print(json.dumps(
        query_mapper(mapper, *args.query, args.dtype, args.op, args.batch),
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
