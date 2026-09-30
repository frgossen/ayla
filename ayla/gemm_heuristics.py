"""Small, traceable GEMM seed heuristic distilled from the OSS shape sweep."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class GemmScheduleSeed:
    captured_config: dict[str, object]
    builder_config: dict[str, object] | None
    adaptations: tuple[str, ...]
    source_shape: tuple[int, int, int] | None
    source_percentage_of_torch: float | None


@dataclass(frozen=True)
class GemmHeuristicSelection:
    leaf: str
    decision_path: tuple[str, ...]
    seeds: tuple[GemmScheduleSeed, ...]


def expand_gemm_heuristic(compact: dict[str, Any]) -> dict[str, Any]:
    """Expand the compact field-table representation into named objects."""
    fields = compact["fields"]
    source_fields = compact["source_fields"]

    def config(index: int) -> dict[str, object]:
        mask, *values = compact["configs"][index]
        present = [
            field for bit, field in enumerate(fields) if mask & (1 << bit)
        ]
        return dict(zip(present, values, strict=True))

    def node(value: list[Any]) -> dict[str, Any]:
        if isinstance(value[0], int):
            return {
                "feature": compact["features"][value[0]],
                "threshold": value[1],
                "left": node(value[2]),
                "right": node(value[3]),
            }
        return {
            "leaf": value[0],
            "seeds": [
                {
                    "configuration": config(seed[0]),
                    "source": dict(zip(source_fields, seed[1], strict=True)),
                    "source_percentage_of_torch": seed[2],
                }
                for seed in value[1]
            ],
        }

    return {
        "version": compact["version"],
        "workload": compact["workload"],
        "source_shape_count": compact["source_shape_count"],
        "layout": compact["layout"],
        "features": compact["features"],
        "source_fields": source_fields,
        "fallback": config(compact["fallback"]),
        "groups": {name: node(tree) for name, tree in compact["groups"].items()},
    }


def collapse_gemm_heuristic(expanded: dict[str, Any]) -> dict[str, Any]:
    """Losslessly deduplicate configurations in an expanded heuristic."""
    configurations = [expanded["fallback"]]

    def visit(value: dict[str, Any]) -> None:
        if "feature" in value:
            visit(value["left"])
            visit(value["right"])
        else:
            configurations.extend(
                seed["configuration"] for seed in value["seeds"]
            )

    for tree in expanded["groups"].values():
        visit(tree)
    fields = sorted({field for config in configurations for field in config})

    rows: list[list[object]] = []
    row_indices: dict[str, int] = {}

    def config_index(config: dict[str, object]) -> int:
        mask = sum(1 << index for index, field in enumerate(fields) if field in config)
        row = [mask, *(config[field] for field in fields if field in config)]
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if key not in row_indices:
            row_indices[key] = len(rows)
            rows.append(row)
        return row_indices[key]

    fallback = config_index(expanded["fallback"])
    feature_indices = {
        name: index for index, name in enumerate(expanded["features"])
    }

    def node(value: dict[str, Any]) -> list[Any]:
        if "feature" in value:
            return [
                feature_indices[value["feature"]],
                value["threshold"],
                node(value["left"]),
                node(value["right"]),
            ]
        return [value["leaf"], [
            [
                config_index(seed["configuration"]),
                [seed["source"][field] for field in expanded["source_fields"]],
                seed["source_percentage_of_torch"],
            ]
            for seed in value["seeds"]
        ]]

    groups = {name: node(tree) for name, tree in expanded["groups"].items()}
    return {
        "version": expanded["version"],
        "workload": expanded["workload"],
        "source_shape_count": expanded["source_shape_count"],
        "layout": expanded["layout"],
        "features": expanded["features"],
        "source_fields": expanded["source_fields"],
        "fields": fields,
        "fallback": fallback,
        "configs": rows,
        "groups": groups,
    }


@cache
def _mapper() -> dict[str, Any]:
    resource = files(__package__).joinpath("oss_gemm_heuristic.json")
    return expand_gemm_heuristic(json.loads(resource.read_text()))


def _features(m: int, n: int, k: int) -> dict[str, float]:
    return {
        "m": float(m),
        "n": float(n),
        "k": float(k),
        "aspect_m_over_n": m / n,
        "output_tiles_64": float(((m + 63) // 64) * ((n + 63) // 64)),
        "reduction_tiles_64": float((k + 63) // 64),
    }


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an integer GEMM configuration value, got {value!r}")
    return value


def _output_groups(
    config: dict[str, object], m: int, n: int,
) -> tuple[int, int, int]:
    tiles = _as_int(config["output_tiles_per_cta"])
    fast_axis = str(config["output_tile_fast_axis"])
    preferred = config["output_cta_count"]
    if preferred is None:
        groups = (tiles, 1) if fast_axis == "m" else (1, tiles)
    else:
        m_tiles = math.ceil(m / _as_int(config["output_m_tile"]))
        n_tiles = math.ceil(n / _as_int(config["output_n_tile"]))
        choices = []
        for axis, m_group, n_group in (
            *(("m", group, 1) for group in (1, 2, 3, 4, 6, 8, 16)),
            *(("n", 1, group) for group in (1, 2, 3, 4, 6, 8, 16)),
        ):
            ctas = math.ceil(m_tiles / m_group) * math.ceil(n_tiles / n_group)
            choices.append((
                abs(ctas - _as_int(preferred)), axis != fast_axis,
                ctas, m_group, n_group,
            ))
        _, _, _, *groups = min(choices)
    m_group, n_group = groups
    ctas = (
        math.ceil(math.ceil(m / _as_int(config["output_m_tile"])) / m_group)
        * math.ceil(math.ceil(n / _as_int(config["output_n_tile"])) / n_group)
    )
    return m_group, n_group, ctas


def _adapt_config(
    captured: dict[str, object], m: int, n: int,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Translate a captured search config to the current builder vocabulary."""
    if captured.get("unsafe_reuse_smem") is True:
        return None, ("unsafe shared-memory lifetime reuse is not supported",)
    if "output_m_tiles_per_cta" in captured:
        return dict(captured), ()
    m_group, n_group, ctas = _output_groups(captured, m, n)
    adaptations = []
    preferred_ctas = captured["output_cta_count"]
    if preferred_ctas is not None:
        adaptations.append(
            f"output_cta_count={preferred_ctas} -> "
            f"output groups {m_group}x{n_group} ({ctas} CTAs)"
        )
    if captured["unroll_output_tiles"] is False:
        adaptations.append("synchronized output loop -> static output grouping")

    lhs = {"l2_cache_hint": captured["lhs_l2_cache_hint"]}
    if captured["lhs_l2_promotion"] is not None:
        lhs["l2_promotion"] = captured["lhs_l2_promotion"]
    rhs = {"l2_cache_hint": captured["rhs_l2_cache_hint"]}
    if captured["multicast"] is True:
        if _as_int(captured["cluster_n_ctas"]) > 1:
            lhs["multicast"] = True
        if _as_int(captured["cluster_m_ctas"]) > 1:
            rhs["multicast"] = True
    builder: dict[str, object] = {
        "output_m_tiles_per_cta": m_group,
        "output_n_tiles_per_cta": n_group,
        "output_m_tile": captured["output_m_tile"],
        "output_n_tile": captured["output_n_tile"],
        "warpgroup_n_tile": captured["warpgroup_n_tile"],
        "operand_stages": captured["operand_stages"],
        "operand_k_tile": captured["operand_k_tile"],
        "output_slices": captured["output_slices"],
        "output_store_buffers": captured["output_store_buffers"],
        "output_tma_inner_size": captured["output_tma_inner_size"],
        "output_store": captured["output_store"],
        "operand_move_attrs": {
            "lhs": lhs,
            "rhs": rhs,
            "dst": {"l2_cache_hint": captured["dst_l2_cache_hint"]},
        },
        "ptx_optimization_level": captured["ptx_optimization_level"],
    }
    optional = (
        "warpgroup_m_tile", "warpgroup_fast_axis", "rhs_tma_load_slices",
        "cluster_m_ctas", "cluster_n_ctas",
    )
    for key in optional:
        value = captured[key]
        if value not in (None, 1, "n"):
            builder[key] = value
    return builder, tuple(adaptations)


def _legal(config: dict[str, object], m: int, n: int, k: int) -> bool:
    m_tile = _as_int(config["output_m_tile"])
    n_tile = _as_int(config["output_n_tile"])
    k_tile = _as_int(config["operand_k_tile"])
    m_group = _as_int(config["output_m_tiles_per_cta"])
    n_group = _as_int(config["output_n_tiles_per_cta"])
    return (
        m >= m_tile
        and n >= n_tile
        and k % k_tile == 0
        and math.ceil(m / m_tile) >= m_group
        and math.ceil(n / n_tile) >= n_group
    )


def select_gemm_schedules(
    m: int,
    n: int,
    k: int,
    *,
    dtype: str = "float16",
    layout: str = "canonical_nt",
) -> GemmHeuristicSelection:
    """Return nearby captured configurations and current-library adaptations."""
    if min(m, n, k) <= 0:
        raise ValueError("GEMM dimensions must be positive")
    mapper = _mapper()
    if layout != mapper["layout"]:
        raise ValueError(f"no GEMM heuristic for layout {layout}")
    group = f"{dtype}/{'k128' if k % 128 == 0 else 'k64'}"
    try:
        node = mapper["groups"][group]
    except KeyError as error:
        raise ValueError(f"no GEMM heuristic for {group}") from error

    values = _features(m, n, k)
    path = []
    while "feature" in node:
        feature = node["feature"]
        threshold = float(node["threshold"])
        branch = "left" if values[feature] <= threshold else "right"
        path.append(f"{feature}={values[feature]:g} {branch} of {threshold:g}")
        node = node[branch]

    seeds = []
    for seed in node["seeds"]:
        captured = dict(seed["configuration"])
        builder, adaptations = _adapt_config(captured, m, n)
        if builder is not None and not _legal(builder, m, n, k):
            continue
        source = seed["source"]
        seeds.append(GemmScheduleSeed(
            captured_config=captured,
            builder_config=builder,
            adaptations=adaptations,
            source_shape=(source["m"], source["n"], source["k"]),
            source_percentage_of_torch=seed["source_percentage_of_torch"],
        ))

    if not any(seed.builder_config is not None for seed in seeds):
        captured = dict(mapper["fallback"])
        builder, adaptations = _adapt_config(captured, m, n)
        if builder is not None and _legal(builder, m, n, k):
            seeds.append(GemmScheduleSeed(
                captured, builder, adaptations, None, None,
            ))
    return GemmHeuristicSelection(node["leaf"], tuple(path), tuple(seeds))
