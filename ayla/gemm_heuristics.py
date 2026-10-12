"""Small, traceable GEMM seed heuristic distilled from the OSS shape sweep."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

from .gpu_dialect import CollectiveAxis


@dataclass(frozen=True)
class GemmScheduleSeed:
    captured_config: dict[str, object]
    builder_config: dict[str, object] | None
    adaptations: tuple[str, ...]
    source_shape: tuple[int, int, int] | None
    source_percentage_of_torch: float | None
    post_builder_passes: tuple[str, ...] = ()
    transpose_output: bool = False
    fixed_output_unroll_factor: int | None = None
    pipeline_output_waves: bool = False


@dataclass(frozen=True)
class GemmHeuristicSelection:
    leaf: str
    decision_path: tuple[str, ...]
    seeds: tuple[GemmScheduleSeed, ...]


def apply_gemm_schedule_seed_post_build(
    module: Any, seed: GemmScheduleSeed,
) -> Any:
    """Apply the measured post-builder pass sequence required by ``seed``."""
    if seed.pipeline_output_waves:
        if (
            seed.fixed_output_unroll_factor is not None
            or seed.builder_config is None
            or seed.builder_config.get("output_grid_size") is None
        ):
            raise ValueError("wave pipelining requires a rolled fixed grid")
        from .base.ir import find_ancestor_of_type, find_ops_of_type
        from .gpu_dialect import AsyncOp, AwaitOp, ForOp
        from .passes.canonicalize import Canonicalize
        from .passes.pipeline_loop import PipelineLoop
        from .passes.schedule_async import ScheduleAsync

        outer_loops = [
            loop for loop in find_ops_of_type(module, ForOp)
            if find_ancestor_of_type(loop, ForOp) is None
        ]
        if len(outer_loops) != 1:
            raise ValueError("wave pipelining requires one outer tile loop")
        outer_loops[0].attrs["label"] = "heuristic_output_grid"
        for body_op in outer_loops[0].body.ops:
            if isinstance(body_op, AwaitOp):
                break
            if isinstance(body_op, AsyncOp) and re.fullmatch(
                r"mm_(lhs|rhs)_smem_iter\d+_async",
                str(body_op.attrs.get("label", "")),
            ):
                body_op.attrs["label"] = "heuristic_wave_prologue_async"
        module = PipelineLoop(
            label_filter_loop="heuristic_output_grid",
            label_filter_op=r"heuristic_wave_prologue_async",
            pipelining_offset=1,
        ).run(module)
        module = ScheduleAsync().run(module)
        module = Canonicalize().run(module)
    output_was_unrolled = (
        seed.fixed_output_unroll_factor is not None
        and seed.fixed_output_unroll_factor > 1
    )
    if output_was_unrolled:
        from .base.ir import find_ancestor_of_type, find_ops_of_type
        from .gpu_dialect import ForOp
        from .passes.canonicalize import Canonicalize
        from .passes.licm import LICM
        from .passes.unroll_loop import UnrollLoop

        module = LICM().run(module)
        module = LICM().run(module)
        outer_loops = [
            loop for loop in find_ops_of_type(module, ForOp)
            if find_ancestor_of_type(loop, ForOp) is None
        ]
        if len(outer_loops) != 1:
            raise ValueError(
                "fixed output-grid unrolling requires one outer tile loop"
            )
        outer_loops[0].attrs["label"] = "heuristic_output_grid"
        module = UnrollLoop(
            unroll_factor=seed.fixed_output_unroll_factor,
            label_filter="heuristic_output_grid",
        ).run(module)
        module = Canonicalize().run(module)
    expected = ("schedule_async", "canonicalize")
    if seed.post_builder_passes and seed.post_builder_passes != expected:
        raise ValueError(
            f"unsupported GEMM post-builder passes: {seed.post_builder_passes}"
        )
    if not output_was_unrolled and not seed.post_builder_passes:
        return module
    # Keep pass imports lazy so selecting a heuristic remains lightweight.
    from .passes.canonicalize import Canonicalize
    from .passes.schedule_async import ScheduleAsync

    return Canonicalize().run(ScheduleAsync().run(module))


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

    exact = [
        {
            "operation": row[0],
            "dtype": row[1],
            "batch": row[2],
            "m": row[3],
            "n": row[4],
            "k": row[5],
            "seed": {
                "configuration": config(row[6]),
                "source": dict(zip(source_fields, row[7], strict=True)),
                "source_percentage_of_torch": row[8],
            },
        }
        for row in compact.get("exact", [])
    ]
    return {
        "version": compact["version"],
        "workload": compact["workload"],
        "source_shape_count": compact["source_shape_count"],
        "layout": compact["layout"],
        "features": compact["features"],
        "source_fields": source_fields,
        "fallback": config(compact["fallback"]),
        "exact": exact,
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
    configurations.extend(
        entry["seed"]["configuration"]
        for entry in expanded.get("exact", [])
    )
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
    exact = [
        [
            entry["operation"], entry["dtype"], entry["batch"],
            entry["m"], entry["n"], entry["k"],
            config_index(entry["seed"]["configuration"]),
            [
                entry["seed"]["source"][field]
                for field in expanded["source_fields"]
            ],
            entry["seed"]["source_percentage_of_torch"],
        ]
        for entry in expanded.get("exact", [])
    ]
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
        "exact": exact,
        "groups": groups,
    }


@cache
def _mapper() -> dict[str, Any]:
    resource = files(__package__).joinpath("oss_gemm_heuristic.json")
    return expand_gemm_heuristic(json.loads(resource.read_text()))


def _features(
    m: int, n: int, k: int, batch: int = 1,
) -> dict[str, float]:
    return {
        "batch": float(batch),
        "m": float(m),
        "n": float(n),
        "k": float(k),
        "aspect_m_over_n": m / n,
        "output_tiles_64": float(
            batch * ((m + 63) // 64) * ((n + 63) // 64)
        ),
        "reduction_tiles_64": float((k + 63) // 64),
    }


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an integer GEMM configuration value, got {value!r}")
    return value


def _output_groups(
    config: dict[str, object], m: int, n: int,
) -> tuple[int, int, int]:
    if "output_n_tiles_per_cta" in config:
        m_group = _as_int(config.get(
            "output_m_tiles_per_cta",
            config.get("output_tiles_per_cta", 1),
        ))
        n_group = _as_int(config["output_n_tiles_per_cta"])
        ctas = (
            math.ceil(
                math.ceil(m / _as_int(config["output_m_tile"])) / m_group,
            )
            * math.ceil(
                math.ceil(n / _as_int(config["output_n_tile"])) / n_group,
            )
        )
        return m_group, n_group, ctas

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
    captured: dict[str, object], m: int, n: int, k: int, *,
    preserve_output_grid: bool = False,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Translate a captured search config to the current builder vocabulary."""
    if captured.get("unsafe_reuse_smem", False) is True:
        return None, ("unsafe shared-memory lifetime reuse is not supported",)
    if "output_m_tiles_per_cta" in captured:
        return dict(captured), ()
    preferred_ctas = captured["output_cta_count"]
    if preferred_ctas is None:
        m_group, n_group, ctas = _output_groups(captured, m, n)
    elif preserve_output_grid:
        m_group, n_group = 1, 1
        ctas = _as_int(preferred_ctas)
    else:
        m_group, n_group = 1, 1
        m_tiles = math.ceil(m / _as_int(captured["output_m_tile"]))
        n_tiles = math.ceil(n / _as_int(captured["output_n_tile"]))
        logical_tiles = m_tiles * n_tiles
        divisors = [
            count for count in range(64, 265)
            if logical_tiles % count == 0
            and logical_tiles // count <= 8
            and not (k > 2048 and logical_tiles // count > 2)
        ]
        ctas = (
            min(divisors, key=lambda count: abs(count - _as_int(preferred_ctas)))
            if divisors else _as_int(preferred_ctas)
        )
    adaptations = []
    if preferred_ctas is not None:
        adaptations.append(
            f"output_cta_count={preferred_ctas} -> "
            f"fixed output grid of {ctas} CTAs"
        )
    if captured["unroll_output_tiles"] is False:
        adaptations.append(
            "retain synchronized grid-stride output loop"
            if preferred_ctas is not None
            else "synchronized output loop -> static output grouping"
        )

    lhs = {"l2_cache_hint": captured["lhs_l2_cache_hint"]}
    if captured.get("lhs_l2_promotion") is not None:
        lhs["l2_promotion"] = captured["lhs_l2_promotion"]
    rhs = {"l2_cache_hint": captured["rhs_l2_cache_hint"]}
    if captured.get("rhs_l2_promotion") is not None:
        rhs["l2_promotion"] = captured["rhs_l2_promotion"]
    if captured.get("multicast", False) is True:
        if _as_int(captured.get("cluster_n_ctas", 1)) > 1:
            lhs["multicast"] = True
        if _as_int(captured.get("cluster_m_ctas", 1)) > 1:
            rhs["multicast"] = True
    builder: dict[str, object] = {
        "output_m_tiles_per_cta": m_group,
        "output_n_tiles_per_cta": n_group,
        "output_m_tile": captured["output_m_tile"],
        "output_n_tile": captured["output_n_tile"],
        "warpgroup_n_tile": captured["warpgroup_n_tile"],
        "operand_stages": captured["operand_stages"],
        "operand_k_tile": captured["operand_k_tile"],
        "cyclic_operand_pipeline": captured.get(
            "cyclic_operand_pipeline", False,
        ),
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
    if preferred_ctas is not None:
        builder["output_grid_size"] = ctas
        builder["output_grid_fast_axis"] = captured.get(
            "output_tile_fast_axis", "n",
        )
    if captured.get("swap_output_grid_axes") is True:
        builder["output_grid_axes"] = (
            CollectiveAxis.Y, CollectiveAxis.X,
        )
    optional = (
        "warpgroup_m_tile", "warpgroup_fast_axis", "rhs_tma_load_slices",
        "cluster_m_ctas", "cluster_n_ctas",
    )
    for key in optional:
        value = captured.get(key)
        if value not in (None, 1, "n"):
            builder[key] = value
    return builder, tuple(adaptations)


def _legal(
    config: dict[str, object],
    m: int,
    n: int,
    k: int,
    *,
    allow_partial_output_tiles: bool = False,
) -> bool:
    m_tile = _as_int(config["output_m_tile"])
    n_tile = _as_int(config["output_n_tile"])
    k_tile = _as_int(config["operand_k_tile"])
    m_group = _as_int(config["output_m_tiles_per_cta"])
    n_group = _as_int(config["output_n_tiles_per_cta"])
    blocked_output = (
        config["output_store"] != "scalar"
        and config["output_tma_inner_size"] is not None
    )
    return (
        (allow_partial_output_tiles or (m >= m_tile and n >= n_tile))
        and k % k_tile == 0
        and (not blocked_output or (n % 64 == 0 and n_tile % 64 == 0))
        and math.ceil(m / m_tile) >= m_group
        and math.ceil(n / n_tile) >= n_group
    )


def select_gemm_schedules(
    m: int,
    n: int,
    k: int,
    *,
    operation: str = "mm",
    batch: int = 1,
    dtype: str = "float16",
    layout: str = "canonical_nt",
) -> GemmHeuristicSelection:
    """Return nearby captured configurations and current-library adaptations."""
    if min(m, n, k) <= 0:
        raise ValueError("GEMM dimensions must be positive")
    if batch <= 0:
        raise ValueError("GEMM batch must be positive")
    mapper = _mapper()
    if layout != mapper["layout"]:
        raise ValueError(f"no GEMM heuristic for layout {layout}")
    if operation not in {"mm", "addmm", "bmm"}:
        raise ValueError(f"no GEMM heuristic for operation {operation}")
    if operation != "bmm" and batch != 1:
        raise ValueError("batch is only meaningful for BMM heuristics")
    exact = next((
        entry for entry in mapper.get("exact", [])
        if (
            entry["operation"], entry["dtype"], entry["batch"],
            entry["m"], entry["n"], entry["k"],
        ) == (operation, dtype, batch, m, n, k)
    ), None)
    group = (
        f"{operation}/{dtype}/{'k128' if k % 128 == 0 else 'k64'}"
    )
    try:
        node = mapper["groups"][group]
    except KeyError as error:
        raise ValueError(f"no GEMM heuristic for {group}") from error

    path = []
    if exact is not None:
        node = {
            "leaf": f"exact:{operation}:{batch}:{m}:{n}:{k}",
            "seeds": [exact["seed"]],
        }
    else:
        values = _features(m, n, k, batch)
        while "feature" in node:
            feature = node["feature"]
            threshold = float(node["threshold"])
            branch = "left" if values[feature] <= threshold else "right"
            path.append(
                f"{feature}={values[feature]:g} {branch} of {threshold:g}"
            )
            node = node[branch]

    seeds = []
    for seed in node["seeds"]:
        captured = dict(seed["configuration"])
        transpose_output = bool(captured.get("transpose_output", False))
        builder_m, builder_n = (n, m) if transpose_output else (m, n)
        builder, adaptations = _adapt_config(
            captured, builder_m, builder_n, k,
            preserve_output_grid=exact is not None,
        )
        if exact is None and builder is not None and not _legal(
            builder,
            builder_m,
            builder_n,
            k,
            allow_partial_output_tiles=operation == "bmm",
        ):
            continue
        unroll_factor = None
        if (
            builder is not None
            and builder.get("output_grid_size") is not None
            and captured.get("unroll_output_tiles", False) is True
        ):
            logical_tiles = (
                math.ceil(builder_m / _as_int(builder["output_m_tile"]))
                * math.ceil(builder_n / _as_int(builder["output_n_tile"]))
            )
            unroll_factor = math.ceil(
                logical_tiles / _as_int(builder["output_grid_size"]),
            )
        source = seed["source"]
        seeds.append(GemmScheduleSeed(
            captured_config=captured,
            builder_config=builder,
            adaptations=adaptations,
            source_shape=(source["m"], source["n"], source["k"]),
            source_percentage_of_torch=seed["source_percentage_of_torch"],
            post_builder_passes=(
                ("schedule_async", "canonicalize")
                if captured.get("reschedule_async") is True else ()
            ),
            transpose_output=transpose_output,
            fixed_output_unroll_factor=unroll_factor,
            pipeline_output_waves=bool(
                captured.get("pipeline_output_waves", False)
            ),
        ))

    if not any(seed.builder_config is not None for seed in seeds):
        captured = dict(mapper["fallback"])
        builder, adaptations = _adapt_config(captured, m, n, k)
        if builder is not None and _legal(
            builder,
            m,
            n,
            k,
            allow_partial_output_tiles=operation == "bmm",
        ):
            seeds.append(GemmScheduleSeed(
                captured, builder, adaptations, None, None,
                transpose_output=bool(
                    captured.get("transpose_output", False)
                ),
            ))
    return GemmHeuristicSelection(node["leaf"], tuple(path), tuple(seeds))
