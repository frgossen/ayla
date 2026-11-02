"""Load measured Better-Benchmark GEMM routes for native Inductor integration.

The packaged JSON is the source of configurations and layout policies.
Keys are (operator, input dtype, batch, M, N, K); None retains Inductor.
"""

import json
from pathlib import Path


ROUTER_PATH = Path(__file__).with_name("better_benchmark_gemm_router.json")
_SIGNATURE_FIELDS = ("op", "dtype", "batch", "m", "n", "k")


def _signature(row):
    key = tuple(row[field] for field in _SIGNATURE_FIELDS)
    if (key[0] not in ("mm", "addmm", "bmm")
            or key[1] not in ("bfloat16", "float16")
            or any(type(size) is not int or size <= 0 for size in key[2:])):
        raise ValueError(f"Invalid BF16/FP16 GEMM signature: {key}")
    return key


def _load_router(path):
    document = json.loads(path.read_text())
    if document.get("version") != 1 or document.get("fallback") != "table":
        raise ValueError("Expected router version=1 and fallback=table")
    routes = {}
    for row in document["routes"]:
        key = _signature(row)
        if key in routes:
            raise ValueError(f"Duplicate GEMM route: {key}")
        config = row["configuration"]
        if config is not None and not isinstance(config, dict):
            raise ValueError(f"Expected configuration object or null: {key}")
        if config is not None and any(name == "simt" or name.startswith("simt_") for name in config):
            raise ValueError("SIMT configurations are experimental and unavailable on consolidated")
        if config is not None and "split_k" in config:
            raise ValueError("Split-K configurations are unsupported")
        if config is not None and any(field in config for field in ("pad_gemm", "half_workspace")):
            raise ValueError("Padding/workspace routes were removed; use native output-store settings")
        if config is not None and "masked_global_store" in config and type(config["masked_global_store"]) is not bool:
            raise ValueError("masked_global_store must be boolean")
        routes[key] = config

    def policy(name):
        selected = frozenset(_signature(row) for row in document[name])
        if any(routes.get(key) is None for key in selected):
            raise ValueError(f"{name} must reference configured GEMM routes")
        return selected

    return routes, policy("direct_layout_gemms"), policy("materialize_gemms")


GRAPH_SCHEDULES, DIRECT_LAYOUT_GEMMS, MATERIALIZE_GEMMS = _load_router(ROUTER_PATH)
