"""Exact-shape Blackwell routes, including FP32 workspace split-K."""

import json
from pathlib import Path
from .common_warp_schedule import validate_common_warp_schedule


def select_blackwell_router_schedule(m, n, k, dtype="f16"):
    """Return an independent route configuration, or None if unmeasured."""
    table_dtype = {"f16": "float16", "bf16": "bfloat16"}.get(dtype, dtype)
    path = (
        Path(__file__).resolve().parents[1]
        / "better_benchmark_gemm_router_blackwell.json"
    )
    for route in json.loads(path.read_text())["routes"]:
        if (
            route.get("op", "mm"),
            route.get("batch", 1),
            route["dtype"],
            route["m"],
            route["n"],
            route["k"],
        ) == ("mm", 1, table_dtype, m, n, k):
            validate_common_warp_schedule(route["configuration"])
            return dict(route["configuration"])
    return None


def make_blackwell_router_launcher(*, m, n, k, dtype="f16", configuration=None):
    """Build the exact route. Allocate ``workspace_shape`` before graph capture.

    Call ``launch(a, bt, out, workspace)``. Direct and cluster Split-K routes
    require no workspace. Workspace Split-K captures producer and finalizer.
    Concurrent executions require separate FP32 workspaces.
    """
    from .blackwell_tcgen05 import blackwell_tcgen05_gemm_module
    from .split_k import BlackwellSplitKLauncher, blackwell_long_k_gemm_modules
    from ..ptx_codegen import make_ptx_kernel

    dtype = {"float16": "f16", "bfloat16": "bf16"}.get(dtype, dtype)
    config = (
        select_blackwell_router_schedule(m, n, k, dtype)
        if configuration is None
        else dict(configuration)
    )
    if config is None:
        raise ValueError(f"no measured Blackwell route for {m}x{n}x{k} {dtype}")
    validate_common_warp_schedule(config)
    prefetch_tmem_reads = config.pop("prefetch_tmem_reads", False)
    if type(prefetch_tmem_reads) is not bool:
        raise ValueError("prefetch_tmem_reads must be a boolean")
    backend = config.pop("split_k_backend", "workspace")
    if backend not in ("cluster", "workspace"):
        raise ValueError("Split-K backend must be cluster or workspace")
    if backend == "cluster":
        from .split_k import blackwell_split_k_gemm_module

        if (
            config.get("split_k", 1) <= 1
            or config.get("transpose_output")
            or config.get("transpose_reduction")
        ):
            raise ValueError(
                "cluster Split-K requires multiple untransposed K partitions"
            )
        partial = blackwell_split_k_gemm_module(m=m, n=n, k=k, dtype=dtype, **config)
        return BlackwellSplitKLauncher(
            make_ptx_kernel(partial, prefetch_tmem_reads=prefetch_tmem_reads),
            None, m=m, n=n, k=k, dtype=dtype
        )
    if "stream_k_workers" in config:
        from .stream_k import make_blackwell_stream_k_launcher

        if (
            prefetch_tmem_reads
            or config.get("split_k", 1) != 1
            or config.get("transpose_output")
            or config.get("transpose_reduction")
            or "native_schedule" in config
        ):
            raise ValueError("Stream-K requires an untransposed, non-Split-K schedule")
        workers = config.pop("stream_k_workers")
        for key in ("split_k", "transpose_output", "transpose_reduction"):
            config.pop(key, None)
        return make_blackwell_stream_k_launcher(
            m=m, n=n, k=k, dtype=dtype, workers=workers, **config
        )
    transpose = config.get("transpose_output", False)
    transpose_reduction = config.get("transpose_reduction", False)
    if config.get("split_k", 1) > 1 or "native_schedule" in config:
        partial, reduction = blackwell_long_k_gemm_modules(
            m=m, n=n, k=k, dtype=dtype, **config
        )
    else:
        config.pop("split_k", None)
        config.pop("transpose_output", None)
        config["output_column_major"] = transpose
        partial = blackwell_tcgen05_gemm_module(
            m=n if transpose else m, n=m if transpose else n, k=k, dtype=dtype, **config
        )
        reduction = None
    return BlackwellSplitKLauncher(
        make_ptx_kernel(partial, prefetch_tmem_reads=prefetch_tmem_reads),
        make_ptx_kernel(reduction) if reduction is not None else None,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        transpose_output=transpose,
        transpose_reduction=transpose_reduction,
    )
