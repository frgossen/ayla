"""Exact-shape Blackwell routes, including FP32 workspace split-K."""

import json
from pathlib import Path
from ..pipelines.ring_schedule import LoopSchedule, PipelineSchedule
from .common_warp_schedule import validate_common_warp_schedule



LEGACY_PIPELINE_FIELDS = {
    "persistent_output_pipeline", "persistent_output_overlap",
    "persistent_output_frames", "persistent_output_job_regions",
    "rolled_operand_pipeline", "rolled_k_unroll", "rolled_job_unroll",
    "peel_first_output_pair", "cyclic_operand_pipeline", "output_writeback_window",
    "batch_output_m_operands",
}


def normalize_blackwell_configuration(configuration, *, default_pending=0):
    """Adapt old library callers to the normalized public schedule contract."""
    config = dict(configuration)
    if "native_schedule" in config:
        config["native_schedule"] = normalize_blackwell_configuration(config["native_schedule"])
    old = {name: config.pop(name) for name in LEGACY_PIPELINE_FIELDS if name in config}
    if old:
        if "pipeline" in config:
            raise ValueError("select pipeline or legacy schedule fields, not both")
        for name, value in old.items():
            kind = int if name in ("rolled_k_unroll", "rolled_job_unroll", "persistent_output_frames") else bool
            if name in ("output_writeback_window", "cyclic_operand_pipeline"):
                continue
            if type(value) is not kind:
                raise ValueError(f"{name} must be a positive integer" if "unroll" in name else f"{name} must be {kind.__name__}")
        cross = old.get("persistent_output_pipeline", False)
        pending = old.get("cyclic_operand_pipeline", default_pending)
        if type(pending) not in (bool, int):
            raise ValueError("cyclic_operand_pipeline must be an integer or boolean")
        plan = PipelineSchedule(
            operand_scope="output_jobs" if cross else "reduction",
            accumulator_slots=old.get("persistent_output_frames", 2) if cross else 1,
            output_overlap=old.get("persistent_output_overlap", True) if cross else False,
            participants="round_robin" if old.get("persistent_output_job_regions", False) else "shared",
            reduction=LoopSchedule("rolled" if old.get("rolled_operand_pipeline", False) else "fixed",
                                   old.get("rolled_k_unroll", 1)),
            output_jobs=LoopSchedule("rolled", old.get("rolled_job_unroll", 1),
                                     2 if old.get("peel_first_output_pair", False) else 0),
            mma_pending=int(pending), writeback_window=old.get("output_writeback_window"),
        )
        if old.get("batch_output_m_operands", False):
            if config.get("mma_m_tile") is not None:
                raise ValueError("select CTA/MMA tile geometry or legacy M batching")
            config["mma_m_tile"] = config.get("output_m_tile", 128)
            config["output_m_tile"] = config["mma_m_tile"] * config.pop("output_m_tiles_per_cta", 1)
    else:
        plan = PipelineSchedule.parse(config.pop("pipeline", None), default_pending=default_pending)
    compact = plan.configuration(default_pending=default_pending)
    if compact or default_pending:
        config["pipeline"] = compact
    return config

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
            make_ptx_kernel(partial),
            None, m=m, n=n, k=k, dtype=dtype
        )
    if "stream_k_workers" in config:
        from .stream_k import make_blackwell_stream_k_launcher

        if (
            config.get("split_k", 1) != 1
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
        make_ptx_kernel(partial),
        make_ptx_kernel(reduction) if reduction is not None else None,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        transpose_output=transpose,
        transpose_reduction=transpose_reduction,
    )
