"""Benchmarks, run as modules from the repository root.

A package so the scripts can import each other and `common` by name:

    python -m benchmarks.bench_gemm_collection
    python -m benchmarks.validate_oss_pytorch_gemm_shapes
    python -m benchmarks.benchmark_oss_pytorch_gemm_shapes
    python -m benchmarks.merge_oss_pytorch_gemm_results <shard JSONs...>
    python -m benchmarks.report_oss_pytorch_gemm_heuristics
    python -m benchmarks.bench_torch_compile.bench_torch_compile

Not shipped -- `pyproject.toml` packages `ayla*` only.
"""
