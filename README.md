# AYLA

A highly experimental compiler toolkit for generating **PTX code for state-of-the-art GEMM kernels
on NVIDIA GPUs**. Kernels are built from a set of correct, well-understood IR transforms — tiling,
async scheduling, and lowering to NVIDIA hardware primitives — using **deep pipelining without warp
specialization**. The aim is a compact toolbox that agents can use to generate SOTA GEMM kernels,
and aspirationally to fuse GEMM epilogues.

## Build

```bash
pip install -e ".[dev]"     # editable install + pytest
```

## Test

```bash
pytest
```

## Benchmark

```bash
python benchmarks/bench_gemm_collection.py
```

Prints the generated IR and a comparison table (median/mean latency, TFLOP/s, and % of peak)
of the ayla kernels against `torch.mm` for a bf16 4096×4096 GEMM.
