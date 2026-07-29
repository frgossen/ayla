# AYLA

A highly experimental compiler toolkit for generating **PTX code for state-of-the-art GEMM kernels
on NVIDIA GPUs**. Kernels are built from a set of correct, well-understood IR transforms — tiling,
async scheduling, and lowering to NVIDIA hardware primitives — using **deep pipelining without warp
specialization**. The aim is a compact toolbox that agents can use to generate SOTA GEMM kernels,
and aspirationally to fuse GEMM epilogues.

## Getting started

```bash
git submodule update --init                 # fetch the fast.cu reference submodule
pip install -e ".[dev]"                     # install (editable) + pytest
pytest                                      # run the tests
python benchmarks/bench_gemm_collection.py  # benchmark
```

The benchmark prints a comparison table (median/mean latency, TFLOP/s, % of peak, and the
relative throughput difference versus `torch.mm`) for a bf16 4096×4096 GEMM.

### Using pixi

If you don't already have pixi installed, get it with curl:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi shell
```

Then set up the environment with [pixi](https://pixi.sh): `pixi run install` provisions Python 3.12 with ayla (editable) plus the CUDA 12.8 (cu128) PyTorch build, then `pixi run test` runs the tests and `pixi run bench` runs the benchmark.
