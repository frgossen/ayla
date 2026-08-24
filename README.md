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

The benchmark prints a comparison table (median/mean latency, TFLOP/s, % of peak, and
throughput as a percentage of `torch.mm`) for a bf16 4096×4096 GEMM.

### Using pixi

If you don't already have pixi installed, get it with curl:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then set up the environment with [pixi](https://pixi.sh): `pixi shell` to activate the env, `pixi run install` provisions Python 3.12 with ayla (editable) plus the CUDA 12.8 (cu128) PyTorch build, then `pixi run test` runs the tests and `pixi run bench` runs the benchmark.

## Models to try

Candidate test subjects, once ayla is ready to serve real GEMM shapes:

- **Muse Glimmer 30B** — the best first target. A Meta model, open weights (Apache 2.0),
  and it serves on a single H100 at TP=1 in bf16, which is exactly what ayla emits. Its
  weight dimensions are all multiples of 256, so the existing tiling divides evenly.
- **Gemma 4 31B** and **Qwen 3.5/3.6 27B** — the two models Glimmer is benchmarked
  against, so they are the natural second and third subjects. Dense, single GPU, fp8.
- **DeepSeek V4**, **Kimi K2.7**, **GLM 5.2** — the frontier, and the furthest away.
  All mixture of experts, mostly fp8 with block scaling, so they need grouped GEMM and a
  narrower operand type before ayla has anything to say about them.
