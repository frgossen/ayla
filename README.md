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

### Generate readable PTX

Run the PTX lifter as a package module from the repository root:

```bash
python -m ayla.ptx_python_codegen
```

This regenerates
`generated_ptx/auto_matmul_k3_4tiles_readable_ptx.py`. To convert an existing
raw PTX file instead, pass its path; the generated Python is written to stdout
unless `-o` is supplied:

```bash
python -m ayla.ptx_python_codegen kernel.ptx -o generated_ptx/kernel.py
```

The generated module retains every PTX directive and instruction. Calling its
`PTX_PROGRAM()` function expands the parameterized blocks and compact schedule
into complete PTX source; comments are omitted and blank lines are normalized.

### Using pixi

If you don't already have pixi installed, get it with curl:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then set up the environment with [pixi](https://pixi.sh): `pixi shell` to activate the env, `pixi run install` provisions Python 3.12 with ayla (editable) plus the CUDA 12.8 (cu128) PyTorch build, then `pixi run test` runs the tests and `pixi run bench` runs the benchmark.
