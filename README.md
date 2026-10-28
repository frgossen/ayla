# AYLA

A highly experimental compiler toolkit for generating **PTX code for state-of-the-art GEMM kernels
on NVIDIA GPUs**. Kernels are built from a set of correct, well-understood IR transforms — tiling,
async scheduling, and lowering to NVIDIA hardware primitives — using **deep pipelining without warp
specialization**. The aim is a compact toolbox that agents can use to generate SOTA GEMM kernels,
and aspirationally to fuse GEMM epilogues.

## Benchmark branch choice

This checkout provides measured GEMM configuration lookup, shape-specialized
FP16/BF16 JIT execution for `mm`, `addmm`, and `bmm`, and routing through
`ayla.torch_compile.install()`. It uses the compiler and scheduling features
available on main. JIT selection skips configurations that require newer
features and uses a conservative unmeasured fallback when needed.

For the **full feature space, performance search, and the best numbers**, use
the consolidated branch:

```bash
git fetch origin
git switch feature/consolidated
```

The saved measurements include schedules using multicast, persistent grids,
cyclic pipelines, and other features from that branch. Main's supported
configurations can be suboptimal; rebenchmark them on the checkout you use.
Configuration recommendations are tied to the compiler revision and hardware
that produced the measurements.

## Getting started

```bash
git submodule update --init                 # fetch the fast.cu reference submodule
pip install -e ".[dev]"                     # install (editable) + pytest
git config core.hooksPath .githooks         # run checks before each push
pytest                                      # run the tests
python -m benchmarks.bench_gemm_collection  # benchmark
```

The benchmark prints a comparison table (median/mean latency, TFLOP/s, % of peak, and
throughput as a percentage of `torch.mm`) for a bf16 4096×4096 GEMM.

### Using pixi

If you don't already have pixi installed, get it with curl:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then set up the environment with [pixi](https://pixi.sh): `pixi shell` to activate the env, `pixi run install` provisions Python 3.12 with ayla (editable) plus the CUDA 12.8 (cu128) PyTorch build, then `pixi run test` runs the tests and `pixi run bench` runs the benchmark.

### GEMM configuration lookup and JIT execution

Select a schedule executable on this checkout without compiling a kernel:

```python
from ayla.gemm_jit import measured_gemm_configs, select_gemm_schedule

schedule = select_gemm_schedule("mm", "bfloat16", m=8192, n=768, k=768)
if schedule is not None:
    print(schedule.captured_config)
    print(schedule.builder_config)
    print(schedule.adaptations)  # transferred or unmeasured fallback, if applicable

# Exact-shape saved configurations, winner first. These may require the
# consolidated branch; this query preserves the original measured records.
print(measured_gemm_configs("mm", "bfloat16", m=8192, n=768, k=768))
```

For direct execution on CUDA, `gemm_op("mm", "bfloat16")(a, bt, out)` writes
`a @ bt.T`; `gemm_op("addmm", "bfloat16")(bias, a, bt, out)` adds a length-N
bias, and `gemm_op("bmm", "bfloat16")(a, bt, out)` accepts batched matrices.
All tensors must be contiguous, have the selected dtype, and share one CUDA
device. `"float16"` is also supported. Invoke the operation once before CUDA
graph capture to populate the specialization cache. Unsupported layouts,
dtypes, or shapes fall through to Inductor when using the torch.compile backend.

### OSS GEMM benchmarks

Validate the corpus without reserving a GPU:

```bash
pixi run bench-oss-validate
```

Run a small benchmark through the library's JIT dispatcher:

```bash
CUDA_VISIBLE_DEVICES=0 pixi run bench-oss --jit-heuristic --workload all \
  --max-shapes 3 --max-timing-rounds 3 \
  --output-json .benchmarks/oss-smoke/raw.json \
  --output-csv .benchmarks/oss-smoke/raw.csv
```

For the full accepted corpus, remove `--max-shapes` and use a separate output
path. Results are written after each shape and the same command resumes completed
rows. `--no-resume` starts fresh. The default inference timing uses hot
activations and cold per-layer weights; `--timing warm` measures repeated calls
on the same tensors. Correctness is checked against Torch before timing.

Without `--jit-heuristic`, the benchmark evaluates candidate configurations
supported by this checkout. Feature-only candidates are skipped. Use
`feature/consolidated` for the full search and feature-enabled winners.
Keep results from different branches in separate output directories.

### better-benchmark

[better-benchmark](third_party/better-benchmark)'s full model graphs measure ayla
inside `torch.compile`: every graph runs once with stock Inductor and once with
`ayla.torch_compile.install()`, each with its own Inductor cache.

```bash
git submodule update --init third_party/better-benchmark
python benchmarks/bench_better_benchmark.py --gpus 0,1,2,3,4,5,6,7 --output-dir .benchmarks/bb
```

It prints each graph's time in both arms and the geomean speedup, comparing
graphs like better-benchmark's `bench_parallel.py --compare` (coordinate-descent
time, else the default compiled time). `--models-root repros/models/hf/infer`
narrows the run, `--output-dir` keeps each arm's raw JSON, and `--bench-root`
points at another better-benchmark checkout (for example from a git worktree).
Workers import this checkout's ayla, even when another one is installed.
