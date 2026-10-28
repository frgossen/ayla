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

### JIT GEMMs and OSS benchmarks

The JIT supports contiguous FP16/BF16 `mm`, `addmm`, and `bmm`, and routes
supported calls through `ayla.torch_compile.install()`.

```python
from ayla.gemm_jit import gemm_op, measured_gemm_configs, select_gemm_schedule

schedule = select_gemm_schedule("mm", "bfloat16", m=8192, n=768, k=768)
print(schedule.builder_config if schedule else None)
print(measured_gemm_configs("mm", "bfloat16", m=8192, n=768, k=768))
# gemm_op("mm", "bfloat16")(a, bt, out) writes out = a @ bt.T.
# ADDMM takes (bias, a, bt, out); BMM takes batched (a, bt, out).
```

`select_gemm_schedule` selects a main-supported configuration, using a nearby
measured seed or an unmeasured fallback. `measured_gemm_configs` returns the
exact-shape saved configurations, winner first, including feature-only records.
For the **full feature space, performance search, and feature-enabled numbers**,
check out `feature/consolidated`. Benchmark commands on this main-based checkout
print checkout instructions and exit; the heuristic requires the full feature set.

Validation works on this checkout. Switch branches before benchmarking:

```bash
pixi run bench-oss-validate
git fetch origin
git switch feature/consolidated
CUDA_VISIBLE_DEVICES=0 pixi run bench-oss --workload all --max-shapes 3 \
  --output-json .benchmarks/oss-smoke/raw.json \
  --output-csv .benchmarks/oss-smoke/raw.csv
```

Remove `--max-shapes` for the full accepted corpus. Results are saved after
each shape; repeating the command resumes successful shapes. `--no-resume`
starts fresh. Inference timing is the default; `--timing warm` measures repeated
calls on the same inputs. Keep outputs from different branches separate.
Use `--corpus benchmarks/better_benchmark_gemm_shapes.json` for its recorded GEMMs.
The full model graph suite is described below.

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
