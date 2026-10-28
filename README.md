# AYLA

A highly experimental compiler toolkit for generating **PTX code for state-of-the-art GEMM kernels
on NVIDIA GPUs**. Kernels are built from a set of correct, well-understood IR transforms — tiling,
async scheduling, and lowering to NVIDIA hardware primitives — using **deep pipelining without warp
specialization**. The aim is a compact toolbox that agents can use to generate SOTA GEMM kernels,
and aspirationally to fuse GEMM epilogues.

## Branches

- `feature/consolidated`: single-GEMM performance work, measured MM/ADDMM/BMM
  schedules, OSS search, and better-benchmark tooling. Use this branch for
  benchmark numbers with the full performance feature set.
- `feature/gemm-chains`: development branch based on consolidated, adding
  generic FX graph capture, workspaces, SwiGLU, and GEMM-to-GEMM kernels.

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

### OSS search

Validate the recorded model shapes on CPU, then benchmark or search on a GPU:

```bash
pixi run bench-oss-validate
pixi run bench-oss --jit-heuristic --workload all
pixi run bench-oss --tune-only --worst-shapes 10
```

The runner checks numerical results, isolates configurations with a watchdog,
and saves resumable CSV/JSON under `.benchmarks/oss_gemm/`. `--help` lists
shape filters, search controls, timing protocols, and output paths.

There is one GEMM heuristic: `ayla.gemm_jit.select_gemm_schedule`, backed by
`ayla/gemm_schedule_table.py`. Measured shapes use their ranked schedules;
unmeasured shapes borrow a legal schedule from the nearest measured shape.
Search results can add schedule families to this same table:

```bash
pixi run python -m benchmarks.select_gemm_schedule_families --update-table .benchmarks/better_benchmark_gemm/*/gpu*.json
```

### JIT-routed GEMMs

`ayla.gemm_jit` compiles FP16/BF16 `mm`, `addmm`, and `bmm` for any shape from
the measured winners in `ayla/gemm_schedule_table.py`: a measured shape gets its
own winner, any other shape the nearest measured shape's first schedule that is
legal for it. `gemm_op(operation, dtype)` is the torch op
(`op([bias,] a, bt, out)` writes `out = [bias +] a @ bt.T`), and
`ayla.torch_compile.install()` routes native-layout `aten.mm`/`addmm`/`bmm` that
have a schedule through it.

`bench-oss --jit-heuristic` times exactly that routing: for every corpus shape
it runs the JIT op against Torch, with the usual correctness check, isolated
workers, and resumable raw rows (configuration `jit_heuristic`):

```bash
pixi run bench-oss --jit-heuristic --workload all --output-json .benchmarks/jit/oss.json --output-csv .benchmarks/jit/oss.csv
```

`--corpus` takes any corpus in the OSS format.
`benchmarks/better_benchmark_gemm_shapes.json` holds better-benchmark's BF16
GEMM shapes; give that run its own validation reports so it leaves the OSS ones
alone:

```bash
pixi run bench-oss --jit-heuristic --workload all --corpus benchmarks/better_benchmark_gemm_shapes.json --validation-csv .benchmarks/jit/bb_validation.csv --validation-json .benchmarks/jit/bb_validation.json --output-json .benchmarks/jit/bb.json --output-csv .benchmarks/jit/bb.csv
```

To split a run across GPUs, start one shard per GPU with
`--num-shards`, `--shard-index`, and its own output files, and pass
`--skip-validation-write` so the shards do not rewrite the shared validation
report, for example:

```bash
CUDA_VISIBLE_DEVICES=1 pixi run bench-oss --jit-heuristic --workload all --skip-validation-write --num-shards 8 --shard-index 1 --output-json .benchmarks/jit/oss_1.json --output-csv .benchmarks/jit/oss_1.csv
```

`benchmarks/bench_better_benchmark.py` times better-benchmark's full graphs
under stock Inductor and with `ayla.torch_compile.install()`, and prints the
per-graph delta. It runs the `third_party/better-benchmark` submodule:

```bash
git submodule update --init third_party/better-benchmark
pixi run python benchmarks/bench_better_benchmark.py --models-root repros/models/hf/infer --gpus 0,1,2,3,4,5,6,7
```

### Using pixi

If you don't already have pixi installed, get it with curl:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then set up the environment with [pixi](https://pixi.sh): `pixi shell` to activate the env, `pixi run install` provisions Python 3.12 with ayla (editable) plus the CUDA 12.8 (cu128) PyTorch build, then `pixi run test` runs the tests and `pixi run bench` runs the benchmark.

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
