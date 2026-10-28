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

The OSS GEMM seed tool maps an FP16 shape to 2–3 measured neighboring schedules and
benchmarks them against `torch.mm`:

```bash
pixi run bench-oss-seed -- 8192 768 768
pixi run bench-oss-seed -- 8192 768 768 --show-only
pixi run bench-oss-seed -- 8192 768 768 --csv results.csv  # optional output
python -m benchmarks.convert_oss_gemm_heuristic expand ayla/oss_gemm_heuristic.json /tmp/heuristic.json
python -m benchmarks.convert_oss_gemm_heuristic collapse /tmp/heuristic.json ayla/oss_gemm_heuristic.json
python -m benchmarks.convert_oss_gemm_heuristic update ayla/oss_gemm_heuristic.json /tmp/updated.json run-*.json
python -m benchmarks.convert_oss_gemm_heuristic merge-mapper ayla/oss_gemm_heuristic.json /tmp/updated.json bmm-mapper.json
```

The converter provides lossless editing and merges any number of result files.
The benchmark generates contiguous `A`/`BT` inputs; it does not replay captured
tensor values or source strides.

Library callers can select the independent fused-addmm schedule groups with
`select_gemm_schedules(m, n, k, operation="addmm")`; omitting `operation`
preserves the existing `mm` routing.
Canonical batched workloads use
`select_gemm_schedules(m, n, k, operation="bmm", batch=batch)`.

The recorded OSS model-shape study is split into restartable phases:

```bash
pixi run bench-oss-validate  # CPU-only supported/skipped report
pixi run bench-oss           # GPU measurements; writes after every shape
pixi run bench-oss-report    # regime-level Markdown from the raw JSON
pixi run bench-oss-map       # collapse per-shape winners into a small mapper
```

Focused follow-up searches can use `bench-oss --tune-only --worst-shapes N` or
`bench-oss --cublas-inspired-only --worst-shapes N`.  `--transfer-only`
reuses the full knob settings of successful neighboring shapes while adapting
static CTA counts to the target geometry. Each configuration runs in an
isolated worker with a 120-second watchdog by default, so pathological
CPU-side compilation is recorded and skipped instead of stalling a shard;
`python -m benchmarks.profile_oss_pytorch_gemm --help` provides a one-launch
CUDA profiler target for comparing a measured Ayla configuration with
`torch.mm`.
Repeated focused passes can be merged with `bench-oss-merge --prefer-last`
when a later run intentionally remeasures the same named configuration.

These write validation CSV/JSON, raw CSV/JSON, per-regime and per-shape
improvement tables, and machine-readable recommended seeds under
`.benchmarks/oss_gemm/`; the generated
heuristic study is kept at `benchmarks/oss_pytorch_gemm_heuristics.md`. The
benchmark runner's `--help` lists smoke-run filters and output overrides.
Recorded operand orders are retained in validation output but pooled into
canonical contiguous A/BT workloads for benchmarking.

The current corpus funnel is: 1,765 source GEMM records → 1,729 pooled
signatures → 855 unbatched `mm` signatures → 487 signatures after dtype,
alignment, small-M, and split-K filtering → 406 canonical M/N/K workloads.

The generated mapper is an interpretable decision tree with 2–3 measured seed
configurations per leaf. Querying it prints the complete branch trace, source
shapes, and any exact-CTA adaptation, for example:

```bash
pixi run python -m benchmarks.generate_oss_gemm_mapper --query 8192 768 768
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
