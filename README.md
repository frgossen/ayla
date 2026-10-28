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

### Which branch to benchmark

The JIT heuristic and benchmark tools let users of `main` query the best known
configuration for a supported shape and run the OSS and better-benchmark suites.
For performance measurements using all of the current compiler features,
**use `feature/consolidated`**:

```bash
git fetch origin
git switch feature/consolidated
```

Run the benchmark commands below from that checkout. The consolidated branch
includes the newer scheduling, multicast, SwiGLU, and staging features. Some
configurations on `main` will be suboptimal without those features; a measured
winner is not a promise of the same performance on every branch or GPU.
Record the branch, commit (`git rev-parse HEAD`), and GPU with your results so
comparisons identify the implementation that was actually measured.

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

`ayla.gemm_jit` compiles FP16/BF16 `mm`, `addmm`, and `bmm` for supported shapes from
the measured winners in `ayla/gemm_schedule_table.py`: a measured shape gets its
own winner, any other shape the nearest measured shape's first schedule that is
legal for it. `gemm_op(operation, dtype)` is the torch op
(`op([bias,] a, bt, out)` writes `out = [bias +] a @ bt.T`), and
`ayla.torch_compile.install()` routes native-layout `aten.mm`/`addmm`/`bmm` that
have a schedule through it.

To inspect the selected configuration without compiling or launching a kernel:

```bash
pixi run python - <<'PY'
from pprint import pprint
from ayla.gemm_jit import select_gemm_schedule

seed = select_gemm_schedule("mm", "bfloat16", m=8192, n=768, k=768)
if seed is None:
    print("No legal measured schedule for this shape.")
else:
    print("Measured configuration:")
    pprint(seed.captured_config)
    print("Builder configuration for this shape:")
    pprint(seed.builder_config)
    print("Adaptations:", seed.adaptations)
PY
```

Use `"float16"` for FP16, `"addmm"` for bias initialization, or `"bmm"`
with `batch=...` for batched GEMM. An exact measured shape uses its recorded
winner; other shapes use a legal configuration transferred from a neighboring
measured shape. Finding the fastest configuration for a new shape still
requires benchmarking.

`bench-oss --jit-heuristic` times exactly that routing: for every corpus shape
it runs the JIT op against Torch, with the usual correctness check, isolated
workers, and resumable raw rows (configuration `jit_heuristic`):

```bash
pixi run bench-oss --jit-heuristic --workload all --max-shapes 3 --output-json .benchmarks/jit/smoke.json --output-csv .benchmarks/jit/smoke.csv
pixi run bench-oss --jit-heuristic --workload all --output-json .benchmarks/jit/oss.json --output-csv .benchmarks/jit/oss.csv
```

The first command is a short smoke run; the second runs the supported corpus.
Use `CUDA_VISIBLE_DEVICES=0` before either command to select one GPU. Completed
rows are saved after each shape and reused on subsequent runs; use `--no-resume`
to replace results when changing branches or configurations.

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

Use `--gpus 0` for a single GPU. `--models-root` identifies the captured graph
directory inside the better-benchmark checkout; it does not download model
weights. The BF16 corpus command above benchmarks individual GEMMs, while this
wrapper compares full graphs with stock Inductor and Ayla enabled. For numbers
that use the complete feature set, run both from `feature/consolidated`.

### Using pixi

If you don't already have pixi installed, get it with curl:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then set up the environment with [pixi](https://pixi.sh): `pixi shell` to activate the env, `pixi run install` provisions Python 3.12 with ayla (editable) plus the CUDA 12.8 (cu128) PyTorch build, then `pixi run test` runs the tests and `pixi run bench` runs the benchmark.
