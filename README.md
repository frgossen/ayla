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
