# Main-backed JIT GEMM branch

Base: upstream main `d2f8fd128e3fbace873fa2d6aa13aa2c71db5fb9` (the locally fetched
main including PR #503). Branch: `jit-gemm-main`.

This is a fresh port from main. It does not merge `jit-gemm-deps`, the old
`jit-gemm-heuristics` history, or `feature/consolidated`.

## Scope

- Preserve the measured table and compact heuristic data. Exact saved configs
  can be queried with `measured_gemm_configs`; recommendations using feature-only
  knobs are identified by the compact heuristic API.
- `select_gemm_schedule` chooses an executable main-compatible measured seed,
  transferring from a nearby shape if necessary. It uses an explicitly
  unmeasured conservative fallback when no saved seed can transfer.
- Direct and torch.compile JIT execution supports native contiguous FP16/BF16
  MM, ADDMM with vector bias, and BMM. Unsupported Inductor inputs are declined.
- Keep main's existing `schedule_matmul` and scheduling passes. The required
  compiler/launcher port is limited to walking singleton batch reshapes for TMA,
  emitting batch TMA coordinates, and encoding/caching rank-N descriptors.
- OSS benchmarking retains validation, resumption, correctness checks, inference,
  cold and warm timing, and candidate selection. Candidate execution filters
  unsupported features. CPU workers generate PTX; main's CUDA driver path assembles
  and loads it. Benchmark outputs record the compiler feature space and reject
  resuming outputs from a different feature space.
- Retain main's improved better-benchmark wrapper. Extend Inductor routing to
  the shape-specialized JIT operations without changing the existing sigmoid
  fusion/prototype IR path.

Use `feature/consolidated` for multicast, persistent grids, cyclic pipelines,
advanced output traversal, the complete search space, and feature-enabled numbers.
Keep results from different checkouts separate. The saved table describes previous
measurements, not a guarantee of the same performance on main.

## Verification

Environment: Python 3.12, PyTorch 2.13/CUDA 13.0, NVIDIA H100. GPU runs used
`TORCH_NO_COMPILER_WRAPPER=1 CXX=/usr/bin/g++` to bypass this machine's broken
Torch compiler wrapper.

- Full GPU suite: **620 passed**, 18 Torch warnings.
- Ruff: passed. Pyright: zero errors, warnings, or information diagnostics.
- New JIT tests: 24 passed, including FP16/BF16 MM/ADDMM/BMM, partial M tiles,
  CUDA graph replay, and Inductor routing for all six operation/dtype combinations.
- Selection over all 749 accepted canonical OSS corpus shapes: none declined.
  This is a selection check, not numerical or performance validation of every shape.
- OSS inference smoke: three MM shapes plus one ADDMM and one BMM; all numerical
  checks passed. Three timing samples per shape are smoke checks, not stable
  performance comparisons. MM samples were 71–79% of Torch throughput; the ADDMM
  and BMM samples were approximately 108–109%. These are not broad speedup claims.
- OSS candidate benchmark with one CPU PTX worker and an isolated GPU worker:
  completed with both Torch and Ayla rows successful.
- better-benchmark wrapper help and its routing paths were verified; the full
  external model suite was not run.

Raw benchmark artifacts are stored locally under the repository's
`.git/pr-triage/jit-main-*.json` and `.csv`.

## Publishing

```bash
git -C /home/arseniivanov/ayla-jit-main push -u origin jit-gemm-main
```

Open a **new PR with base `main` and head `jit-gemm-main`**. PR #492 was already
merged into `jit-gemm-deps`; it cannot be retargeted after merging. The old deps
branch can remain unused. This port does not require reverting or merging it.
