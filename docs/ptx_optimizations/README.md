# Measured PTX lowering and scheduling changes

Two changes ship unconditionally. They were measured as "P3+P4" while they
were opt-in feature flags:

- **TMA lowering (P3).** TMA load/store leader election reuses the kernel's
  warp-uniform `WarpIdsOp` IDs. Blocked coordinates with power-of-two block
  widths use shift/mask instead of `div`/`rem`
  (`ayla/passes/lower_to_ptx_template.py`).
- **Retained WGMMA groups (P4).** `ScheduleAsync` sinks a full WGMMA await
  below the next group when they only share same-shape accumulators. The await
  uses the existing `pending: 1` attribute. Sync analysis treats every await as
  completing its listed lock (`ayla/passes/schedule_async.py`,
  `ayla/analysis/sync.py`). The ordering argument is in [ordering.md](ordering.md).

There is no `ptx_features` attribute or `make_ptx_kernel` parameter. The opt-in
experiments P1, P2 and P5–P9 were removed: P1 and P2 regress long reductions
(down to 0.830×), P6 regresses (0.956×), P5 and P9 leave SASS unchanged, and P7
and P8 were mostly sub-percent.

## Refactor equivalence

The flag version and the refactor were compared on all 3,108 buildable exact
schedule-table and graph routes:

- 842 have identical PTX and 1,587 differ only in where the retained
  `wait_group 1` sits. `ScheduleAsync` sinks it past independent setup, such as
  the next mbarrier wait.
- The rest retain more groups. Retention now happens wherever `ScheduleAsync`
  runs, so tails of unrolled output tiles and multi-warpgroup CTAs (which the
  flag version excluded) can retain. Single-warpgroup kernels can also drop a
  block gather after `PipelineLoop`'s loop-top partial wait.
- Paired timing, H100 at 1500 MHz, 31×200-launch graphs, reference vs refactor
  on a stratified sample of 144 differing kernels: geomean **1.0002**, all
  outputs bitwise identical. Kernel range 0.973–1.024 on re-measurement; the
  two below 0.99 are multi-warpgroup.
- The eight headline shapes below, both dtypes, 41×400 launches: geomean
  **1.001** vs the flag version, bitwise identical.

## Fixed-schedule results

Measured on NVIDIA H100 using CUDA 13.0.88 PTXAS and Torch 2.13.0+cu130. The eight
FP16 shapes are the exact OSS subset used by the SASS comparison document.
BF16 uses the same dimensions with the library's selected BF16 schedules.
Every comparison holds its selected seed, layouts, stages, launch geometry and
PTX optimization level fixed. Levels differ between selected schedules; no
optimization-level sweep or schedule search was performed.

Each median uses **41 alternating graph measurements, 400 launches per graph**.
Every variant is checked against Torch after three launches, resetting output
to NaN each time. All measured outputs were **bitwise identical to their paired
PTX baseline**, including ADDMM. Sub-percent timing differences are inconclusive.

### TMA and WGMMA-queue changes (P3+P4)

Speedup is the median before these changes divided by the median after them.

| OSS operation M×N×K | FP16 baseline µs | FP16 P3+P4 µs | FP16 speedup | BF16 speedup |
| --- | ---: | ---: | ---: | ---: |
| MM 128×128×64 | 1.496 | 1.448 | 1.034× | 1.032× |
| ADDMM 64×1000×128 | 1.783 | 1.716 | 1.039× | 1.037× |
| BMM 64×64×448 (batch 12) | 2.612 | 2.401 | 1.088× | 1.118× |
| MM 512×768×768 | 3.374 | 3.190 | 1.058× | 1.036× |
| ADDMM 512×768×768 | 3.327 | 2.982 | 1.116× | 1.145× |
| MM 512×768×3072 | 8.418 | 8.048 | 1.046× | 1.106× |
| ADDMM 512×768×3072 | 7.733 | 7.330 | 1.055× | 1.075× |
| MM 768×768×8192 | 20.967 | 20.227 | 1.037× | 1.032× |

Geometric-mean speedup: **1.059× FP16 / 1.072× BF16** (5.9% / 7.2%). Medium
ADDMM reaches **11.6% FP16 / 14.5% BF16**; BF16 time falls from 4.300 to 3.754 µs.
These are results for this selected subset, not an OSS-wide score.

## Validation

- `tests/passes/test_schedule_async.py`: retention and its same-shape
  requirement. `tests/passes/test_insert_syncs.py`: retained-group operand
  reuse needs no gather on a single-warpgroup CTA.
- `tests/test_gemm.py -k retained`: FP16/BF16 MM/ADDMM, masked row/column/K
  tails, a three-stage ring, transposed output, multi-warpgroup CTAs and both
  multicast cluster axes, repeated launches. Compute Sanitizer memcheck: 0
  errors; racecheck: 0 hazards.
