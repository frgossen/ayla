# H100 GEMM performance-improvement summary

Date: 2026-08-17

This summary covers the BF16 `4096x4096x4096` GEMM work built on current
`main` (`181af1f`), the clean PR stack, the experimental stack ending at
`perf/output-cache-sass-ir` (`32f2d32`), the current uncommitted worktree, the
benchmark harnesses, and all performance notes and recorded result tables
under `docs/`.

Runtime deltas below are paired candidate-versus-baseline results unless noted.
“Faster” avoids the sign ambiguity in the source reports, where a negative
latency delta means an improvement. “Not isolated” means the repository has no
credible standalone A/B result for that individual change; no delta is inferred
from a multi-change commit.

## Branch and stack state

- Baseline: `main` / `origin/main` at `181af1f`.
- Clean PR stack: `45d9034` (blocked layouts), `5107a42` (explicit STMatrix),
  `origin/STSM-stores`, then `origin/pr58` through `origin/pr67`, ending at
  `origin/perf/four-stage-split-output` / `stack/rebuild` (`d317dae`).
- Experimental performance stack: `c24694c` through `32f2d32`, with useful
  checkpoints at local `STSM-stores` (`4a27b85`), local
  `perf/four-stage-split-output` (`e8ea520`), and current
  `perf/output-cache-sass-ir` (`32f2d32`).
- The local and remote `perf/four-stage-split-output` names do not point to the
  same history: local is `e8ea520`; remote is the clean rebuilt stack at
  `d317dae`.
- The worktree is dirty. Its additional retained-best settings—O1 Driver JIT,
  A-descriptor precomputation, exact `.reqntid`, direct warpgroup ID, and PTX
  special-register sharing—are documented below but are not part of HEAD.

## Improvements

| # | Improvement | Stack / implementation | Performance delta | What it does |
|---:|---|---|---|---|
| 1 | Blocked physical layouts and explicit STMatrix output | `45d9034`, `5107a42`, `origin/STSM-stores`; experimental `c24694c..4a27b85` | Not isolated. The resulting two-tile STMatrix baseline profiles at 245.28 us, 84.43% tensor/SM throughput, and 17.341 M executed instructions. | Models the physical shared layout and stages accumulator registers to shared memory with `stmatrix` before TMA output. This is the foundation for later swizzling and full-width stores. |
| 2 | Four output tiles per CTA with pipelined/split writeback | experimental `547174a`; clean `pr65..pr67`, `d317dae` | No clean standalone A/B was recorded. Composite two-tile-STMatrix to current-final NCU movement is 17.341 M to 13.992 M executed instructions (-19.3%) and 84.43% to 92.05% tensor active (+7.62 pp), but includes many later changes. | Amortizes CTA-fixed setup across four `128x256` outputs and changes the launch from 256 to 128 CTAs. |
| 3 | Reuse asymmetric named barriers | `origin/pr59` / compound experimental lowering | Not isolated in this stack. Replacing the retained asymmetric barriers with full-CTA barriers was 15.7% slower. | Reuses named barrier IDs for repeated handoffs; only warp 0 waits while other warps arrive and continue toward the next wait. |
| 4 | Fold accumulator zeroing into first WGMMA | `origin/pr60`; experimental `54b3ef0` | Removes 256 static PTX zero moves and about 524,288 executed warp instructions, approximately 3.0% of the earlier instruction count; runtime was not isolated. | Allocates the accumulator uninitialized and emits first-WGMMA `scale_d=0`; later WGMMAs use `scale_d=1`. |
| 5 | Pack WRMEM conversions with x2 instructions | `origin/pr61`; experimental compound commit | Not isolated; the lowering replaces pairs of scalar register-to-BF16 conversions with packed x2 forms. | Cuts conversion instruction count while preserving the output value order used by STMatrix. |
| 6 | Reuse the per-thread STMatrix offset | `41291f5` / clean canonical lowering | Static SASS 2,608 to 2,584 (-24, -0.92%); timing only measured as part of the neutral combined static-cleanup bundle. | Computes the lane/warp-to-output offset once and reuses it for all eight STMatrix groups. |
| 7 | Explicit lane-0 TMA election | `41291f5` | Static SASS 2,584 to 2,504 (-80, -3.10%); combined timing was neutral at the measurement floor. | Gives PTXAS an explicit one-lane proof and removes TMA serialization loops while retaining exact thread-0 semantics. |
| 8 | Timeout-free asynchronous waits | `origin/pr62`; `41291f5` | Static SASS 2,504 to 2,448 (-56, -2.24%); paired change -0.133% (neutral/noise). | Omits the optional `mbarrier.try_wait` time hint and its out-of-line nanosleep/phase-check slow paths while retaining polling. |
| 9 | Shared `%tid.x` SSA read | `41291f5` / clean canonical lowering | Static SASS 2,448 to 2,440 (-8, -0.33%); paired change -0.039% (neutral). | Reuses one stable thread-ID value instead of repeatedly reading/copying the special register. |
| 10 | In-place WGMMA descriptor recurrence | `origin/pr63`; `41291f5` | Static SASS 2,440 to 2,424 (-16, -0.66%); paired change -0.053% (neutral). | Advances the two descriptor bases by `+2` between four consecutive WGMMAs instead of creating six one-use descriptor names. |
| 11 | A/C evict-first, B default cache policy | `origin/pr64`; experimental `ae1def4`, `41291f5` | Adding C evict-first was 0.765-0.945% faster. After the full-store change, removing A or C evict-first was 0.887% and 0.909% slower; removing all hints was 1.863% slower. | Keeps reusable B at default priority while preferentially evicting streamed A and write-only C. |
| 12 | Dense row-major output swizzle | `2472079` | 0.0973% faster, 95% CI 0.0542-0.1288% faster; about 8 KiB less shared memory than padding. | Replaces padding with a dense 128-byte XOR swizzle after regrouping STMatrix fragments. |
| 13 | Conflict-free block-major output swizzle | `2472079` | A further 0.2688% faster than dense row-major, 95% CI 0.2331-0.3053%; STMatrix conflicts 262,144 to 0 and wavefronts 524,288 to 262,144. | Makes each 64-column block physically block-major so all three row bits participate in the 128-byte swizzle. |
| 14 | Reuse the K=3 tail mbarrier | `2472079` / clean generalized pipeline | Paired median 0.0047% faster, CI crosses zero; static `bar.sync` sites 43 to 42. | Reuses the now-idle stage-0 transaction barrier for the one-stage pipeline remainder. |
| 15 | Batch startup mbarrier initialization | `2472079` | Paired median 0.0375% faster, CI `[-0.0899%, +0.0025%]`; executed instructions -4,096 and barrier-stall component -0.40%. | Initializes three barriers under one leader section, then publishes them with one release fence and rendezvous. |
| 16 | One full-width output arena and TMA store | `ecfce42` / clean output-slicing infrastructure | Median 0.183% faster across three runs (0.388%, 0.113%, 0.183%); executed instructions 13.478 M to 13.444 M and output TMAs 1,024 to 512. | Replaces two `128x128` arenas/stores per output with one block-major `128x256`, 64 KiB arena/store. |
| 17 | A-only 256-byte TMA L2 promotion | `ecfce42` / `origin/pr64` support | 0.097-0.115% faster in three long confirmations; DRAM reads 71.80 MB to 70.00 MB. | Promotes only A fetches at 256-byte granularity; B/C and mixed promotion policies were neutral or slower. |
| 18 | Critical startup TMA order `0,2,1` | `ecfce42` | Three paired improvements: 0.364%, 0.423%, and 0.236% faster (median 0.364%). | Issues the immediately consumed K=0 stage first, then K=128 and K=64, without changing steady-state stage reuse. |
| 19 | Direct warp-0 plus `%laneid==0` producer guard | `32f2d32` | About 0.05-0.06% faster; longest paired result 0.051% faster, CI 0.001-0.086% faster. Static SASS 2,384 to 2,224 (-160), `ELECT` 31 to 0, `R2UR` 37 to 5. | Improves PTXAS's uniformity proof and removes all election/serialization machinery from producer guards. |
| 20 | Share kernel-scoped `%laneid` and direct special-register operands | current worktree | Paired median 0.016% faster with CI crossing zero; total PTX lines 2,519 to 2,421 (-98), symbolic b32 registers 331 to 233, SASS unchanged at 2,224. | Removes repeated template-local copies and 35 duplicate `%laneid` reads without changing hardware resources. |
| 21 | Driver PTX JIT optimization level O1 | current worktree/runtime | O1 was 0.102-0.137% faster than default JIT in three recurrence comparisons. | Exposes Driver JIT options and retains O1 for this explicit Hopper kernel; identical PTX can otherwise produce materially different SASS. |
| 22 | Precompute only A-side WGMMA descriptors | current worktree | Cooled rotated-input median 0.418% faster, 95% CI 0.154-0.483% faster; recorded Driver-O1 SASS 2,800 to 2,616. | Precomputes the A descriptor stream but keeps B as short-lived recurrence; precomputing B or both operands did not improve latency. |
| 23 | Exact `.reqntid 256,1,1` with A precompute | current worktree | Incremental 0.138% faster, 95% CI 0.044-0.296% faster; recorded Driver-O1 SASS 2,616 to 2,565 and `R2UR` 5 to 1. | Gives PTXAS the exact block-size contract; it helps only in the combined compiler form, not by itself. |
| 24 | Direct `tid.x >> 7` warpgroup ID | current worktree | Incremental 0.0865% faster, 95% CI 0.0530-0.1226% faster; recorded Driver-O1 SASS 2,565 to 2,551 (-14). | Replaces the lane-0 shuffle with a direct shift, valid for the fixed 256-thread, 128-thread-warpgroup shape. |

The final three-part compiler form was neutral end-to-end in one sustained-hot
50-by-100 comparison (`+0.0225%`, CI crossing zero). It is retained because its
components win their relevant isolated protocols and it removes substantial
SASS at unchanged resources. This is a protocol-dependent result, not a claim
that the component deltas add arithmetically.

## Start and final PTX/SASS sizes

### Live, reproducible main-to-worktree audit

Generated with the repository's builders and CUDA 13.0
`ptxas -arch=sm_90a -O1`, then counted from `nvdisasm -c`:

| State | Outputs / CTA | PTX bytes | PTX lines | PTX semicolon statements | Static SASS | Registers | Cubin bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `main` two-tile K3 kernel (`181af1f`) | 2 | 114,281 | 2,305 | 1,759 | 1,784 | 154 | 37,720 |
| Current retained-best worktree | 4 | 175,095 | 2,420 | 1,436 | 2,136 | 154 | 40,152 |

The final kernel statically contains more SASS because it exposes four outputs
per CTA rather than two. Per assigned output tile, static SASS falls from about
892 to 534 instructions (-40.1%). PTX semicolon statements fall 18.4% even
before normalizing for the doubled per-CTA output work. The closest recorded
dynamic comparison moves from the two-tile STMatrix profile's 17.341 M
executed instructions to the current final profile's 13.992 M (-19.3%), though
that comparison spans the full improvement bundle.

### Like-for-like current compiler-form audit

For the same current four-output kernel, comparing the recurrence reference to
the final A-precompute + `.reqntid` + direct-warpgroup-ID form under the same
direct PTXAS O1 audit:

| Form | PTX lines | PTX semicolon statements | Static SASS | Cubin bytes |
|---|---:|---:|---:|---:|
| Recurrence reference | 2,421 | 1,438 | 2,384 | 44,120 |
| Final retained form | 2,420 | 1,436 | 2,136 | 40,152 |
| Delta | -1 | -2 | **-248 (-10.4%)** | -3,968 (-9.0%) |

The recorded CUDA Driver O1 disassemblies in the performance notes show the
same transformation family as 2,800 to 2,551 SASS instructions (-249, -8.9%).
Direct PTXAS and Driver JIT counts must not be mixed; the live table above uses
one compiler path for both arms.

The earlier O3 static-audit sequence, before the later structural changes, was
2,608 to 2,424 SASS instructions (-184, -7.1%) at 154 registers and zero
spills. The later direct-lane change was separately 2,384 to 2,224 (-160,
-6.7%). These are valid within-sequence deltas, not a single continuous count.

## Recorded end-state benchmark results

| Benchmark | Recorded result |
|---|---|
| Long interleaved Ayla vs `torch.mm`, 650 W | 194.962 us vs 198.003 us; Ayla 1.531% faster, 95% CI 1.244-1.810% faster; 705.0 vs 694.1 TFLOP/s; 465/480 wins. |
| Paired power sweep vs `torch.mm` | Ayla faster at every tested limit: 1.314% at 300 W, 1.620% at 400 W, 1.481% at 500 W, 1.418% at 600 W, and 1.470% at 700 W. |
| Shape sweep | 4096 is the specialization point: 1.36% faster than `torch.mm`. It is 350.41% slower at 1024 and 26.34% slower at 16384. Sizes 512, 2048, and 8192 are unsupported because the two-tail lowering is incorrect. |
| Current full NCU comparison | Ayla: 92.05% tensor active, 13.992 M instructions, 86.73 MB DRAM, 12.41% achieved occupancy, 241.216 us replay workload. cuBLAS: 89.50%, 16.575 M, 175.38 MB, 14.64%, 244.512 us. |
| Standard filtered benchmark after final compiler work | 0.185 ms, 743.0 TFLOP/s, 102.0% of matched `torch.mm` in that run. |
| Four-framework isolated power sweep | Ayla reached 494.6/568.6/621.3/631.2 TFLOP/s at 350/450/550/650 W. This is a DVFS/power-efficiency result, not an equal-clock kernel ranking; at fixed 750 MHz Ayla was 1.82% slower than cuBLAS. |

The `benchmarks/` directory contains the repository GEMM harness plus the
current PTX-JIT comparison/sweep scripts. `extrapolation/benchmark/` contains
benchmark infrastructure, not an additional recorded history of this GEMM
stack. Raw CSV/JSON and NCU artifacts under `docs/profiling/` were used where
their accompanying notes establish comparable conditions.

## Primary evidence

- `docs/profiling/static-ptx-audit-2026-08-10.md`
- `docs/profiling/four-tile-followup-2026-08-11.md`
- `docs/profiling/l2-scheduling-and-k3-tail-2026-08-13.md`
- `docs/profiling/k3-nontensor-work-audit-2026-08-14.md`
- `docs/profiling/k3-fullstore-tma-cache-resweep-2026-08-14.md`
- `docs/profiling/k3-ptx-flow-audit-2026-08-14.md`
- `docs/profiling/ayla-best-vs-torch-mm-interleaved-2026-08-14.md`
- `docs/profiling/best-vs-torch-size-sweep-h100-gpu7-2026-08-14.md`
- `docs/profiling/blackwell-transfer-audit-h100-k3-2026-08-17.md`
- `docs/profiling/k3-readable-ptx-hot-loop-audit-2026-08-17.md`
- `docs/profiling/bf16-gemm-ncu-comparison-2026-08-17.md`
- `docs/profiling/gpu-kernel-benchmarking-methodology.md`
