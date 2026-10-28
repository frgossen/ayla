# Consolidated PR branches

## Branches

- `review/main-small-fixes`: based on main `24e1e2a02bf6b5b444ef13afa46cc767aabbf61b`, tip `0ca0460`.
  Worktree: `/home/arseniivanov/ayla-main-review`.
  Contains #477 (stable PTX cache keys) and #490 (benchmark comparison), plus
  fixes for relative benchmark/output paths and an empty common benchmark set.
  Changes do not alter scheduling or the core compiler.
- `feature/consolidated`: includes the main branch above, all 40 selected
  feature PR heads, the committed `jit-gemm-heuristics` baseline `85cf491`, and
  later multicast work from `cluster-mbar-reuse` at `faad274`.
  Worktree: `/home/arseniivanov/ayla-feature-consolidated`.
  Merge commits preserve the selected PR ancestry. Use this branch for ongoing
  core library and performance development.

No branches were pushed and upstream main was not changed. Existing worktrees
were preserved. Uncommitted experiments in the original worktrees were not
swept into this branch; this consolidation covers the PR snapshot and the two
committed integration baselines named above.

## Integration repairs

The PRs overlap, so merging their heads required adapting the combined code:

- Reconciled static module JIT, shape specialization, FX route plans, and lazy
  registration under one device/shape/stride/dtype metadata contract.
- Preserved measured GEMM scheduling for BF16 and FP16, including addmm and
  batched GEMM, in the FX route. Verified CUDA graph replay after warmup.
- Combined pipeline extraction with the extended tiling, fixed-grid traversal,
  reverse traversal, and cyclic WGMMA scheduling controls.
- Carried scalar/tensor matmul initialization and scale-d overwrite through
  promotion and tiling. Ported SwiGLU to axis-specific block IDs, WGRMEM
  fragments, and explicit fills.
- Combined narrow blocked staging and transposed STMatrix stores; preserved
  rank-N TMA descriptors and batched views.
- Kept multicast split release and mbarrier reuse. Initialization publication
  uses relaxed cluster arrival, while other cluster barriers retain release.
- Updated older test snapshots for concise constant syntax, folded redundant
  hardware-index modulo operations, and the new publication/release behavior.
  Correctness tests remain active. Lint fixes were needed in inherited scripts.

## Verification

Environment: PyTorch 2.13 CUDA 13.0, NVIDIA H100. The Pixi environment is at
`/home/arseniivanov/ayla/.pixi/envs/default`. GPU verification used isolated
visible devices 0 and 1. Torch's automatic ccache wrapper is broken on this
machine; `TORCH_NO_COMPILER_WRAPPER=1 CXX=/usr/bin/g++` bypassed it.

| Branch | Check | Result |
| --- | --- | --- |
| main fixes | Full suite with CUDA unavailable | 451 passed, 64 skipped |
| main fixes | Ruff / Pyright | Passed / 0 errors |
| main fixes | Benchmark CLI checks | Relative paths, empty intersection, candidate selection, failed-row filtering, cache isolation and retained outputs checked |
| feature | Full suite, GPU enabled | **816 passed**, no skips, 36.13 seconds |
| feature | Ruff / Pyright | Passed / 0 errors |
| feature | Kernel collection, end-to-end and lock subset | 129 passed |
| feature | Additional BF16/FP16 mm, addmm, bmm and graph replay | 6 passed; included in final full suite |
| feature | Selected PR ancestry | All 40 feature heads verified |

Final test warnings were PyTorch deprecation/cache warnings. All selected PR
heads refer to the collector snapshot from 2026-10-28; later upstream changes
are outside this verification.

### Benchmarks

The standard 4096 BF16 GEMM benchmark completed all 10 configurations, with
200 rounds each. Torch median: **0.1958 ms**, **701.8 TFLOPS**. Fastest Ayla
configuration (packed allocations, split output, four tiles per CTA):
**0.1908 ms**, **720.2 TFLOPS**. The 2.6% throughput difference is small compared
with the observed timing variation and should not be treated as a broad speedup.
Kernel numerical correctness was exercised in the GPU suite.

SwiGLU was compared with eager and compiled Torch on the same graph. Each timed
variant was numerically checked against Torch (`atol=0.125`, `rtol=0.006`, as in
the existing SwiGLU GPU tests), and checked for finite outputs before capture.

| M | Torch compiled median | Ayla compiled median | Relative throughput |
| --- | --- | --- | --- |
| 1 | 0.3939 ms | 0.3877 ms | 1.6% |
| 32 | 0.4041 ms | 0.3840 ms | 5.2% |
| 512 | 0.6883 ms | 0.6297 ms | 9.3% |

These are the standard hot-cache CUDA graph measurements. The full OSS shape
corpus, cold-cache performance, and other GPU architectures were not benchmarked.
The results support using the feature branch; they do not establish that every
schedule is faster than Torch.

## Push commands

Run these after reviewing this report. They push the prepared branches, not main:

```bash
git -C /home/arseniivanov/ayla-main-review push -u origin review/main-small-fixes
git -C /home/arseniivanov/ayla-feature-consolidated push -u origin feature/consolidated
```

Merge `review/main-small-fixes` into upstream main through the normal GitHub
flow. Keep `feature/consolidated` as the development branch. If the upstream
repository does not allow feature-branch pushes, add a fork remote and replace
`origin` in the second command with that remote; the branch is ready to push.

## GitHub PR cleanup

**Keep these nine PRs open:**

| PR | Title | Reason to keep |
| --- | --- | --- |
| [#81](https://github.com/frgossen/ayla/pull/81) | Benchmark the recorded model shapes ayla has a kernel for | Draft benchmark of the collection's recorded model shapes. Reconcile with newer OSS corpus/JIT benchmarking before keeping another harness. No urgent main fix. |
| [#196](https://github.com/frgossen/ayla/pull/196) | [DO NOT MERGE] Debug-only fastest GEMM transformation pipeline | Explicit DO NOT MERGE; preserves a debug-only inlined transformation pipeline. Keep as a reference. |
| [#239](https://github.com/frgossen/ayla/pull/239) | Prototype Muse Glimmer MLP subgraph capture | Hardcoded Glimmer MLP capture with unimplemented lowering; newer generic FX/SwiGLU work addresses this area. Preserve useful tests instead of installing the old matcher. |
| [#257](https://github.com/frgossen/ayla/pull/257) | Capture Muse Glimmer mean reduction | Draft mean-reduction stack with incomplete lowering and stale `integer-dict-attrs` base. Extract useful reduction work later after current IR/JIT choices settle. |
| [#292](https://github.com/frgossen/ayla/pull/292) | [DO NOT MERGE] Contains affine index analysis, which may be useful at some later point  | Explicit DO NOT MERGE; affine/TMA-grid analysis prototype and missing grid-barrier lowering. Preserve for future analysis. |
| [#312](https://github.com/frgossen/ayla/pull/312) | [DO NOT MERGE] Benchmark captured Torch graphs | Explicit DO NOT MERGE. Benchmark work also replaces compiler routing; useful graph/profiling parts could be extracted after FX integration. |
| [#432](https://github.com/frgossen/ayla/pull/432) | [DEFER TO QUEUE, BUT IMPORTANT] Pipeline and overlap wgmma group execution | Explicitly deferred in favor of a WGMMA queue design. Existing integration snapshots contain older variants; identify and replace them with the agreed queue design before treating the feature branch as ready. |
| [#485](https://github.com/frgossen/ayla/pull/485) | Remove four slow GEMM kernels | Removes four named kernel implementations and their tests. It is not a required correctness fix. Decide whether they serve as examples/reference coverage before deleting them; resolve registry/lowering overlap if adopted. |
| [#487](https://github.com/frgossen/ayla/pull/487) | Remove unused CUDA experiments | Deletes CUDA examples and the fast.cu submodule. Depends on #485 because the removed kernel uses that reference. If cleanup is wanted, order #485 then #487. |

Do not close the selected feature PRs until the consolidated branch is pushed
and its commits are safely stored remotely. Their code is integrated on the
feature branch, so they can then be closed as superseded by that branch. This
is not the same as merging their features into upstream main. Close #477 and
#490 once the small-fixes branch has actually been merged into main.

Selected feature PRs:

#315, #321, #342, #343, #344, #345, #346, #347, #348, #349, #351, #356, #410, #425, #429, #430, #434, #435, #436, #455, #456, #461, #465, #467, #468, #472, #476, #479, #480, #482, #483, #486, #489, #491, #492, #493, #498, #500, #501, #502.

## When to fork

A busy reviewer alone does not require a fork: the upstream feature branch
provides room to integrate and validate the work while preserving a small,
stable main. Fork when independent maintenance becomes necessary: incompatible
public APIs or scheduler choices, a separate release/support policy, loss of
upstream push authority, or sustained divergence that upstream will not accept.
At that point, push this existing feature branch to the fork, retain upstream
as a remote, and continue submitting small broadly useful fixes upstream.

## Local evidence

Collector metadata, individual PR diffs, routing decisions, exact command logs,
and benchmark results live in `/home/arseniivanov/ayla/.git/pr-triage/`:

- `routing.json`, `fetched-inventory.json`, `REVIEW.md`
- `feature-tests-final.txt`, `feature-lint-final.txt`, `feature-typecheck-final.txt`
- `feature-benchmark-gemm.txt`, `feature-benchmark-swiglu.txt`,
  `feature-benchmark-swiglu.json`
- `snapshot-reconciliation.json` records the four updated snapshot differences.
