# Combined FP16 and BF16 GEMM heuristic

`ayla/oss_gemm_heuristic.json` includes the existing OSS FP16 winners and the
saved BF16 winners from the better-benchmark dataset. Exact dispatch uses
`(operation, dtype, batch, M, N, K)`. The 55 MM geometries shared by the two
datasets retain independent configurations and measured scores for each dtype.

| Operation | Existing FP16 signatures | Added BF16 signatures | Total |
| --- | ---: | ---: | ---: |
| MM | 405 | 262 | 667 |
| addmm | 205 | 123 | 328 |
| BMM | 139 | 46 | 185 |
| All | 749 | 431 | 1,180 |

All 749 existing FP16 exact entries, their decision trees, and the fallback
are preserved. Six BF16 operation/alignment groups provide neighboring seeds
for unmeasured shapes. The 431 measured BF16 signatures have exact entries;
the trees do not replace their winning configurations. Unsupported dtypes
still fail instead of silently borrowing a schedule from another dtype.

## BF16 measurement source

Source: `.benchmarks/better_benchmark_gemm/ncu_goal_20261013/172_full_winners.json`
in the search worktree. SHA-256:

```text
94059a6a0761d9d36f73111f3abf42780e15bb516a426e34e63206f71a29ddfa
```

The latest full eight-GPU measurement is **107.266386214% geomean** versus
Torch over all 262 canonical BF16 MM shapes, with 16 below Torch. All 2,096
Ayla measurements passed correctness under inference-v2. Exact shape-median
geomean: 107.269552790%. The 108% goal remains open.

All 262 BF16 MM exact configurations and measured scores are refreshed from
full172, together with 27 existing BF16 MM tree exemplar seeds. Tree boundaries
are preserved. The 123 BF16 addmm and 46 BF16 BMM exact entries, their trees,
all 749 FP16 entries and trees, and the fallback are unchanged. The datasets'
measurements and geomeans are not combined.

`BETTER_BENCHMARK_BF16_IMPORT.json` records the source hash, full validation,
compiler/caller source freeze, and import preservation checks.

### Search results and pause

The latest transfer round completed 4,274 correct trials (4,148 fresh variants
across 126 shapes plus controls). Eight-GPU confirmation accepted two
M128/N160/s3 two-column-warpgroup phase-reuse schedules:

| Shape M×N×K | Paired normalized gain | Paired raw gain | Normalized wins |
| --- | ---: | ---: | ---: |
| 4096×50264×2048 | 1.332666% | 0.636649% | 8/8 |
| 512×197952×256 | 3.459545% | 3.546083% | 8/8 |

The full172 aggregate above supersedes the mixed confirmation aggregate.
Subsequent blocked-transposed budget probes yielded 12 correct pairs and 36
TMA descriptor rejections before kernel execution, with no lead. Larger-K
probes yielded 96 correct pairs and no lead. Row-partitioned two-WG probes
were stopped after 49 complete correct pairs; partial records are retained in
the search checkout. No winner from an unfinished round was imported. The user
paused the search, and all owned GPU contexts were released.

### Library and benchmark dependencies

No compiler/library code was added during this continuation or this import;
the 13 compiler source hashes remain unchanged. Confirmed resource budgets
and two-column writer ownership were composed in the benchmark shared-memory
reuse helper using existing native IR and passes.

The measured ranking depends on the search checkout's benchmark callers,
including `oss_gemm_smem_reuse.py` (completed compute/writeback phase reuse),
`oss_gemm_input_gather.py` (exact measured full-block input rendezvous), and
existing narrow-store, output-wave, WGMMA-completion and split-K compositions.
Those compositions are not encoded by tile dimensions alone. The import keeps
the complete captured configurations and their source scores; the older
compiler base and generic seed compiler on this heuristic branch have not
been validated to reproduce 107.266386214%. The source freeze identifies the
required compiler and caller versions. This is a heuristic data update, with
no backend port or fresh GPU measurements on the export branch.

## Selecting a dtype

```python
from ayla import select_gemm_schedules

fp16 = select_gemm_schedules(128, 4096, 4096, dtype="float16")
bf16 = select_gemm_schedules(128, 4096, 4096, dtype="bfloat16")
```

`float16` remains the default. The same dtype option works with
`operation="addmm"` and `operation="bmm", batch=...`.

```bash
pixi run bench-oss-seed -- 128 4096 4096 --dtype float16 --show-only
pixi run bench-oss-seed -- 128 4096 4096 --dtype bfloat16 --show-only
```

The seed tool passes the selected dtype to the semantic IR, tensor allocation,
Torch baseline, correctness tolerance, and result metadata.

## Merge and verification

The existing `merge-mapper` converter now retains incoming exact entries and
rejects duplicate operation/dtype/batch/shape keys. Previously it merged only
the trees. Compact expand/collapse round trips remain lossless.

CPU verification checks exact dispatch for every one of the 1,180 signatures,
preservation of all existing FP16 entries/trees/fallback, and preservation of
all active BF16 captured settings. Inactive historical fields are omitted.
Captured BF16 controls, including warpgroup tiling, physical writeback settings,
WGMMA completion and split-K choices, are retained in the JSON. Generic builder
translation alone does not reproduce the benchmark compositions listed above.

This branch stores heuristic data and routing. Compiling the captured schedules
requires the compiler/scheduling stack used by the sweep, including the existing
implementations for those controls. Importing a measured configuration does not
establish its timing on the older compiler base of this branch. No compiler
backend changes or new GPU measurements are part of this import; the sweep
remains paused so the GPUs can be used for other work.
