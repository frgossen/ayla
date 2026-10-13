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

Source: `.benchmarks/better_benchmark_gemm/resume_20261013/13_reference_winners.json`
in the search worktree. SHA-256:

```text
97b4eb082e60fc4ea482bd28873cef6a27041dd94ad9057adc478e695e620f3d
```

The saved MM selection geomean is **106.350713%** versus Torch over 262 canonical
BF16 MM shapes, with 35 below Torch. This combines the full eight-GPU rebaseline
with subsequent replicated replacements under inference-v2. Round 13 accepted
7 of 15 candidates. A fresh full-corpus remeasurement of this final selection
is pending; 108% has not been reached.

The 123 BF16 addmm and 46 BF16 BMM configurations are the unchanged saved winners
in the same source file. The October 13 continuation searched MM. The two
datasets' measurements and geomeans are not combined.

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
One BF16 MM winner uses WGMMA group size 4 and one pending group; two use
epilogue pipelining. These active controls are retained in the builder mapping.

This branch stores heuristic data and routing. Compiling the captured schedules
requires the compiler/scheduling stack used by the sweep, including the existing
implementations for those controls. Importing a measured configuration does not
establish its timing on the older compiler base of this branch. No compiler
backend changes or new GPU measurements are part of this import; the sweep
remains paused so the GPUs can be used for other work.
