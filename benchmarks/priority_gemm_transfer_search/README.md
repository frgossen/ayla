# Priority BF16 transfer search evidence

See [the results report](../PRIORITY_BF16_GEMM_TRANSFER_RESULTS.md) for score
scope, all 35 shape results, donor transfers, NCU outcomes, and admission gates.

`evidence.json.gz` is a gzip-compressed JSON object. `source` records the
pre-search branch/commit and source hashes, `summary` records the aggregate
results, and `artifacts` maps relative artifact filenames to JSON contents.
It includes every screening result, rejected configuration, fixed-candidate
manifest, independent confirmation, installed-route comparison, and graph check.
NCU summaries include metric units. Full PTX/cubin/SASS and NCU reports remain
in the original ignored artifact directory.

These runners require `jit-gemm-heuristics` and its normal production JIT API.
The OSS branch stores the same source-confirmed data and compressed evidence;
its minimal static builder does not implement every captured schedule control.

## Inspect without GPU work

```bash
python3 - <<'PY'
import gzip, json
from pathlib import Path
evidence = json.loads(gzip.decompress(Path('benchmarks/priority_gemm_transfer_search/evidence.json.gz').read_bytes()))
print(json.dumps(evidence['summary'], indent=2))
print(json.dumps(evidence['artifacts']['verified.json'], indent=2))
PY
```

## Reproduce installed-route comparisons

From the JIT worktree, use its Torch/CUDA Python environment. Unpacking is CPU
work and preserves the original evidence. The second command uses the GPUs
listed in `--gpus`; choose the available devices.

```bash
python benchmarks/priority_gemm_transfer_search/unpack.py
python .benchmarks/priority_transfer_replay_20261014/drive.py \
  .benchmarks/priority_transfer_replay_20261014/final_jobs.json \
  --phase rerun_final --gpus 0,1,2,3,4,5,6,7 \
  --seconds 1.5 --rounds 750 --timeout 240
```

Use `confirmation_jobs.json`, `--confirm --seconds 2 --rounds 1000`, and a new
phase directory for fresh independent confirmations. `graph_jobs.json` with
`--graph` runs the installed-route changed-input CUDA graph checks.

The manifests are frozen. `prepare.py` documents the original priority/donor
capture and refers to the original attachment and legacy worktree; it is not
needed to replay the frozen manifests. Reruns on a changed compiler require
fresh measurements; the original scores remain tied to their source hashes.

The runners change only caller sampling options. They use normal library seeds,
the production JIT builder, and the unchanged inference-v2 timing function.
Every completed pair is saved before the next trial. The driver starts one
worker per GPU, checks for other applications before starting each job, and
uses job timeouts. Only final fixed candidates use SASS executable fingerprints.
