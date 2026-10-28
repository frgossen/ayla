"""Audit the installed-route run and produce a report with explicit score scope."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ps=json.loads((HERE/'priorities.json').read_text());verified=json.loads((HERE/'verified.json').read_text())
winners={r['target']:r for r in verified if r['status']=='updated'}
rows_by_shape={}
for path in (HERE/'final').glob('[0-9][0-9][0-9][0-9].json'):
    rows=json.loads(path.read_text());assert len(rows)==2 and all(r['status']=='ok' for r in rows),(path,rows)
    rows_by_shape[rows[0]['target']]={r['entry']['kind']:r for r in rows}
assert set(rows_by_shape)=={p['target'] for p in ps}
graph=[r for path in (HERE/'graph').glob('[0-9][0-9][0-9][0-9].json') for r in json.loads(path.read_text())]
assert len(graph)==len(winners) and {r['target'] for r in graph}==set(winners)
assert all(r['status']=='ok' and len(r['checks'])==9 for r in graph)
confirmation=[r for path in (HERE/'confirmation').glob('[0-9][0-9][0-9][0-9].json') for r in json.loads(path.read_text())]
assert len(confirmation)==288 and all(r['status']=='ok' for r in confirmation)
for target,rows in rows_by_shape.items():
    old,new=rows['control'],rows['installed']
    if target in winners:
        assert old['code_fingerprint']!=new['code_fingerprint']
        hashes={r['code_fingerprint'] for r in confirmation if r['target']==target and r['entry']['kind']=='candidate'}
        assert hashes=={new['code_fingerprint']},(target,hashes,new['code_fingerprint'])
    else:assert old['code_fingerprint']==new['code_fingerprint'],target
gm=lambda values:math.exp(st.mean(math.log(x) for x in values))
metrics=dict(scope='35 requested BF16 shapes, equally weighted; current production JIT',
    before_geomean_pct=gm([r['control']['torch_pct'] for r in rows_by_shape.values()]),
    after_geomean_pct=gm([r['installed']['torch_pct'] for r in rows_by_shape.values()]),
    before_call_weighted_pct=100*sum(p['calls']*rows_by_shape[p['target']]['control']['torch_ms'] for p in ps)/sum(p['calls']*rows_by_shape[p['target']]['control']['ayla_ms'] for p in ps),
    after_call_weighted_pct=100*sum(p['calls']*rows_by_shape[p['target']]['installed']['torch_ms'] for p in ps)/sum(p['calls']*rows_by_shape[p['target']]['installed']['ayla_ms'] for p in ps),
    unchanged_kernel_shapes=len(ps)-len(winners),changed_kernel_shapes=len(winners),
    unchanged_median_normalized_ratio=st.median(rows['installed']['torch_pct']/rows['control']['torch_pct'] for target,rows in rows_by_shape.items() if target not in winners),
    changed_only_geomean_ratio=gm([rows['installed']['torch_pct']/rows['control']['torch_pct'] if target in winners else 1 for target,rows in rows_by_shape.items()]),
    eight_replica_confirmed_geomean_ratio=gm([winners[target]['candidate_pct']/winners[target]['old_pct'] if target in winners else 1 for target in rows_by_shape]))
metrics['relative_geomean_gain']=metrics['after_geomean_pct']/metrics['before_geomean_pct']-1
screen=[r for d in ('transfer','followup','ncu_concepts') for path in (HERE/d).glob('[0-9][0-9][0-9][0-9].json') for r in json.loads(path.read_text())]
candidate=[r for r in screen if r['entry']['kind']!='control']
metrics.update(screen_trials=len(screen),candidate_trials=len(candidate),screen_controls=len(screen)-len(candidate),
    screen_rejections=sum(r['status']!='ok' for r in candidate),
    distinct_generated_code_signatures=len({(r['target'],r['code_fingerprint']) for r in candidate if r['status']=='ok'}),
    confirmation_pairs=len(confirmation),installed_route_pairs=70,installed_route_graph_checks=99,
    ncu_profiles=len(json.loads((HERE/'ncu_summary.json').read_text())))
(HERE/'final_summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
summary=json.loads((HERE/'screen_summary.json').read_text())
search_counts={r['target']:r['trials'] for r in summary}
report=f'''# Priority BF16 GEMM transfer search

## Current results

The 35 BF16 signatures supplied in the priority list were measured again on
`jit-gemm-heuristics`, starting at `45c9ded`. All eight H100 GPUs were idle when
this search began. These are fresh production-JIT measurements, with the same
normal compiler and inference-v2 benchmark used before and after.

| Scope | Before, % of Torch | Installed, % of Torch |
| --- | ---: | ---: |
| Equal-weight geomean, all 35 requested shapes | {metrics['before_geomean_pct']:.3f} | {metrics['after_geomean_pct']:.3f} |
| Call-weighted speed, the same 35 shapes | {metrics['before_call_weighted_pct']:.3f} | {metrics['after_call_weighted_pct']:.3f} |

The geomean gain is {100*metrics['relative_geomean_gain']:.3f}% relative.
Eleven kernels changed; 24 kernels are identical before and after. Holding
unchanged shapes neutral gives a {100*(metrics['changed_only_geomean_ratio']-1):.3f}%
geomean gain in the final run. The independent eight-GPU confirmations give
{100*(metrics['eight_replica_confirmed_geomean_ratio']-1):.3f}% for that same
change-only calculation. The median normalized ratio on unchanged kernels is
{metrics['unchanged_median_normalized_ratio']:.5f}; small before/after differences
on those rows are measurement variation.

This is the requested 35-shape scope, which accounts for about 75.7% of the
historical claimed Torch GEMM time. It is not a full-corpus score. The full
MM/BF16 corpus and the legacy native backend were not remeasured, and these
numbers must not be combined with their historical geomeans. The 108% target
has not been reached in this 35-shape scope.

## Confirmed updates

Each row below is the median of eight independent GPU replicas, measured with
fixed candidate schedules and the original incumbent. This table has its own
sampling runs; the 35-shape table below comes from the final installed-route run.

| Priority | Operation M×N×K | Before % | Candidate % | Conservative gain | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
'''
for r in verified:
    report+=f"| {r['rank']} | {r['op']} {r['m']}×{r['n']}×{r['k']} | {r['old_pct']:.3f} | {r['candidate_pct']:.3f} | {100*(r['conservative_gain']-1):.3f}% | {r['status']} |\n"
report+='''
The deep-K 4096×4096×16384 candidate gained 0.990%, just below the predeclared
1% admission threshold, and was retained. Every installed winner had positive
conservative gains on all eight GPUs, stable Torch references on all eight,
and a different compiled executable from its incumbent.

## What was searched

- Read and froze all 35 priorities before GPU work. A fresh baseline identified
  18 shapes below 100.5% of Torch, including near-ties.
- First pass: 366 distinct configurations, 18–24 per shape. Donors came from
  successful same-N/K shapes at other M values, nearby BF16 MM/addmm shapes,
  FP16 schedules, and legacy native measurements used as hypotheses.
- Second pass: 67 configurations over seven remaining deficits. Newly successful
  25216-M and 32768-M schedules were transferred back to the large-M family;
  fixed-grid donors were tested with their existing rolled-loop option.
- Last pass: ten configurations. Six rolled-grid trials followed the NCU
  large-N observation; four checked legal buffer/slice and resource combinations.
- Total: 443 distinct candidate configurations, 107 interspersed controls,
  423 successful candidate trials, 20 explicit compiler/resource/layout
  rejections, and 370 distinct normalized PTX/launch/optimization signatures.
  This last count describes generated code; final executable distinctness is
  checked separately using SASS, resources, and launch signatures.
- Diversity: M tiles 64/128/192/256; 13 N tile sizes from 80 through 256;
  K tiles 64/128; 2/3/4/5/6/8 operand stages; M/N output grouping;
  fixed unrolled and rolled grids; warpgroup geometry; output slices and
  buffers; cyclic pipelines; clusters; cache hints/promotions; PTX optimization
  levels; normal async rescheduling. Successful measured donors all used
  stmatrix output stores, so this pass did not enumerate other store mechanisms.

The successful transfers were:

- 175360×3072/2304×768 schedules to 25216/25344/32768-M shapes: the same
  existing JIT configuration now covers four additional targets.
- The 25216/25344×3072×768 geometry to their 2304-wide siblings: about 10.7%.
- MM schedules to addmm at 16384×4096×1024 and 32768×768×3072.
- A nearby 4096×6144×1536 MM geometry to 4096×8192×2048 addmm.
- A 4096×2560×2560 BF16 MM donor to 1000×10240×2560 MM: about 12.6%.
- A 4096-M same-N/K donor to 8192×1024×1024 addmm.

Only two new JIT configuration rows were needed. Nine updates reuse existing
rows. This pass did not repeat the earlier 18,352-tuple Cartesian sweep.

## NCU outcomes

Five kernel-replay profiles were collected using normal Torch/Ayla kernels,
cold-cache counter passes, and unchanged clocks. NCU durations are diagnostics
and are never used to select or admit configurations.

| Shape | Observation | Outcome |
| --- | --- | --- |
| addmm 16384×4096×1024 | Incumbent SM throughput 76.69%; transferred M256 geometry 81.72%; Torch 80.66%. Registers/thread fall from 177 to 107; threads/CTA rise from 256 to 512. | The transfer passed independent benchmark confirmation. |
| mm 8192×128112×1024 | Ayla uses 56,960 CTAs and 82.30% SM throughput; Torch uses 132 persistent CTAs and 85.51%. | Six normal rolled-grid trials were slower: 3.44–9.25 ms versus roughly 3.21–3.24 ms for the incumbent. Retained incumbent. |

Remaining deficits include the large-N MM, wide 4096×16384×4096 addmm,
the 175360-M family, and the improved 16384×4096×1024 addmm. Transfers
from smaller-M winners frequently lost performance at 175360 M. No compiler
changes were made to pursue these remaining gaps.

## Measurement and correctness

The ordinary production `build_gemm_module` and `GemmScheduleSeed` path built
all candidates. No compiler passes or shared-memory lifetime rules were added
by the benchmark. `unsafe_reuse_smem` was not enabled. The compiler, runtime,
JIT builder, heuristic selector, and `benchmarks/common.py` match their
pre-search source hashes; only heuristic data changed in the library tree.

Screen ranking uses the smaller of the raw Ayla speedup and the paired
Torch-percentage speedup against nearby incumbent controls. Identical compiled
programs receive no gain. A candidate's own unusually slow Torch reference
therefore cannot establish a win.

Twelve fixed candidates were confirmed across eight independent GPUs each,
with varied row order and repeated incumbent controls: 288 normal Torch/Ayla
pairs. Admission requires a 1% median conservative gain, a 1% raw-latency gain,
at least six positive replicas, at least six stable references, a median
Torch-reference ratio within 2%, and changed SASS/resources/launch signature.
All eleven installed winners exceeded these gates.

Each confirmation used three correctness seeds, BF16 maximum absolute error
32, relative RMS error 0.002, and explicit finite-error checks. The final run
timed the actual installed `gemm_op` routes for all 35 shapes against their
frozen pre-search schedules, and checked that installed compiled programs
match the confirmed candidates. All eleven installed routes also passed nine
CUDA graph checks each, using independent weight/bias sets and subsequently
changed inputs.

Sampling options are applied in the worker process, avoiding the earlier spawn
inheritance bug. Screening used a 0.5-second adaptive budget and 200-round cap;
the final ten concept trials used 0.75 seconds and a 300-round cap;
confirmations used 2 seconds and a 1,000-round cap; final installed-route
comparisons used 1.5 seconds and a 750-round cap. Actual rounds and layer counts
are stored per result; caps are not claimed as actual sample counts.

## Final installed-route results for all 35 priorities

| Priority | Operation M×N×K | Before % of Torch | Installed % of Torch | Candidate trials | Config changed |
| ---: | --- | ---: | ---: | ---: | --- |
'''
for pr in ps:
    old,new=rows_by_shape[pr['target']]['control'],rows_by_shape[pr['target']]['installed']
    report+=f"| {pr['rank']} | {pr['op']} {pr['m']}×{pr['n']}×{pr['k']} | {old['torch_pct']:.3f} | {new['torch_pct']:.3f} | {search_counts.get(pr['target'],0)} | {'yes' if pr['target'] in winners else 'no'} |\n"
report+='''
## Integration and evidence

The eleven winners are merged into the existing branches
`oss-gemm-benchmark-heuristics` and `jit-gemm-heuristics`.
Both existing JSONs have eleven additional typed BF16 exact rows. All 755
pre-existing exact rows, including all 749 FP16 rows and the six previous BF16
winners, are preserved. New BF16 groups retain the old tree/fallback for other
shapes. The JIT table updates eleven routes; all original seeds remain as
fallbacks, and all unrelated routes and original configuration rows are preserved.

JIT CPU tests: 21 passed. OSS heuristic CPU tests: 24 passed. GPU correctness,
installed-route equivalence, and changed-input graph checks passed as described
above. These fresh GPU results are for the JIT branch. The OSS branch carries
the same source-confirmed data; its separate static backend was not GPU timed.
That branch's minimal static builder does not implement every captured control
(for example fixed output grids and explicit M warpgroup tiling), so these
source-JIT timings are not a claim that its static seed runner reproduces them.
The reproduction runners require the JIT branch. No builder APIs were added
to the OSS branch in this data update.

`priority_gemm_transfer_search/evidence.json.gz` stores the frozen priorities,
source hashes, manifests, raw results, confirmations, graph checks, and NCU
summaries. Full PTX/cubin/SASS and NCU reports remain in
`.benchmarks/priority_transfer_20261014/` in the JIT worktree. The benchmark-only
runners are included alongside the evidence with reproduction instructions.
'''
(HERE/'report.md').write_text(report)
print(json.dumps(metrics,indent=2))
