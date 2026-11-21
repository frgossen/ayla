# Blackwell schedule refactor

The native builder now accepts a normalized `PipelineSchedule`, as an immutable
object or a dictionary. The production and split-K maps use this contract.
Legacy library callers are adapted at the builder boundary.

```python
from ayla.pipelines.ring_schedule import LoopSchedule, PipelineSchedule

pipeline = PipelineSchedule(
    operand_scope="output_jobs",
    participants="round_robin",
    reduction=LoopSchedule("rolled", unroll=2),
    output_jobs=LoopSchedule("rolled", unroll=2, peel_front=2),
)
```

The output-job scope resolves to two accumulator slots and overlapping output
by default. A serial ring explicitly selects one slot and disables overlap.
Persistence follows from the existing grid/work assignment; the operand scope
specifies whether buffering continues across those jobs.

| Normalized dimension | Meaning |
|---|---|
| `operand_scope` | Per-reduction or continuous across output jobs |
| `accumulator_slots` | One or two live output accumulators |
| `output_overlap` | Interleave output work with subsequent computation |
| `participants` | Shared participants or whole-job round-robin ownership |
| `reduction` | Fixed/rolled form, unroll factor and boundary specialization |
| `output_jobs` | Job-loop unroll factor and front/back specialization |
| `mma_pending` | Requested MMA completion lead; rolled banks require at least one |
| `writeback_window` | K-batch interval for interleaved output work |

Output-job peeling counts logical jobs; the current emitter groups two jobs
per iteration, so counts must be even. Unroll factors count emitted iterations.
Loop specialization uses `PeelLoop` and `UnrollLoop`. Required K-edge handling
is applied separately from optional specialization, preserving the measured
ordering of transformations.

For M batching, the CTA footprint and MMA shape replace the batching flag:
`output_m_tile=512, mma_m_tile=128` stages four M128 operand slices together
with one shared B operand. The retained batching backend supports M128 MMA
tiles. Other native and composed shapes continue to infer their MMA geometry.

`PipelineLoop` now prepares both rolled operand rings and cross-job resource
pipelines. Whole-job participant assignment is part of this common pass.
The dedicated `PipelineSharedRing` and `PipelineRolledK` frontend classes are
removed, along with the separate shared-ring and participant-assignment pass
modules. The fixed and indexed backend emitters retain their verified order.

The obsolete per-job carried-event generator and its lowering were removed.
They belonged to output families removed by the preceding cleanup. Their raw
IR form now fails explicitly; the retained cross-job ring uses a continuous
ordinal and independently protects operand banks, last accumulator reads and
output staging.

The changed library files fell from **6,357 to 6,346 lines**, a net reduction
of **11 lines** relative to the pre-refactor workspace, including the new typed model and compatibility adapter.
Source relocation is included in these totals; tests, maps and reports are
excluded.

## Validation

- 379 production routes and 56 split-K-table routes rebuilt with identical
  cubin bytes, launch geometry, shared-memory usage and workspace shapes.
- The isolated commit passed 891 CPU tests, with 199 skipped. The broader
  pre-existing test workspace also passed 947 tests.
- 96 direct/changed-input graph-replay checks passed in FP16 and BF16 under
  Compute Sanitizer memcheck, with zero errors. Coverage includes static and
  rolled rings, rotating participants, serial single-frame output, both M
  batching geometries, workspace split-K and cluster split-K.
- Protocol tests cover combined peeling/unrolling, uneven job counts, K tails,
  operand-bank generations and accumulator ownership. A regression test keeps
  an explicit zero MMA lead when a backend defaults to one.

Binary identity makes this refactor preserve the previously measured kernel
performance; no new schedules were selected. Validation isolates the refactor
over the frozen pre-task compiler. Concurrent descriptor changes and the later
1152x768x3072 queue-depth tuning update remain in the workspace and are excluded
from that comparison.

Detailed artifacts are in `.benchmarks/blackwell_schedule_refactor_20261120/`.
The adjacent JSON file records the compact validation and source-line summary.
