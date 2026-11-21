# Generalizing the Blackwell schedule model

This is a design proposal, based on the current production map and compiler.
The normalized plan, generic loop shaping, and common resource/participant
entry point are now implemented; see `blackwell_schedule_refactor_20261120.md`.
The completed IR regions and fixed/indexed backend emitters remain to preserve
the measured binary behavior.

## What is already general

`peel_first_output_pair` already calls `PeelLoop` in
`PrepareRolledSharedRing`. The job loop steps by two, so peeling one iteration
specializes output jobs zero and one. This is independent of paired CTAs.

`PipelineLoop` changes the relative iteration of selected work or completion
waits. Its prologue contains selected work; `PeelLoop` specializes complete
iterations without changing their relative work schedule. They share
`clone_loop_iteration` already. Peeling is not a second implementation of
`PipelineLoop`.

The ordinary operand schedule already uses `MultiBuffer`, `PipelineLoop`,
`PeelLoop`, and `UnrollLoop`. `PipelineLoop` already shares TMEM alias proofs
with async scheduling and supports ordered Blackwell MMA completion events.
`RingSchedule` already shares policy between the static and rolled shared-ring
emitters. These are the starting points for consolidation.

## What PipelineLoop cannot currently derive

The shared ring needs more information than a selected operation and an
iteration offset:

- Operand bank identity and completion generation can span nested output-job
  and K loops: ordinal = job * K_batches + k.
- An operand bank becomes reusable after its consuming MMA completes.
- A TMEM accumulator becomes readable after final MMA completion, but becomes
  reusable after its final TMEM read. A store can still be outstanding then.
- Shared output staging becomes reusable after the store finishes reading its
  source. Global destination completion is a separate requirement.
- Work may rotate between participant groups. Event generations and waits
  must use the issuer/observer that owns each resource.
- Ragged job tails suppress future issues and drain only issued work.

The existing `PipelineLoop` transforms preallocated resources and conservatively
checks aliasing. It does not allocate those rings or derive all their release
edges. Its fixed TMEM-region tests accept disjoint columns and reject unknown
dynamic regions. The design should extend the proof model, not bypass those
rejections.

## Proposed public schedule dimensions

Separate the mathematical tiling, work assignment, resource plan, and emitted
loop form. A schedule is resolved against target capabilities before lowering.

| Dimension | Meaning | Current controls it absorbs |
|---|---|---|
| Tile geometry | CTA operand/output footprint and MMA instruction footprint | Output tile sizes, tiles per CTA, `batch_output_m_operands` |
| Work assignment | Which finite output jobs each CTA or CTA group executes | Grid size, traversal, CTA group, multicast cluster |
| Operand buffering | Bank count and whether bank lifetime spans K or output-job boundaries | `operand_stages`, part of `persistent_output_pipeline` |
| Completion lead | Number of newer MMA batches allowed outstanding before reuse waits | `cyclic_operand_pipeline` |
| Accumulator buffering | Number of output accumulators kept live | `persistent_output_frames` |
| Stage placement | Placement of previous-output reads/conversion/stores relative to current computation | `persistent_output_overlap`, `output_writeback_window` |
| Participant mapping | Shared participants or whole jobs assigned round-robin to groups | `persistent_output_job_regions`, writeback warp participation |
| Loop shaping | Rolling, partial unrolling, and boundary specialization per semantic loop | `rolled_operand_pipeline`, `rolled_k_unroll`, `rolled_job_unroll`, `peel_first_output_pair` |
| Store queue policy | Outstanding bulk-store groups and source-retirement waits | `output_store_pending_groups`, output buffer count |

The table is a normalized model, not an assertion that every combination can
be lowered today. Unsupported combinations must fail during plan resolution.
Output slices, layout and register budget remain writeback geometry controls.
Cache hints and descriptor scheduling remain separate policies.

For batched M operands, CTA operand rows exceed MMA rows while multiple MMA
subtiles share one B operand. That relationship can derive batching. Larger
instruction shapes must still be selected only when supported by the target.

Loop shaping needs stable semantic loop names such as `reduction` and
`output_jobs`, rather than `shared_operand_output_ring` implementation labels.
Peeling must state its units: one current ring iteration contains two jobs;
the equivalent setting specializes the first two jobs. Unroll settings must
likewise preserve the current iteration grouping during migration.

## Compiler representation

Represent a planned pipeline with:

1. A serial semantic loop body with a completed boundary.
2. Stage nodes for load, MMA, accumulator read, conversion, staging, and store.
3. Dependency edges carrying resource identity, producer/consumer participants,
   and iteration distance (including nested job/K coordinates).
4. A buffer-to-slot mapping and explicit reuse frontier for each resource.
5. Independent code-form policies for emitting fixed slots or indexed slots.

Use the existing generic IR operations for the implementation: `ForOp`,
`AsyncOp`, `AwaitOp`, `LockViewOp`, and ordinary memory operations. Keep the
current completed `OperandPipelineOp`/`PersistentPipelineOp` boundaries until
the common representation has equivalent memory-effect and synchronization
verification. A generic completed pipeline region can eventually replace both.

Evolve `PipelineLoop` around that resource-aware plan. Keep its existing
selected-op/offset interface as the simple case. Reuse `MultiBuffer` for slot
allocation where applicable, extending allocation scope and event generation
handling for rings spanning nested loops. Reuse `RingSchedule` stage policy
while extracting its resource dependencies. Do not add a parallel generic
pipeline pass with another prologue/epilogue generator.

Do not put raw hardware barrier phases into the tuning map. Derive slot and
event generation from resource reuse. Lower completion edges using target
capabilities: Hopper pending-group waits, Blackwell completion events, and bulk
store queues have different mechanisms.

## Mapping the current schedules

The 379 production configuration records contain these families:

| Current family | Routes | Normalized description |
|---|---:|---|
| Ordinary route/backend | 294 | Existing work assignment and per-job operand scheduling |
| Static shared ring | 4 | Cross-job operand ring, two accumulators, interleaved output, fixed-slot emission |
| Serial rolled ring | 1 | Cross-job operand ring, one accumulator, serial output, indexed-slot emission |
| Common rolled overlap | 19 | Cross-job operand ring, two accumulators, interleaved output, indexed-slot emission |
| Whole-job rotation | 61 | Cross-job operand ring, two accumulators, rotating four-warp job owners, indexed-slot emission |

The ordinary family includes workspace and cluster split-K backends, which
retain their existing lowering. Persistence follows from the work assignment
and the existence of multiple jobs; it does not need an additional boolean.
The cross-job buffering policy remains an independent decision. A serial
cross-job ring is not equivalent to disabling that ring entirely.

## Migration and acceptance

1. Introduce a typed normalized plan and adapters from every current route.
   Centralize legality/default resolution. Initially select the same emitters
   and preserve their exact pass order and loop grouping.
2. Introduce semantic loop-shaping policy. Dispatch peeling and unrolling
   through `PeelLoop`/`UnrollLoop`; remove kernel-specific shaping flags after
   adapters and other library callers migrate.
3. Extract resource-frontier construction shared by the operand and output
   rings. Feed it to the resource-aware `PipelineLoop` path. Preserve existing
   verification of issuer ownership, generations, aliases, and completion.
4. Consolidate fixed-slot and indexed-slot emission behind that plan. Remove
   specialized emitters only once their complete route families are covered.
5. Serialize normalized plans in the production map and remove legacy adapters
   after decoder, split-K and other library users have migrated.

For steps one and two, require exact cubin, launch, shared-memory and workspace
identity across the 379 production and 56 split-K-table routes. Add meaningful
tests for alias conflicts, slot reuse before last read, late store source
reads, actor-specific event generations, single-job and ragged tails, and
equivalence of transformed loop iteration counts. Keep the previous timings as
baseline. If later consolidation changes binaries, run correctness/memcheck
and paired performance comparisons for all affected routes before promotion.

The existing generic loop tests were checked during this investigation:
16 passed across `test_pipeline_loop`, `test_pipeline_loop_tmem_regions`, and
`test_peel_loop`.
