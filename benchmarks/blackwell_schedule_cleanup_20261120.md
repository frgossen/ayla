# Blackwell scheduling cleanup — 2026-11-20

Eight production routes were replaced: six with faster schedules and two by changes that produce identical code. The map still has 379 FP16 MM routes: 298 native, 63 workspace split-K and 18 cluster split-K.

Configuration fields fell from 5,494 to 3,621. Default-valued options, zero issuer annotations, inferred cluster-M sizes and ignored output-column overrides were removed. Three wrapped direct routes were flattened with their effective defaults preserved. Split-K dispatch markers stay explicit even when they equal a builder default.

The native frontend no longer exposes 22 unused or redundant controls. Four unselected output families (`paired`, `pipelined`, `framed`, `shared`) and ten unused pass modules were removed. Shared-ring and whole-job ping-pong schedules remain. Gather selection follows TMEM lifetime automatically, and integer indexing follows rolled-loop selection.

TMEM allocation reuse remains in the library because the separate split-K table uses it. Operand/output storage reuse remains because decoder routes use both single-operand and complete-arena reuse. Neither is an independent selection in this production MM map. Generic completion and memory-visibility machinery required by the retained schedules remains.

## Reconfirmed replacements

Ratios below are replacement/incumbent calibrated medians; lower is faster. Identical cubins qualify by byte identity rather than noisy timing differences between independent module loads.

| Shape M×N×K | Complete GPU0 / GPU1 | Inference-v2 GPU0 / GPU1 |
|---|---|---|
| 6272x1280x256 | Identical code | Identical code |
| 8192x512x512 | 0.9786 / 0.9860 | 0.9792 / 0.9784 |
| 8192x768x768 | 0.9483 / 0.9543 | 0.9538 / 0.9597 |
| 12544x384x1152 | 0.9586 / 0.9543 | 0.9667 / 0.9633 |
| 12544x384x1536 | 0.9661 / 0.9659 | 0.9763 / 0.9805 |
| 12544x512x1024 | 0.9830 / 0.9831 | 0.9904 / 0.9987 |
| 25216x320x960 | Identical code | Identical code |
| 401536x64x192 | 0.9762 / 0.9849 | 0.9478 / 0.9641 |

Changed executables passed on both GPUs and both metrics: no slower calibrated median, one-sided 95% bootstrap upper bound below 1.01, and median within 1% of the faster identical incumbent control. Each confirmation used two reversed module-load/capture orders and three batches of 21 alternating rounds per metric. This bounds measured regressions; it is not a whole-model performance claim.

435 schedules (379 production plus 56 split-K table entries) were rebuilt before and after cleanup using the same selected geometry. All matched cubin bytes, launch geometry, shared-memory size and workspace shape. The isolated commit tree reproduced all 435. The eight updated production routes additionally passed 96 direct and changed-input graph-replay checks, with FP32 references, output guards and input ownership checks, in FP16 and BF16 under Compute Sanitizer memcheck (zero errors).

The isolated commit tests passed: 792 tests, with 199 skipped in the CPU run. Undefined-name and unused-import checks passed for all changed Python sources and tests.

Raw measurements, cubins, PTX, source snapshots and compiler comparisons are retained locally under `.benchmarks/blackwell_cleanup_20261120/`. The compact performance summary is in the adjacent JSON report.

## Remaining heuristic groups

These are selections present in the 379-route production map, including resolved builder defaults. They are not exhaustive hardware/API limits. Booleans below mean both values are selected somewhere unless marked fixed.

### Whole-GEMM strategies / geometry

- Orientation: `transpose_output=false/true` (TNN/TNT equivalence).
- MMA pairing: `cta_group=1/2`, independent of multicast cluster size.
- Multicast: enabled; cluster-M follows CTA group, `cluster_n_ctas=1/2`; physical cluster is inferred or `[4,1,1]`.
- K partitioning: `split_k=1,2,3,4,6,8,12,16,32,64,128`; backend is workspace or cluster.
- Workspace reduction geometry: `reduction_threads=32/256`, `reduction_elements_per_thread=2/4`.

### Operand tile parameters

- `output_m_tile=64/128/192`.
- `output_n_tile=16,32,48,64,80,96,112,120,128,144,160,176,192,224,240,256,352,384`.
- `operand_k_tile=16/32/64/128/192/256`.
- `operand_stages=1..13`, subject to K and shared-memory capacity.
- `batch_output_m_operands=false/true`; batching remains selected on both tiny-K routes.
- Tail padding/masking follows the required geometry; `allow_partial_tiles` remains a builder validity option, rather than a separate overlap strategy.

### Grid traversal and loop schedules

- `output_grid_size=None/128/132/144/147/148/296`; None uses launch-derived traversal.
- `output_grid_fast_axis=m/n`.
- `output_m_tiles_per_cta=1/2/4/8`; N grouping is fixed at one in this map.
- `rolled_operand_pipeline=false/true`.
- `rolled_k_unroll=1/2/4/8`, `rolled_job_unroll=1/2/4`, `peel_first_output_pair=false/true`.
- `persistent_output_pipeline=false/true` selects ordinary versus shared-ring output scheduling.
- `persistent_output_overlap=false/true`, `persistent_output_frames=1/2`; the one-frame route requires a serial rolled ring.
- `persistent_output_job_regions=false/true` rotates whole jobs across participant groups. Each group loads, computes and writes its own job.

### MMA parameters

- `cyclic_operand_pipeline=0/1/2/3` (False/True are aliases for 0/1); queue depth must leave a refill bank.
- `interleave_mma_descriptors=false/true`.
- `hoist_mma_descriptors=false/true`.
- `compact_mma_batch=false/true`.
- `reuse_mma_collector_a=false/true`; one route still needs the selected reuse alternative.
- `warp_specific_operand_waits=false/true`; one route retains this selection.
- Cluster split-K: `cluster_bank_pipeline=false/true`. Native paired phase relay and workspace paired phase relay are automatic/enabled policies.

### Writeback parameters

- `output_store=global/stmatrix`: direct register-to-global stores, or register-to-SMEM STMatrix followed by TMA.
- `output_layout=None/tcgen05_32x32b/tcgen05_16x256b/tcgen05_16x32bx2`; None selects the geometry-compatible layout.
- `tmem_writeback_register_budget=32/64/128` and `tmem_writeback_warps=4/8`.
- `output_slices=1/2/4/8`.
- `output_store_buffers=1/2/3/4/8`.
- `output_store_pending_groups=0/1/2` in the map; API supports 0..7. This queue is separate from MMA completion.
- `output_writeback_window=None/[11,15]/[36,48]`, in K-batch units.
- Workspace partial stores are fixed to global; cluster operand-storage reuse remains enabled as an implementation policy.

### Auxiliary

- LHS/RHS cache hints: default/`evict_first`/`evict_last`; output uses default/`evict_last`.
- `prefetch_tma_descriptors=false/true`.
- TMA L2 promotion: default/128B/256B in this map (64B is also supported).
- PTXAS optimization uses its default; this map does not vary optimization level.

## Removed frontend controls

`reuse_persistent_lhs`, `reuse_persistent_rhs`, `grouped_compute_overlap`, `resident_phase_order_split`, `persistent_output_schedule`, `compute_batches_per_fragment`, `pipeline_output_tiles`, `carry_output_store_queue`, `automatic_optimizations`, `block_gather_backend`, `ordered_mma_completion`, `mma_issue_group_size`, `rolled_refill_before_compute`, `preserve_prefetch_issue_order`, `hoist_invariant_operand_copies`, `reuse_completed_operand_copies`, `reuse_operand_copies_across_barriers`, `consumer_specific_mma_waits`, `paired_completion_handoffs`, `reuse_completion_locks`, `integer_index_division`, `prefetch_tmem_reads`.

The commit includes the compiler repairs already required by the measured map, plus the operand-ring, scalar-packing and rolled-index helpers needed by those repairs. Buffer-lifetime analysis now covers the associated descriptor, predicate, lock-view and completed pipeline operations. Other integration and research changes in the workspace are outside this commit.
