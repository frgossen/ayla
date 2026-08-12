# H100 GEMM next optimizations

## Summary

Baseline: the BF16 4096³ N=256, K=3 kernel with two output tiles per CTA,
TMA output stores, and `stmatrix` register-to-shared staging. All experiments
retain symmetric compute warp groups; warp specialization is out of scope.

Implement and evaluate these optimizations in order, always branching from the
fastest retained kernel:

1. Four output tiles per CTA / 128-CTA launch.
2. Fold accumulator zeroing into the first WGMMA.
3. Two-CTA vertical clusters with B-tile TMA multicast.

An optimization is retained only if it passes correctness and improves median
runtime by at least 1% across three order-reversed A/B runs. Failed variants
are reverted after their NCU reports are saved under `tools/`.

## 1. Four output tiles per CTA

- Generalize `output_tiles_per_cta` from `{1,2}` to `{1,2,4}`.
- Derive `grid_x = 32 / output_tiles_per_cta`, the outer M span as
  `grid_x * 128`, and the output-loop unroll factor from the parameter.
- Register a temporary four-tile variant with `grid=[8,16,1]` (128 CTAs).
- Preserve the three-stage K pipeline, N=256 WGMMA, `stmatrix` staging, and
  asynchronous TMA-store overlap.
- Compare L2 hit rate, tensor utilization, waves/SM, exposed store time, and
  runtime against the two-tile kernel.
- If it wins, use it as the base for later stages; otherwise retain two tiles.

## 2. Eliminate accumulator clearing

- Add a static `accumulate`/`scale_d` property to `MatmulOp`, defaulting true.
- Peel the first main-K iteration for each output accumulator and mark only its
  first K=16 WGMMA as non-accumulating.
- Promote the destination to uninitialized `wrmem` only when that overwrite
  dominates every use; keep explicit initialization as the safe fallback.
- Lower the first WGMMA with `scale-d=0` and all later WGMMAs with `scale-d=1`.
- Confirm that the 256 PTX zero moves and SASS `CS2R ..., SRZ` sequence vanish.

## 3. Vertical cluster and B multicast

- Add `cluster_dim`, default `(1,1,1)`, to `KernelOp` and `PtxKernel`, and
  propagate it through PTX emission and runtime launch metadata.
- Launch clustered kernels through `cuLaunchKernelEx`; require grid dimensions
  to be divisible by cluster dimensions.
- Use `cluster_dim=(2,1,1)` so adjacent M-axis CTAs share B while loading their
  own A tiles.
- Add cluster-scoped TMA attributes: both CTA leaders load A, cluster rank 0
  multicasts B with mask `0x3`, and each CTA expects its full local A+B bytes.
- Synchronize the cluster once after local mbarrier initialization and use
  cluster acquire semantics for multicast completion.
- Keep output stores CTA-local. Test cluster size two only.

## Validation

- Add IR/PTX tests for four-tile structure, `scale-d`, cluster metadata, and
  exactly one RHS multicast per cluster load step.
- Run the existing three-seed GPU correctness suite for every variant, plus
  repeated cluster launches to exercise reusable barrier parity.
- Benchmark same-process A/B in both orders, repeated three times; report
  median, minimum, standard deviation, TFLOP/s, and percentage of `torch.mm`.
- Run full NCU on every retained winner and check spills, occupancy, TMA waits,
  L2 behavior, tensor utilization, and cluster/multicast instruction emission.

## Assumptions

- The current N=256 two-tile `stmatrix` kernel is the baseline.
- Later stages apply to the fastest variant retained from earlier stages.
- Warp specialization and cluster sizes larger than two are out of scope.
