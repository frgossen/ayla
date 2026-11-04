# Ordering argument for retained WGMMA groups

Reference: [NVIDIA PTX ISA — asynchronous warpgroup matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions),
the `wgmma.fence`, `wgmma.commit_group` and `wgmma.wait_group` sections.

`ScheduleAsync` sinks each await toward its first consumer. One `AsyncOp` on a
WGMMA lock becomes one committed hardware group. When a full await of group A
stops at the next group B, it may move below B and wait with `pending: 1`:

```text
A; await A; independent setup; B; ...
A; independent setup; B; await A {pending: 1}; ...
```

Conditions, all derived from existing IR:

1. The lock supports partial awaits, A's lock is not B's, and the await is a
   full wait. Every matmul issued on A's lock and in B has one instruction shape
   (accumulator sizes, accumulator/operand dtypes).
2. Apart from accumulators those matmuls write, `_can_swap` must allow the
   swap: no conflicting memory effect, SSA use or ordering effect.
3. A partial await never crosses another group on its queue, so `pending`
   always counts the groups newer than A.

PTX defines `wait_group N` as completion of every group except at most the N
newest. B is the newest group at the moved await, so `wait_group 1` completes
A and everything older. WGMMAs of the same shape access a shared accumulator
in queue order, so B needs no fence against A. An ordinary accumulator access
still conflicts with the await, so the final drain stays ahead of the epilogue.

Hoisting runs before sinking. A refill of A's inputs therefore sits between the
await and B, which blocks retention, unless the refill appears only after a
later unroll. `PipelineLoop` pipelines the steady-state ring with the same
`pending: 1` form.

The sync analysis takes any await as completing its listed lock. For a
nonclustered CTA with exactly one warpgroup, that completes all of the CTA's
reads of A's shared operands, so a leader-issued refill needs no gather. With
several warpgroups or a cluster, effects stay live and `InsertSyncs` keeps the
block/cluster gathers.
