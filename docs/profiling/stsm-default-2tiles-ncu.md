# NCU: STSM two-tile default

Profiled kernel: `auto_matmul_4096_wgmma256_wg_tma_store_k3_2tiles`
on H100, BF16 4096x4096x4096. The full report is
`stsm_default_2tiles_full.ncu-rep` in this directory.

## Headline metrics

| Metric | Value |
|---|---:|
| NCU duration | 245.28 us |
| Tensor/SM throughput | 84.43% |
| Tensor utilization while SM active | 88.87% |
| DRAM throughput | 26.50% |
| L2 throughput | 42.48% |
| L2 hit rate | 74.82% |
| Registers/thread | 167 |
| Static shared memory/CTA | 213.14 KiB |
| Achieved occupancy | 12.33% (one 8-warp CTA/SM) |
| Executed instructions | 17.341 M |
| Local-memory spills | 0 |
| Scheduler cycles with no eligible warp | 88.39% |

The low issue-slot percentage (10.91%) is not evidence that ordinary scalar
instructions are the primary bottleneck: WGMMA is a long-latency asynchronous
operation and keeps the tensor pipe active without continuously consuming issue
slots. Tensor utilization is the more useful compute metric here.

## Stall breakdown

NCU reports 17.19 warp cycles per issued instruction. Dividing each stall value
by that total gives the approximate share:

| Stall | Cycles/issued instruction | Approx. share |
|---|---:|---:|
| Barrier | 10.93 | 63.6% |
| Wait | 1.66 | 9.7% |
| Long scoreboard | 1.28 | 7.5% |
| Selected | 1.00 | 5.8% |
| GMMA | 0.78 | 4.6% |
| Short scoreboard | 0.59 | 3.4% |
| MIO throttle | 0.31 | 1.8% |

The generated PTX contains block-to-leader synchronization before each TMA
reload of a reused shared-memory stage. It currently lowers to a named
`bar.sync` for warp 0 and `bar.arrive` for all other warps. This ordering is
real: both symmetric compute warp groups must finish WGMMA reads before thread
0 overwrites their shared B stage. The ordering cannot simply be deleted.

The lowering also emits four independent `mbarrier.init` sequences at kernel
startup, each followed by a full `bar.sync 0`, plus the output-staging barrier.
Those fixed barriers are candidates for batching, but the repeated
block-to-leader barriers are the larger target.

## Recommended experiments

1. **Lower block-to-leader barriers with Hopper mbarriers.** Keep both warp
   groups symmetric compute consumers. After `wgmma.wait_group`, elect one lane
   per warp group to arrive on a reusable mbarrier initialized with count 2;
   thread 0 performs the parity wait before issuing the TMA reload. This changes
   only the lowering of the existing `BarrierOp`, uses two synchronization
   tokens instead of all 256 threads, and does not introduce warp
   specialization. Preserve release/acquire ordering and validate reusable
   phases across loop iterations.

2. **Use `scale-d=0` on the first WGMMA for each accumulator.** The PTX has 256
   static `mov.f32` zeroing instructions. Across 256 CTAs and eight warps, these
   account for approximately 524,288 executed warp instructions, about 3.0% of
   the reported instruction count. This is the lowest-risk item already present
   in the optimization plan.

3. **Sweep WGMMA commit-group depth instead of always waiting for zero.** The
   three shared-memory stages permit testing `wgmma.wait_group 1` and `2`, with
   a final `wait_group 0` before output use. The compiler must associate a wait
   with the oldest stage before reuse; blindly changing every immediate wait is
   incorrect. This targets the 9.7% wait and 4.6% GMMA stall components without
   assigning producer/consumer roles to different warps.

4. **Then test four output tiles per CTA.** This remains a reasonable plan item
   because it amortizes lock initialization and other CTA-fixed work. Its
   128-CTA launch leaves four of 132 SMs unused (a hard ~3% underfill), so it
   should be evaluated after fixed overhead is reduced and retained only on a
   clear measured win.

5. **Lower-priority: output shared-layout swizzle.** NCU reports 1.835 M
   excessive shared-memory wavefronts over 21.234 M ideal wavefronts (8.6%),
   associated with shared stores. A swizzled STSM-to-TMA output layout could
   reduce this, but MIO throttle is only 1.8%, so expected upside is small.

## Metrics that are not direct optimization targets

- **Occupancy:** one CTA/SM is forced independently by both 167 registers/thread
  and 213 KiB shared memory. Reaching two CTAs would require at most 128
  registers/thread *and* at most roughly 116 KiB shared memory/CTA. Chasing the
  12.5% occupancy number would require changing the kernel shape and pipeline,
  not a local tuning fix.
- **DRAM bandwidth:** only 26.5% utilized, with a 74.8% L2 hit rate. The kernel
  is not DRAM-bound. The completed cache-policy and TMA L2-promotion sweeps are
  consistent with this.
- **All barrier stalls:** the shared B stage creates a genuine cross-warp-group
  lifetime dependency. Faster/minimal synchronization is actionable; removing
  synchronization is not.
- **Peak-spec gap:** NCU measured a 1.22 GHz SM clock while the tensor pipe was
  active 84.4% of elapsed cycles. Power/clock limits are system properties, not
  compiler fixes.

## Relationship to the existing plan

Accumulator clearing remains well supported by the profile. Four tiles/CTA is
still credible but is not the strongest first experiment. Cluster multicast is
deprioritized: prior benchmarks were slower, and the new report shows ample
DRAM headroom plus high L2 reuse while synchronization is the dominant stall.
The mbarrier-backed block-to-leader lowering should be inserted before the
four-tile and multicast work.
