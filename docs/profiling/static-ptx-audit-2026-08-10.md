# Static PTX/SASS audit of the K=3 four-tile kernel (2026-08-10)

Target: `auto_matmul_4096_wgmma256_wg_tma_store_k3_4tiles_split2_dbuf2`
on an H100, assembled with CUDA 13.0 `ptxas -arch=sm_90a -O3`.  All variants
use 221336 bytes of shared memory and six hardware barriers.

## Retained changes

| variant | SASS instructions | registers | spills |
|---|---:|---:|---:|
| recompute every STMatrix offset, ordinary TMA guard | 2608 | 154 | 0 |
| reuse the per-thread STMatrix offset | 2584 | 154 | 0 |
| reuse offset + explicit lane-0 TMA election | **2504** | **154** | **0** |
| above + timeout-free mbarrier waits | **2448** | **154** | **0** |
| above + shared `%tid.x` SSA read | **2440** | **154** | **0** |
| above + in-place WGMMA descriptor recurrence | **2424** | **154** | **0** |

The eight STMatrix groups all use the same lane/warp-to-row offset.  Keeping
the first calculation as an SSA result and adding it to each output arena
removes 24 final instructions.  This is a real SASS reduction, not merely
cleaner PTX.

TMA loads were guarded by `tid.x != 0`.  Although exactly one thread reached
the load, ptxas wrapped each `UTMALDG` in an `ELECT`/`BRA.U.ANY` serialization
loop.  After the existing thread-0 branch, the compiler now emits:

```ptx
elect.sync _|%p, 1;
@!%p bra done;
```

The member mask contains only lane 0, so this retains exact thread-0 semantics
even if a future TMA coordinate is not warp-uniform.  It removes another 80
static SASS instructions.  ptxas does introduce 42 additional `R2UR`s, but the
net instruction count still falls by 80 and register allocation is unchanged.

The combined result is 104 fewer instructions (4.0%) with the same resource
usage.

Omitting the optional `timeHint` from each `mbarrier.try_wait` removes the
out-of-line `NANOSLEEP.SYNCS`/phase-check slow paths generated for 28 static
wait sites.  The explicit software retry loop remains, so incomplete barriers
are still polled.  This removes 56 more SASS instructions.  Replacing 75 raw
template reads of stable `%tid.x` with one kernel-scoped SSA value removes 74
PTX reads and eight additional SASS instructions.  Together with the earlier
changes, the kernel is 168 instructions (6.4%) smaller than the
2608-instruction recompute baseline.  Finally, the four consecutive WGMMA
tiles now advance the two base descriptors in place by `+2` between matrix
instructions.  The six offsets are real and remain in SASS, but this avoids
six one-use PTX descriptor names per group and gives ptxas a better local
schedule.  It removes another 16 SASS instructions without changing register,
barrier, shared-memory, or spill usage: 184 instructions (7.1%) total versus
the recompute baseline.

## Timing and correctness

All variants were loaded in one process and timed in alternating 20-launch
bursts after 3000 paired warmups.  On the machine's current 500 W power limit:

| variant | median | paired result |
|---|---:|---:|
| original | 222.718 us | baseline |
| combined | 222.772 us | -0.057% |

The central 90% paired range was -1.337% to +0.593%, so the change is
performance-neutral at this noise floor.  A separate framework benchmark gave
0.223 ms / 616.7 TFLOP/s, 98.4% of matched `torch.mm(A, B.T)`.

Outputs were bitwise identical between PTX variants.  The integrated kernel
also passed the three correctness seeds (42, 43, 44), with maximum BF16 error
1 versus the FP32-accumulated reference.

After moving immediate cleanup into IR lowering/DCE, a fresh framework run
measured 0.222 ms / 620.3 TFLOP/s, 99.3% of matched `torch.mm(A, B.T)`.
The post-refactor cubin disassembles identically to the previously benchmarked
integrated kernel.

Alternating A/B timing found the timeout-free wait neutral: 223.179 us for the
2504-instruction form versus 223.189 us for the 2448-instruction form (paired
-0.133%, within noise).  Sharing `%tid.x` was also neutral: 220.874 us versus
221.179 us (paired -0.039%).  Both variants were bitwise identical to the
baseline and retained maximum BF16 reference error 1.
A final framework run of the integrated 2440-instruction kernel measured
0.220 ms / 625.6 TFLOP/s, 100.7% of matched `torch.mm(A, B.T)`.

The descriptor-recurrence cubin was bitwise correct against the
2440-instruction form and the FP32-accumulated reference (maximum BF16 error
1).  Alternating 20-launch bursts measured 222.628 us for the old form and
223.090 us for the 2424-instruction form; the paired change was -0.053%, with
a central 90% range of -1.304% to +1.165%.  It is performance-neutral at the
measurement noise floor while being statically smaller.

## Candidates rejected or already handled by ptxas

- The 41 constants used only as PTX immediates are now omitted from the
  lowered template operands.  This makes their definitions genuinely dead in
  SSA, so the ordinary IR DCE pass removes them before PTX register naming.
  It shortens PTX by 41 lines while producing the exact same
  2504-instruction SASS; there is no post-render dead-move cleanup.
- Hoisting the WGMMA descriptor calculations from the four K loops removes 144
  PTX instructions but makes SASS much worse: 2584 to 3200 instructions and
  154 to 180 registers, including 480 `R2UR`s.  The descriptors should remain
  rematerialized near each WGMMA group.
- Repeated `%tid.x` loads, block-leader predicates, descriptor constants, and
  most address expressions visible in the readable PTX are already folded or
  moved to uniform registers by ptxas.  Sharing `%tid.x` in SSA still saves
  eight SASS instructions; packing each WGMMA descriptor with `mov.b64` instead
  of `cvt.u64.u32` + `or.b64` leaves the SASS instruction count unchanged.
- Keeping only four explicit `wgmma.fence` sites looks legal at PTX level, but
  ptxas injects 112 `WARPGROUP.ARRIVE` instructions, warns that the injected
  operations serialize WGMMA on divergent paths, and grows SASS from 2424 to
  2592.  One explicit fence per four-MMA commit group is retained.
- Splitting the WGMMA descriptor into a 32-bit dynamic low half and static high
  half removes carry arithmetic, but packing it raises `R2UR` from 41 to 76 and
  grows SASS from 2424 to 2432.  The 64-bit recurrence is smaller.
- Replacing `mbarrier.try_wait` with nonblocking `mbarrier.test_wait` reduces
  SASS further to 2368 instructions, but active polling regresses paired timing
  by about 0.66%, so it was rejected.
- PTX ISA 9.3 corresponds to CUDA 13.3 and driver r610.  This machine currently
  has CUDA 13.0 (`ptxas` supports PTX 9.0) and driver r580.  Of the 9.3
  additions, only `.pragma "mma_throughput"` is relevant to this kernel; the
  new mbarrier phase-reporting/layout forms, carryless `clmad`, multimemory,
  and fabric operations do not replace any current operation.  CUDA 13.0 warns
  that `mma_throughput` is unsupported, so the PTX version remains 9.0 until a
  CUDA 13.3/r610 environment can verify its actual Hopper SASS and timing.
