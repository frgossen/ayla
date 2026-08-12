# Ayla asymmetric warpgroup ping-pong experiment

This experiment changed only generated PTX. The Ayla framework and retained
baseline kernel were not modified.

## Schedule

The baseline makes warp 0 wait at each operand-slot release while the other
seven warps arrive, so both compute warpgroups rendezvous before thread 0
issues the TMA refill.

The final prototype reuses the same named barrier asymmetrically:

```text
WG0 (128 threads): bar.arrive, then advance to the next prefetched stage
WG1 (128 threads): bar.sync, then tid 128 issues the released slot's TMA refill
```

Thus WG0 can lead WG1 without a dedicated producer warpgroup, extra operand
stages, or longer accumulator lifetimes. The barrier still proves that both
warpgroups have completed their WGMMA reads before TMA overwrites the slot.

An earlier prototype used three extra shared mbarriers, with WG0's leader
signalling and WG1's leader polling. It was correct but added substantially
more SASS and was slower. The named-barrier form is the minimal implementation.

## Correctness and SASS

- Bitwise identical to baseline for random BF16 4096x4096 inputs.
- 154 registers/thread and zero spills, unchanged.
- 221,336 bytes shared memory, unchanged.
- Six named barrier resources, unchanged.
- Static SASS: 2,424 baseline versus 2,432 ping-pong.
- The 25 operand-release sites become WG0 `BAR.ARV` / WG1 `BAR.SYNC` sites.
- TMA ownership moves from tid 0 to tid 128.

## Timing

The controlled result used physical H100 GPU 3 at a locked 1350 MHz, 5,000
ramp launches, and 200 paired/interleaved timing batches of 20 launches each:

| kernel | median runtime | throughput |
|---|---:|---:|
| baseline | 209.042 us | 657.5 TFLOP/s |
| named-barrier ping-pong | 208.978 us | 657.7 TFLOP/s |

Paired median change was -0.023%, with a 95% bootstrap interval from -0.073%
to +0.031%. Negative is faster. This is neutral.

At a free clock after 30,000 alternating ramp launches, the paired result was
+0.075% slower with a 95% interval of +0.022% to +0.121%. It did not provide a
power-regime improvement.

NCU showed an apparent rise in tensor activity under profiling, but ordinary
locked-clock event timing did not reproduce the corresponding cycle reduction.
The NCU result is therefore treated as profiler perturbation rather than a
speedup.

## Conclusion

The asymmetric handoff is legal, inexpensive, and does create the intended
one-way schedule, but it does not improve steady-state runtime. The existing
CTA-wide rendezvous is not the source of a large exposed bubble in this kernel.
This closes the proposed symmetric-compute ping-pong opportunity without
introducing the register costs of dedicated producer warp specialization.

The disposable transforms and benchmark are:

- `tools/ptx_experiments/pingpong_named_barrier.py`
- `tools/ptx_experiments/pingpong_warpgroups.py`
- `tools/ptx_experiments/ab_pingpong.py`
