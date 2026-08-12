# H100 BF16 GEMM power-limit sweep (2026-08-10)

Physical GPU 3 (`GPU-b1239341-1830-45e1-822e-9142ffa94b9c`) was idle before the run. Its normal 500 W limit was temporarily changed to 350, 450, 550, and 650 W, then restored to 500 W.

The workload is a 4096x4096x4096 BF16 GEMM. All implementations use preallocated output tensors and compute `A @ B.T`. The four outputs were bitwise identical for seed 0:

- PyTorch `torch.mm(..., out=...)` using cuBLAS;
- Ayla's `k3_4tiles_split2_dbuf2` PTX kernel;
- Triton's `128x256x64`, three-stage, eight-warp TMA-persistent kernel;
- CUTLASS 4.7 CuTe DSL's `128x256` Hopper persistent kernel.

These plots are a **steady-state power-cap experiment, not a general framework ranking**. The selected CuTe DSL kernel is faster than the selected Ayla kernel at a common clock and in the normal short paired benchmark. Ayla leads the isolated low-power sweep because it reaches a different, higher DVFS equilibrium under the same board-power limit.

## Throughput

| limit | cuBLAS | Ayla | Triton | CUTLASS CuTe DSL |
|---:|---:|---:|---:|---:|
| 350 W | 360.8 | 494.6 | 359.1 | 356.5 |
| 450 W | 541.3 | 568.6 | 537.8 | 530.3 |
| 550 W | 603.9 | 621.3 | 594.7 | 592.8 |
| 650 W | 628.7 | 631.2 | 621.3 | 621.7 |

Values are TFLOP/s calculated from median CUDA-event runtime. At 650 W the four implementations converge because the workload is near its natural clock/power ceiling. Ayla draws about 554 W there, while cuBLAS, Triton, and CuTe DSL draw approximately 590, 605, and 608 W respectively.

## Power-cap-aware compute roofline

The dashed black line in the throughput and runtime plots is an empirical
compute roofline for this exact BF16 4096x4096x4096 workload.  It does not use
H100's headline boost-clock peak.  The benchmark inputs are random-normal BF16,
so their switching activity and the resulting voltage/frequency equilibrium are
already represented in each uninterrupted power-sweep measurement.

NCU reports 254,200 tensor-active cycles per SM for the useful GEMM work.  This
is exactly the expected dense-BF16 work divided over the H100's tensor pipes.
For each power cap and each implementation with an NCU report, the roofline
projects its measured throughput from its observed tensor duty cycle to 100%:

`projected roof = measured TFLOP/s / NCU tensor-active fraction`

The plotted point is the upper envelope of the Ayla, cuBLAS, and CuTe
projections.  This retains the best demonstrated random-data DVFS operating
point at that cap while removing scheduling and pipeline gaps.  cuBLAS's own
operating-point projection is also retained in `roofline.csv`; it is not used as
the universal line because at low power its greater switching/traffic forces a
lower clock than Ayla, and its projected ceiling would therefore sit below a
real measured Ayla result.

| limit | roofline | runtime floor | source operating point |
|---:|---:|---:|---|
| 350 W | 562.1 TFLOP/s | 244.5 us | Ayla, 88.0% tensor-active |
| 450 W | 646.3 TFLOP/s | 212.7 us | Ayla, 88.0% tensor-active |
| 550 W | 706.1 TFLOP/s | 194.6 us | Ayla, 88.0% tensor-active |
| 650 W | 717.4 TFLOP/s | 191.6 us | Ayla, 88.0% tensor-active |

This is still an optimistic roofline, not a thermodynamic hard bound: closing
all tensor-idle gaps could itself raise instantaneous power and make the
governor lower the clock.  Measuring that feedback requires a purpose-built
100%-duty random-data tensor burn.  The present line is nevertheless much more
representative than applying a fixed maximum clock at every power cap.

The kernel is compute-bound.  Even using cuBLAS's measured 175.4 MB of DRAM
traffic rather than the smaller compulsory byte count, its arithmetic intensity
is about 784 FLOP/byte.  H100 SXM's 3.35 TB/s memory ceiling would therefore be
about 2,625 TFLOP/s, far above every power-specific compute roofline above.

## Energy efficiency

| limit | cuBLAS | Ayla | Triton | CUTLASS CuTe DSL |
|---:|---:|---:|---:|---:|
| 350 W | 1.034 | 1.408 | 1.029 | 1.020 |
| 450 W | 1.204 | 1.267 | 1.195 | 1.180 |
| 550 W | 1.102 | 1.134 | 1.083 | 1.080 |
| 650 W | 1.066 | 1.139 | 1.026 | 1.022 |

Values are TFLOP/s per measured watt, equivalently TFLOP per joule. Ayla is the only implementation that maintains a substantially higher SM clock under the 350 W limit; the other three settle to essentially the same low-power behavior.

## Measurement method

For 450, 550, and 650 W, each implementation received a 1.5-second load ramp followed by eight seconds of CUDA-event timing in 50-launch batches. Every power level used both forward and reverse implementation order. The reported row combines those two passes, giving each implementation one early and one late position. Power, SM clock, temperature, and power-cap state were sampled through `nvidia-smi` every 0.2 seconds.

The 350 W controller state persists too long for ordering reversal to remove its history. Those four graph points therefore come from separate processes with the same approximately 50 C starting temperature, a five-second ramp, and a 20-second measurement:

| implementation | TFLOP/s | mean power | SM clock | max temperature | power-capped samples |
|---|---:|---:|---:|---:|---:|
| cuBLAS | 360.8 | 348.8 W | 742.5 MHz | 63 C | 100% |
| Ayla | 494.6 | 351.3 W | 930 MHz | 63 C | 100% |
| Triton | 359.1 | 349.1 W | 735 MHz | 62 C | 100% |
| CUTLASS CuTe DSL | 356.5 | 349.5 W | 735 MHz | 63 C | 100% |

The isolated files are under `isolated-350-four-backends/`. The shared-process forward/reverse 350 W values remain reproducible from the metadata but are intentionally not used in the main CSV or figures.

## Why the 350 W ranking differs from the normal benchmark

The earlier normal 500 W paired benchmark interleaved 20-launch batches. It measured CuTe DSL at 609.5 TFLOP/s and Ayla at 599.4 TFLOP/s, so CuTe was about 1.7% faster. A paired run with all kernels locked to 750 MHz gives the same ordering:

| implementation | TFLOP/s at fixed 750 MHz | paired delta vs cuBLAS |
|---|---:|---:|
| CUTLASS CuTe DSL | 367.1 | -0.40% (interval crosses zero) |
| cuBLAS | 365.9 | baseline |
| Triton | 361.7 | +0.90% |
| Ayla | 358.3 | +1.82% |

Negative paired delta is faster. At a common clock, Ayla is not the fastest kernel. The raw result is `fixed-750mhz-paired.json`.

An additional 350 W paired run rapidly interleaved all four kernels so they shared one power-controller history. CuTe and cuBLAS were statistically tied, while Ayla was 1.16% slower than cuBLAS. Its raw result is `power-350w-interleaved-paired.json`. In contrast, the isolated 20-second runs allow each kernel to drive the governor to its own steady state; those are the values plotted above.

### Concrete NCU and power evidence

NCU reports at 350 W show that Ayla does not complete the tensor work in fewer cycles. All three kernels spend almost exactly 254,200 absolute cycles with the tensor pipe active. CuTe and cuBLAS finish in about 283-284k SM cycles, while Ayla needs about 289k cycles because its tensor work occupies a slightly smaller fraction of elapsed cycles.

| counter | Ayla | cuBLAS | CuTe DSL |
|---|---:|---:|---:|
| total SM cycles | 288,913 | 284,030 | 283,350 |
| tensor-pipe active | 87.99% | 89.50% | 89.71% |
| total executed instructions | 13.46M | 16.58M | 15.71M |
| instructions/cycle/SM partition | 0.088 | 0.111 | 0.105 |
| DRAM bytes | 106.7 MB | 175.4 MB | 203.0 MB |
| registers/thread | 154 | 168 | 168 |
| threads/CTA | 256 | 384 | 384 |

NCU pauses and replays kernels, so its measured clock and duration are not representative of an uninterrupted 350 W load. The cycle counts and activity/resource counters remain useful. Full reports and a compact CSV are under `ncu-350w/`.

The causal check is board power at an identical locked 1050 MHz. During separate continuous runs, Ayla averaged 380.8 W, cuBLAS 406.7 W, and CuTe DSL 429.8 W. Thus Ayla uses about 11.4% less board power than CuTe at the same clock, despite being slightly worse per cycle. Under a 350 W cap, the governor compensates by holding Ayla near 930 MHz while cuBLAS and CuTe settle near 735-743 MHz. The nonlinear voltage/frequency response magnifies that power difference into the large steady-state throughput gap.

The lower power is consistent with Ayla issuing 16-19% fewer instructions and moving less measured DRAM data. Its lower instruction-issue density also means fewer simultaneously switching units. That is an energy/DVFS advantage under this workload, not evidence that the Ayla framework generates an intrinsically faster GEMM.

See `metadata.json` for exact source configurations and environment details, and `power_efficiency.csv` for unrounded results. Regenerate the figures from the repository root with:

```bash
.pixi/envs/default/bin/python tools/plot_power_efficiency.py \
  docs/profiling/power-sweep-h100-gpu3-2026-08-10/power_efficiency.csv
```

The plotting script automatically reads `ncu-350w/summary.csv` and writes the
derived values to `roofline.csv` alongside the figures.

![Power efficiency summary](power_efficiency_summary.png)

![TFLOP/s comparison](tflops_barplot.png)

The equal-clock control shows the intrinsic runtime ordering without the power governor selecting a different clock for each kernel:

![Equal-clock runtime comparison](equal_clock_runtime_750mhz.png)
