# SASS review of the K=3/K=4 two-tile kernels (2026-08-05)

Static SASS analysis of the current top kernels. **No timing was collected** — an
8-GPU `sglang` server was pinning every device at 100% for the whole session, so
every A/B run was unusable (see "Benchmarking blocked" below). Everything here is
either a static instruction count or a bitwise-correctness result; nothing is a
measured speedup.

Toolchain: the pixi env has no `ptxas`/`nvdisasm`. Use CUDA **13.0** — the
generated PTX is `.version 9.0`, which 12.8's ptxas rejects:

```
export PATH=/usr/local/cuda-13.0/bin:$PATH
ptxas -arch=sm_90a -O3 --verbose k.ptx -o k.cubin && nvdisasm -c k.cubin > k.sass
```

Helper scripts live in `tools/ptx_experiments/`.

## Anatomy of one k64 stage

`auto_matmul_..._k3_2tiles` main K loop: **177 SASS instructions, 12 HGMMA**
(14.8 instructions per HGMMA). Per stage, in issue order:

1. `SHF` + `SYNCS.PHASECHK.TRANS64.TRYWAIT` + `BRA` — mbarrier spin
2. 8 uniform ops rebuilding the two 64-bit GMMA descriptors
   (`USHF.R.U32.HI` >>4, `ULOP3` |0x10000, `UMOV` 0x40000040 high word)
3. 6 × `UIADD3.64` deriving the +2/+4/+6 (k16-step) descriptor quads
4. 4 × `HGMMA.64x256x16.F32.BF16`, with 3 `NOP`s interleaved
5. `WARPGROUP.DEPBAR.LE gsb0, 0x0` — the drain
6. `BAR.SYNC`/`BAR.ARV` split barrier
7. `@P1 BRA` over a ~22-instruction thread-0 TMA block
8. `SYNCS.ARRIVE.TRANS64` + `BSYNC`

Steps 1–3 sit between the drain and the next HGMMA, so they are dead tensor time.

## Finding 1 — hoisting the descriptors out of the loop is counterproductive

`lower_to_ptx_template.py:604` says descriptors are "deliberately rematerialized
near each WGMMA group: keeping them live across the loop raises register
pressure." The descriptors *are* provably loop-invariant — they derive only from
fixed multi-buffer SMEM slot addresses (`%r103`, `%r112`, each defined once in the
prologue) — so this looked like an easy 36-instruction-per-iteration win.

It is not. `tools/ptx_experiments/hoist_desc.py` performs the hoist at the PTX
level (36 of 209 loop-body lines per K loop, invariance-checked). Result:

| kernel | loop insns | registers | `R2UR` in loop |
|---|---|---|---|
| k3_2t baseline | 177 | 154 | 0 |
| k3_2t hoisted | 184 | 176 | 32 |
| k4_split2 baseline | 234 | 154 | 0 |
| k4_split2 hoisted | 274 | 201 | 68 |

Rematerializing lets ptxas compute the descriptor **directly into uniform
registers**, which HGMMA's `gdesc[URn]` operand requires. Hoisting makes the value
live across the backedge, ptxas demotes it to the vector register file, and then
has to insert an `R2UR` before every HGMMA to get it back. The comment in the
source is right, and this refines the earlier "`R2UR` is 8-per-warp-per-stage
bookkeeping" note: `R2UR` is what you *get* if you try to fix this, not what is
there now. **Do not retry this.**

## Finding 2 — `elect.sync` removes the TMA serialization loops (ready to test)

The TMA issue is guarded by `setp.ne.s32 %p, %tid.x, 0; @%p bra skip`. ptxas
cannot prove that leaves at most one lane active, so it wraps *each* `UTMALDG` in
a serialization loop:

```
.L_x_15:  @P0 ELECT P3, URZ, PT
          UTMALDG.2D [UR16], [UR8]
          @P3 PLOP3.LUT P0, PT, P3, PT, PT, 0x8, 0x0
          PLOP3.LUT P3, PT, PT, PT, PT, 0x8, 0x0
          @P0 BRA.U.ANY `(.L_x_15)
```

Two of these per stage, six per loop iteration. Replacing the guard with an
explicit warp-0 test plus `elect.sync` states the single-lane property directly:

```ptx
setp.gt.u32 %p, %tid.x, 31;    // was: setp.ne.s32 %p, %tid.x, 0
@%p bra skip;
elect.sync _|%p, -1;
@!%p bra skip;
```

ptxas then emits one `ELECT` + `@!P2 BRA` and both `UTMALDG`s straight-line — it
also *merges* the two previously separate guarded regions into one:

| | loop insns | `PLOP3` | `BRA` | registers | spills |
|---|---|---|---|---|---|
| k3_2t baseline | 177 | 19 | 13 | 154 | 0 |
| k3_2t elect.sync | **165** | **0** | **10** | 154 | 0 |

Correctness, on GPU: output is **bitwise identical** to baseline for all three
configs (k3_2t, k3_split2, k4_split2), and `maxerr` vs `(a.float() @
b.float().t()).bfloat16()` is 1.0 for both, i.e. ordinary bf16 rounding.
Legality also checked statically — `tools/ptx_experiments/taint.py` confirms no
`cp.async.bulk.tensor` or `mbarrier.arrive.expect_tx` operand is tainted by
`%tid`/`%laneid`, so any lane of warp 0 computes the same address, and the
`mbarrier.init` arrive-count of 1 is still matched by exactly one arrival.

**This is unmeasured.** 12 fewer instructions per iteration at identical register
count is a static improvement, not a proven one — the loop is behind a
`wgmma.wait_group 0` drain that is 64% of stall samples, so instructions in its
shadow may be free. Run `tools/ptx_experiments/ab_elect.py` on an idle GPU before
deciding. If it wins, the compiler-side change is in
`_thread_zero_guard` (`lower_to_ptx_template.py:140`), which would need to emit
the warp-0 + `elect.sync` form for TMA-load guards specifically.

## Benchmarking blocked

Every timing number this session was garbage. `nvidia-smi` showed
`sglang::scheduler_TP0..TP7` holding ~50 GB on each of the 8 GPUs at a sustained
100% utilization and ~490 W against a 500 W cap. Same-kernel medians moved
230→486 µs run to run, and stdev reached 118 µs. Check
`nvidia-smi --query-compute-apps` **before** any sweep — the clock-locking
protocol in the existing notes assumes an otherwise-idle device and does nothing
about a co-tenant.
