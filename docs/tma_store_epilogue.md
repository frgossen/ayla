# TMA store epilogue

The first change added a TMA-backed writeback option to the existing
buffer-promotion flow. Its single-tile form performs:

1. convert the distributed `f32` WGMMA accumulator to the destination type;
2. move the distributed WRMEM value into a block-wide SMEM tile;
3. issue one asynchronous TMA SMEM-to-GMEM move;
4. wait for that bulk group before the kernel returns.

The option uses the existing `tma=True` promotion policy. For input operands,
that policy already means GMEM-to-SMEM TMA. For a WRMEM-promoted destination,
TMA cannot read registers directly, so writeback gains the necessary SMEM
staging move.

## IR shape

After destination promotion, ordinary direct writeback looks like:

```text
%acc_f32 = alloc : wrmem<64x256xf32>
matmul ..., %acc_f32
%acc_bf16 = alloc : wrmem<64x256xbf16>
convert %acc_bf16, %acc_f32
move %c_wg, %acc_bf16
```

With `tma=True`, only the writeback changes:

```text
%acc_f32 = alloc : wrmem<64x256xf32>
matmul ..., %acc_f32
%acc_bf16 = alloc : wrmem<64x256xbf16>
convert %acc_bf16, %acc_f32
%c_smem = alloc : smem<128x256xbf16>
move %c_smem, %acc_bf16
move %c_block, %c_smem {tma = true}
```

The first move is an ordinary layout-aware move. Two warp groups each own a
64x256 WRMEM fragment and write disjoint rows of the 128x256 SMEM tile. The
second move follows the existing TMA path:

```text
move {tma = true}
  -> tma_move
  -> async<bulk> { tma_move }; await<bulk>
  -> cp.async.bulk.tensor.2d; commit_group; wait_group
```

WRMEM stores to SMEM and GMEM share one lowering. Both use the same WGMMA
register-layout offsets; only address width, PTX memory space, and the optional
warp-group row offset differ.

## Current experiment

The 4096x4096 benchmark uses three input stages because the block-wide output
tile occupies 64 KiB of SMEM. Four input stages plus this tile exceed Hopper's
per-block shared-memory capacity. Stage count is a kernel configuration, not
part of the generic TMA writeback transform.

## Two-tile pipeline

The pipelined experiment reduces the physical grid from 32x16 to 16x16 and
assigns two logical M tiles to each CTA. Hierarchical tiling expresses the
mapping without a new scheduling op:

```text
for outm = 0 to 2:
  tile_m = outm * 16 + block.x
  compute output[tile_m, block.y]
  stage output in SMEM
  issue TMA store
  await TMA store
```

LICM first moves reusable uninitialized buffers out of this loop. Fully
unrolling the two iterations then exposes both bulk operations in one IR block.
The existing async scheduler sinks the first await across independent work but
stops before the shared output buffer is reused:

```text
compute tile 0
WRMEM -> output SMEM
issue TMA store 0

compute tile 1

await TMA store 0
WRMEM -> output SMEM
issue TMA store 1
await TMA store 1
```

This is real cross-tile store pipelining: tile 0's SMEM-to-GMEM transfer is in
flight while tile 1 computes. One output buffer remains sufficient because its
previous TMA read is awaited immediately before the buffer is overwritten.
