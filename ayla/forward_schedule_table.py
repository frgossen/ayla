"""Shape-specific schedules for scalar phases of resident forward graphs."""

# Static wide normalization: threads cooperating on one row. Remaining
# threads process other rows concurrently. Qualified independently of GEMMs.
STATIC_NORMALIZATION_ROW_THREADS: dict[tuple[int, int, int, int], int] = {}

# Static source GEMMs may use geometry tuned for consumer readiness. Keys:
# (resident blocks, threads, M, N, K). Defaults require complete-graph timing.
STATIC_GEMM_LAYOUT_OVERRIDES: dict[tuple[int, int, int, int, int], dict[str, object]] = {}

# First read-only matrix: columns per thread carried from the next row.
# Keys: (resident blocks, threads, rows, width, occurrence of this row shape).
# One wide region at a time avoids the combined schedule's register spills.
# The forward scheduler restricts these choices to BF16 double reductions.
PERSISTENT_ROW_PREFETCH_SCHEDULES: dict[tuple[int, int, int, int, int], int] = {
    (132, 256, 2048, 6656, 1): 26,
}

# Threads per head row, with the original 32-lane reduction order preserved.
# Keys: (resident blocks, threads, flattened head rows, head width).
# Qualified on the 2048-token H100 body; the 512-token result was inconclusive.
PERSISTENT_WARP_ROW_THREADS: dict[tuple[int, int, int, int], int] = {
    (132, 256, 4096, 128): 16,
    (132, 256, 65536, 128): 16,
}
