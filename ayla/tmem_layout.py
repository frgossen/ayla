"""Tensor-memory copy shapes and their register-fragment geometry."""

from dataclasses import dataclass
from collections.abc import Iterator


@dataclass(frozen=True)
class TmemWarpLayout:
    """Logical and physical row ownership of a TMEM register fragment.

    Four warps own the four physical 32-row partitions. Eight warps split
    each partition into two 16-row halves: warp w+4 owns the upper half of
    warp w's partition. M64 occupies only the first 16 rows of each partition;
    eight warps share those rows by splitting columns between w and w+4. This
    distributes one value across the CTA without warpgroup specialization.

    Paired local M64 instead owns 32 rows per partition. Partitions 0/1
    contain the first column half, and 2/3 the second half. Eight warps
    divide each partition's rows in half.
    """

    rows: int
    warps: int = 4
    cta_group: int = 1

    def __post_init__(self):
        if self.cta_group not in (1, 2):
            raise ValueError("TMEM CTA group must be one or two")
        if self.rows not in (64, 128) or self.warps not in (4, 8):
            raise ValueError("TMEM row ownership supports M64/M128 and four/eight warps")

    @property
    def paired_m64(self) -> bool:
        return self.rows == 64 and self.cta_group == 2

    @property
    def rows_per_warp(self) -> int:
        return 128 // self.warps if self.paired_m64 else self.rows // (4 if self.column_groups == 2 else self.warps)

    @property
    def row_stride(self) -> int:
        return 32 if self.paired_m64 else self.rows // 4

    @property
    def group_row_stride(self) -> int:
        return self.rows_per_warp if self.warps == 8 and (self.rows == 128 or self.paired_m64) else 0

    @property
    def column_groups(self) -> int:
        return 2 if self.rows == 64 and (self.warps == 8 or self.paired_m64) else 1

    def columns_per_warp(self, columns: int) -> int:
        if columns % self.column_groups:
            raise ValueError("columns must divide evenly between warp column groups")
        return columns // self.column_groups

    def group_column_stride(self, columns: int) -> int:
        return self.columns_per_warp(columns) if self.column_groups == 2 else 0

    def column_origin(self, warp: int, columns: int) -> int:
        if not 0 <= warp < self.warps:
            raise ValueError("warp index is outside the fragment")
        group = (warp % 4) // 2 if self.paired_m64 else warp // 4
        return group * self.group_column_stride(columns)

    def row_origin(self, warp: int, *, physical: bool = False) -> int:
        if not 0 <= warp < self.warps:
            raise ValueError("warp index is outside the fragment")
        return (warp % (4 if physical or not self.paired_m64 else 2)) * (32 if physical else self.row_stride) + (
            warp // 4
        ) * self.group_row_stride


@dataclass(frozen=True)
class TmemFragmentLayout:
    shape: str
    lanes: int
    columns: int
    registers: int
    max_repetitions: int
    split_offset: int | None = None
    row_fragment: bool = False

    def writeback_slice_columns(
        self,
        rows: int,
        columns: int,
        minimum_slices: int = 1,
        register_budget: int = 32,
        warps: int = 4,
        cta_group: int = 1,
    ) -> int:
        """Partition an accumulator within a per-thread F32 budget."""
        ownership = TmemWarpLayout(rows, warps, cta_group)
        if ownership.rows_per_warp % self.lanes:
            raise ValueError("the TMEM copy atom cannot cover this warp's rows")
        alignment = max(8, self.columns) * ownership.column_groups
        maximum = register_budget * warps * 32 // rows
        for slices in range(minimum_slices, columns + 1):
            if (
                columns % slices == 0
                and columns // slices <= maximum
                and (columns // slices) % alignment == 0
            ):
                return columns // slices
        raise ValueError(
            "TMEM writeback cannot satisfy slice alignment and register budget"
        )

    def accesses(
        self, columns: int, rows: int = 32
    ) -> Iterator[tuple[int, int, int, int, int]]:
        """Yield row, column, first register, register count, PTX repetitions.

        A warp drains 16 or 32 rows of its TMEM partition. Repetitions combine adjacent copy
        atoms only when that preserves the declared thread/register mapping.
        """
        if columns % self.columns:
            raise ValueError(
                f"{self.shape} fragments require columns divisible by {self.columns}"
            )
        if rows not in (16, 32) or rows % self.lanes:
            raise ValueError(
                f"{self.shape} copy atoms cannot cover {rows} rows per warp"
            )
        repetitions = columns // self.columns
        for tile in range(rows // self.lanes):
            repetition = 0
            while repetition < repetitions:
                count = min(
                    1 << ((repetitions - repetition).bit_length() - 1),
                    self.max_repetitions,
                )
                yield (
                    tile * self.lanes,
                    repetition * self.columns,
                    (tile * repetitions + repetition) * self.registers,
                    count * self.registers,
                    count,
                )
                repetition += count


TMEM_FRAGMENT_LAYOUTS = {
    "tcgen05_32x32b": TmemFragmentLayout("32x32b", 32, 1, 1, 128, row_fragment=True),
    "tcgen05_16x256b": TmemFragmentLayout("16x256b", 16, 8, 4, 32),
    # Two lanes share a row, each holding eight adjacent F32 elements. After
    # conversion to BF16/F16 their two 16-byte vectors fill one 32-byte sector.
    # Keep x8 atoms separate: widening x8 to x16 changes the lane split and
    # would put their stores into different sectors.
    "tcgen05_16x32bx2": TmemFragmentLayout(
        "16x32bx2", 16, 16, 8, 1, split_offset=8, row_fragment=True
    ),
}
