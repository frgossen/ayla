"""Tensor-memory copy shapes and their register-fragment geometry."""

from dataclasses import dataclass
from collections.abc import Iterator


@dataclass(frozen=True)
class TmemFragmentLayout:
    shape: str
    lanes: int
    columns: int
    registers: int
    max_repetitions: int
    split_offset: int | None = None
    row_fragment: bool = False

    def accesses(self, columns: int) -> Iterator[tuple[int, int, int, int, int]]:
        """Yield row, column, first register, register count, PTX repetitions.

        A warp drains its own 32 TMEM rows. Repetitions combine adjacent copy
        atoms only when that preserves the declared thread/register mapping.
        """
        if columns % self.columns:
            raise ValueError(
                f"{self.shape} fragments require columns divisible by {self.columns}"
            )
        repetitions = columns // self.columns
        for tile in range(32 // self.lanes):
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
