"""CTA rank relations shared by tile mapping, movement and completion lowering."""

from dataclasses import dataclass
import math


def cluster_size(kernel):
    return math.prod(d or 1 for d in kernel.cluster_dim)


def pair_mask(rank):
    return 3 << (rank & ~1)


def validate_masks(masks, size, *, groups=False):
    if not isinstance(masks, (list, tuple)) or len(masks) != size:
        raise ValueError("CTA masks must contain one mask per cluster rank")
    if any(
        type(m) is not int or m <= 0 or m >> size or not m & (1 << r)
        for r, m in enumerate(masks)
    ):
        raise ValueError("CTA masks must include self and stay within the cluster")
    if groups and any(
        masks[q] != m for m in masks for q in range(size) if m & (1 << q)
    ):
        raise ValueError("multicast masks must describe disjoint recipient groups")
    return tuple(masks)


def multicast_sources(masks, ordinal):
    """Issuing rank of each recipient's copy of the ``ordinal``-th transfer to ``masks``.

    Transfers to the same recipient groups rotate through the group's members
    in rank order, so slices of one tile, and equal-group operands, spread
    their issue across the group.
    """
    members = [[r for r in range(len(masks)) if mask >> r & 1] for mask in masks]
    return [group[ordinal % len(group)] for group in members]


def arrival_counts(masks, group=1):
    return tuple(
        sum(
            bool(masks[source] & (1 << target))
            for source in range(0, len(masks), group)
        )
        for target in range(len(masks))
    )


@dataclass(frozen=True)
class ClusterLayout:
    m: int
    n: int
    physical: tuple[int, int, int]
    mma_group: int = 1

    def __post_init__(self):
        if (
            type(self.m) is not int
            or type(self.n) is not int
            or min(self.m, self.n) < 1
        ):
            raise ValueError("logical cluster dimensions must be positive integers")
        if len(self.physical) != 3 or any(
            type(d) is not int or d < 1 for d in self.physical
        ):
            raise ValueError(
                "physical cluster dimensions must be three positive integers"
            )
        if self.m * self.n != math.prod(self.physical) or self.m * self.n > 16:
            raise ValueError(
                "logical and physical cluster sizes must match and fit 16 CTA ranks"
            )
        if self.mma_group not in (1, 2) or self.m % self.mma_group:
            raise ValueError(
                "paired GEMM mapping requires complete adjacent M-rank pairs"
            )

    @property
    def size(self):
        return self.m * self.n

    def operand_masks(self, operand):
        if operand == "lhs":
            return tuple(
                sum(1 << (r % self.m + self.m * y) for y in range(self.n))
                for r in range(self.size)
            )
        if operand == "rhs":
            return tuple(
                sum(
                    1 << (x + self.m * (r // self.m))
                    for x in range(r % self.m % self.mma_group, self.m, self.mma_group)
                )
                for r in range(self.size)
            )
        raise ValueError("cluster operands must be lhs or rhs")

    def bank_release_masks(self, lhs, rhs):
        return tuple(
            sum(
                1 << q
                for q in range(self.size)
                if any(
                    (lhs[s] | rhs[s]) & (1 << q)
                    for s in range(
                        r // self.mma_group * self.mma_group,
                        r // self.mma_group * self.mma_group + self.mma_group,
                    )
                )
            )
            for r in range(self.size)
        )


def rank_table_ptx(name, table, *, bits=32, rank="cluster_rank"):
    """Small uniform rank tables; constant and shifted group masks stay compact."""
    table = tuple(table)
    dest = f"${{b{bits}:{name}}}"
    source = f"${{b32:{rank}}}"
    if len(set(table)) == 1:
        return f"mov.b{bits} {dest}, {table[0]};\n"
    # Common row/column masks are shifts of the first mask by a periodic or
    # grouped rank coordinate. Keep these tables out of the stage's branch path.
    for period in range(1, len(table) + 1):
        for divide in [False, True]:
            shifts = [
                r // period * period if divide else r % period
                for r in range(len(table))
            ]
            if all(table[r] == table[0] << shifts[r] for r in range(len(table))):
                # ptxas lowers rem/div.u32 by an immediate to a floating
                # reciprocal chain on the issue path; ranks are unsigned,
                # so power-of-two periods reduce exactly to masks.
                pow2 = period & (period - 1) == 0
                if not divide:
                    expr = (
                        f"and.b32 ${{b32:{name}_shift}}, {source}, {period - 1};\n"
                        if pow2
                        else f"rem.u32 ${{b32:{name}_shift}}, {source}, {period};\n"
                    )
                else:
                    expr = (
                        f"and.b32 ${{b32:{name}_shift}}, {source}, {-period & 0xFFFFFFFF};\n"
                        if pow2
                        else (
                            f"div.u32 ${{b32:{name}_shift}}, {source}, {period};\n"
                            f"mul.lo.u32 ${{b32:{name}_shift}}, ${{b32:{name}_shift}}, {period};\n"
                        )
                    )
                return (
                    expr + f"mov.b32 ${{b32:{name}_value}}, {table[0]};\n"
                    f"shl.b32 ${{b32:{name}_value}}, ${{b32:{name}_value}}, ${{b32:{name}_shift}};\n"
                    + (
                        f"cvt.u16.u32 {dest}, ${{b32:{name}_value}};\n"
                        if bits == 16
                        else f"mov.b32 {dest}, ${{b32:{name}_value}};\n"
                    )
                )
    return f"mov.b{bits} {dest}, {table[0]};\n" + "".join(
        f"setp.eq.u32 ${{pred:{name}_is{r}}}, {source}, {r};\n"
        f"@${{pred:{name}_is{r}}} mov.b{bits} {dest}, {v};\n"
        for r, v in enumerate(table[1:], 1)
    )


def index_expr(ops, cls, a, b):
    """Append ``cls(a, b)`` to ``ops``; integer operands become index constants."""
    from .gpu_dialect import IndexConstantOp

    operands = []
    for value in (a, b):
        if isinstance(value, int):
            constant = IndexConstantOp.build(value)
            ops.append(constant)
            value = constant.result()
        operands.append(value)
    op = cls.build(*operands)
    ops.append(op)
    return op.result()


def logical_tile_coordinates(
    rank, group, layout, groups_m, groups_n, ops, fast_axis="m"
):
    """Build logical CTA coordinates from a cluster rank and work-group index."""
    from .gpu_dialect import IndexAddOp, IndexMulOp, IndexDivOp, IndexModOp

    def expr(cls, a, b):
        return index_expr(ops, cls, a, b)

    if fast_axis == "m":
        gm = expr(IndexModOp, group, groups_m)
        gn = expr(IndexDivOp, group, groups_m)
    else:
        gm = expr(IndexDivOp, group, groups_n)
        gn = expr(IndexModOp, group, groups_n)
    m = expr(
        IndexAddOp, expr(IndexMulOp, gm, layout.m), expr(IndexModOp, rank, layout.m)
    )
    n = expr(
        IndexAddOp, expr(IndexMulOp, gn, layout.n), expr(IndexDivOp, rank, layout.m)
    )
    return m, n
