"""Plan the OSS GEMM corpus benchmark without importing or compiling Ayla.

The source corpus contains operator calls, not just three dimensions.  This
module keeps their layouts for reporting, then pools accepted calls by their
mathematical GEMM shape. Benchmarks create canonical contiguous A/BT operands,
so a recorded data order is never grounds for rejection. It deliberately has
no torch dependency: validation should be cheap and safe before reserving a GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable


CORPUS = Path(__file__).resolve().parent / "oss_pytorch_gemm_shapes.json"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".benchmarks" / "oss_gemm"
VALIDATION_CSV = OUTPUT_DIR / "validation.csv"
VALIDATION_JSON = OUTPUT_DIR / "validation.json"
RAW_CSV = OUTPUT_DIR / "raw.csv"
RAW_JSON = OUTPUT_DIR / "raw.json"
HEURISTICS_MD = Path(__file__).resolve().parent / "oss_pytorch_gemm_heuristics.md"
IMPROVEMENTS_CSV = OUTPUT_DIR / "improvements.csv"
IMPROVEMENTS_JSON = OUTPUT_DIR / "improvements.json"
SHAPE_IMPROVEMENTS_CSV = OUTPUT_DIR / "shape_improvements.csv"
SHAPE_IMPROVEMENTS_JSON = OUTPUT_DIR / "shape_improvements.json"
SEEDS_JSON = OUTPUT_DIR / "heuristic_seeds.json"

SUPPORTED_DTYPES = frozenset({"float16", "bfloat16"})

# Proxy for when an output-only CTA grid cannot expose enough parallelism for
# the reduction. Count 64-element K tiles per minimum 64x64 WGMMA output tile.
# NCU shows two split-K regimes: extreme reduction pressure, and an underfilled
# (<128 tile) output grid whose K-tile count is at least its output-tile count.
# Keep these calls visible in validation until cross-CTA reduction exists.
MIN_GEMM_TILE = 64
SPLIT_K_REDUCTION_PRESSURE_THRESHOLD = 32
MIN_OUTPUT_TILES_WITHOUT_SPLIT_K = 128


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> TensorSpec:
        return cls(
            shape=tuple(int(x) for x in value["shape"]),
            stride=tuple(int(x) for x in value["stride"]),
            dtype=str(value["dtype"]),
        )


@dataclass(frozen=True)
class Shape:
    """One operator signature, pooled across suites that recorded it."""

    op: str
    batch: int | None
    m: int
    n: int
    k: int
    args: tuple[TensorSpec, ...]
    count: int
    suites: tuple[str, ...]
    source_records: int

    @property
    def dtype(self) -> str:
        dtypes = {arg.dtype for arg in self.args}
        return next(iter(dtypes)) if len(dtypes) == 1 else "mixed"

    @property
    def shape_id(self) -> str:
        signature = (
            self.op, self.batch, self.m, self.n, self.k,
            tuple((arg.shape, arg.stride, arg.dtype) for arg in self.args),
        )
        digest = hashlib.sha1(repr(signature).encode()).hexdigest()[:12]
        return f"{self.op}-m{self.m}-n{self.n}-k{self.k}-{digest}"


@dataclass(frozen=True)
class Validation:
    shape: Shape
    reason_codes: tuple[str, ...]
    reason_details: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.reason_codes


@dataclass(frozen=True, kw_only=True)
class ScheduleConfig:
    """Every configurable knob on the static GEMM builder.

    Cache attributes are flattened here so CSV output remains useful.  They
    are expanded back into ``operand_move_attrs`` by :meth:`builder_kwargs`.
    Multicast, two-dimensional workgroup geometry, and exact CTA assignment are
    recorded explicitly so the raw results and generated heuristic document
    preserve every builder knob used by the search.
    """

    name: str
    output_tiles_per_cta: int
    output_m_tile: int
    output_n_tile: int
    warpgroup_n_tile: int
    operand_stages: int
    operand_k_tile: int
    output_slices: int
    output_store_buffers: int
    output_tma_inner_size: int | None
    output_store: str
    lhs_l2_cache_hint: str
    rhs_l2_cache_hint: str
    dst_l2_cache_hint: str
    lhs_l2_promotion: str | None = None
    ptx_optimization_level: int | None = None
    multicast: bool = False
    unsafe_reuse_smem: bool = False
    warpgroup_m_tile: int | None = None
    warpgroup_fast_axis: str = "n"
    cluster_m_ctas: int = 1
    cluster_n_ctas: int = 1
    output_cta_count: int | None = None
    output_tile_fast_axis: str = "n"
    unroll_output_tiles: bool = True
    rhs_tma_load_slices: int | None = None

    def builder_kwargs(self) -> dict[str, object]:
        if self.multicast and (self.cluster_m_ctas > 1) == (self.cluster_n_ctas > 1):
            raise ValueError("multicast requires exactly one clustered axis")
        if not self.multicast and (
            self.cluster_m_ctas > 1 or self.cluster_n_ctas > 1
        ):
            raise ValueError("cluster dimensions require multicast")
        lhs: dict[str, object] = {"l2_cache_hint": self.lhs_l2_cache_hint}
        rhs: dict[str, object] = {"l2_cache_hint": self.rhs_l2_cache_hint}
        if self.lhs_l2_promotion is not None:
            lhs["l2_promotion"] = self.lhs_l2_promotion
        if self.cluster_n_ctas > 1:
            lhs["multicast"] = True
        if self.cluster_m_ctas > 1:
            rhs["multicast"] = True
        return {
            "output_tiles_per_cta": self.output_tiles_per_cta,
            "output_cta_count": self.output_cta_count,
            "unroll_output_tiles": self.unroll_output_tiles,
            "output_tile_fast_axis": self.output_tile_fast_axis,
            "output_m_tile": self.output_m_tile,
            "output_n_tile": self.output_n_tile,
            "warpgroup_m_tile": self.warpgroup_m_tile,
            "warpgroup_n_tile": self.warpgroup_n_tile,
            "warpgroup_fast_axis": self.warpgroup_fast_axis,
            "cluster_m_ctas": self.cluster_m_ctas,
            "cluster_n_ctas": self.cluster_n_ctas,
            "operand_stages": self.operand_stages,
            "operand_k_tile": self.operand_k_tile,
            "rhs_tma_load_slices": self.rhs_tma_load_slices,
            "output_slices": self.output_slices,
            "output_store_buffers": self.output_store_buffers,
            "output_tma_inner_size": self.output_tma_inner_size,
            "output_store": self.output_store,
            "unsafe_reuse_smem": self.unsafe_reuse_smem,
            "operand_move_attrs": {
                "lhs": lhs,
                "rhs": rhs,
                "dst": {"l2_cache_hint": self.dst_l2_cache_hint},
            },
            "ptx_optimization_level": self.ptx_optimization_level,
        }


def _entry_key(entry: dict[str, Any]) -> tuple[object, ...]:
    args = tuple(
        (
            tuple(int(x) for x in arg["shape"]),
            tuple(int(x) for x in arg["stride"]),
            str(arg["dtype"]),
        )
        for arg in entry["args"]
    )
    return (
        str(entry["op"]), entry.get("b"), int(entry["m"]),
        int(entry["n"]), int(entry["k"]), args,
    )


def load_shapes(path: Path = CORPUS) -> list[Shape]:
    """Read the corpus and pool identical calls across recording suites."""
    entries = json.loads(path.read_text())
    counts: Counter[tuple[object, ...]] = Counter()
    suites: defaultdict[tuple[object, ...], set[str]] = defaultdict(set)
    records: Counter[tuple[object, ...]] = Counter()
    exemplars: dict[tuple[object, ...], dict[str, Any]] = {}
    for entry in entries:
        key = _entry_key(entry)
        counts[key] += int(entry["count"])
        suites[key].add(str(entry["suite"]))
        records[key] += 1
        exemplars[key] = entry

    shapes = []
    for key, count in counts.items():
        entry = exemplars[key]
        shapes.append(Shape(
            op=str(entry["op"]),
            batch=(None if entry.get("b") is None else int(entry["b"])),
            m=int(entry["m"]), n=int(entry["n"]), k=int(entry["k"]),
            args=tuple(TensorSpec.from_json(arg) for arg in entry["args"]),
            count=count,
            suites=tuple(sorted(suites[key])),
            source_records=records[key],
        ))
    shapes.sort(key=lambda shape: (-shape.count, shape.shape_id))
    return shapes


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _equivalent_strides(
    shape: tuple[int, ...], actual: tuple[int, ...], expected: tuple[int, ...],
) -> bool:
    """Compare layouts while ignoring arbitrary strides on singleton dims."""
    return len(shape) == len(actual) == len(expected) and all(
        extent == 1 or got == wanted
        for extent, got, wanted in zip(shape, actual, expected, strict=True)
    )


def source_layout(shape: Shape) -> str:
    """A compact description of the recorded operands' physical data order."""
    if len(shape.args) != 2 or any(len(arg.shape) != 2 for arg in shape.args):
        return "other"
    lhs, rhs = shape.args
    lhs_order = (
        "row" if _equivalent_strides(
            lhs.shape, lhs.stride, _contiguous_strides(lhs.shape),
        ) else "column" if _equivalent_strides(
            lhs.shape, lhs.stride, (1, lhs.shape[0]),
        ) else "other"
    )
    rhs_order = (
        "row" if _equivalent_strides(
            rhs.shape, rhs.stride, _contiguous_strides(rhs.shape),
        ) else "column" if _equivalent_strides(
            rhs.shape, rhs.stride, (1, rhs.shape[0]),
        ) else "other"
    )
    return f"{lhs_order}_{rhs_order}"


def validate_shape(shape: Shape) -> Validation:
    """Classify one shape before any module construction or GPU allocation."""
    reasons: list[tuple[str, str]] = []
    if shape.batch is not None or shape.op == "bmm":
        reasons.append(("batching", "batched GEMM is not implemented"))
    elif shape.op != "mm":
        reasons.append(("op", f"{shape.op} semantics are not implemented"))

    if shape.dtype not in SUPPORTED_DTYPES:
        reasons.append((
            "dtype",
            f"{shape.dtype} is unsupported; expected one of "
            f"{sorted(SUPPORTED_DTYPES)}",
        ))
    positive_dimensions = min(shape.m, shape.n, shape.k) > 0
    if not positive_dimensions:
        reasons.append(("shape", "M, N, and K must be positive"))
    if positive_dimensions:
        output_tiles = (
            (shape.m + MIN_GEMM_TILE - 1) // MIN_GEMM_TILE
            * ((shape.n + MIN_GEMM_TILE - 1) // MIN_GEMM_TILE)
        )
        reduction_tiles = (
            shape.k + MIN_GEMM_TILE - 1
        ) // MIN_GEMM_TILE
        high_reduction_pressure = (
            reduction_tiles
            >= SPLIT_K_REDUCTION_PRESSURE_THRESHOLD * output_tiles
        )
        underfilled_deep_reduction = (
            output_tiles < MIN_OUTPUT_TILES_WITHOUT_SPLIT_K
            and reduction_tiles >= output_tiles
        )
        needs_split_k = high_reduction_pressure or underfilled_deep_reduction
        if shape.op == "mm" and needs_split_k:
            reasons.append((
                "split_k",
                "temporarily skipped: reduction pressure is "
                f"{reduction_tiles / output_tiles:.1f} K tiles per output tile "
                f"across {output_tiles} output tiles; an "
                "efficient kernel requires split-K/Stream-K",
            ))
    if shape.op == "mm" and positive_dimensions and shape.m < MIN_GEMM_TILE:
        reasons.append((
            "small_m",
            f"M={shape.m} is below the current WGMMA M tile ({MIN_GEMM_TILE})",
        ))
    if shape.op == "mm" and (
        len(shape.args) != 2
        or shape.args[0].shape != (shape.m, shape.k)
        or shape.args[1].shape != (shape.k, shape.n)
    ):
        reasons.append(("shape", "operand shapes do not agree with M, N, and K"))
    alignment = []
    if shape.k % 64:
        alignment.append(
            f"K={shape.k} is not divisible by the minimum K tile (64)"
        )
    if shape.n % 8:
        alignment.append(
            f"N={shape.n} makes the FP16 TMA output row stride "
            "not divisible by 16 bytes"
        )
    if alignment:
        reasons.append(("alignment", "; ".join(alignment)))

    return Validation(
        shape=shape,
        reason_codes=tuple(code for code, _ in reasons),
        reason_details=tuple(detail for _, detail in reasons),
    )


def validate_shapes(shapes: Iterable[Shape]) -> list[Validation]:
    return [validate_shape(shape) for shape in shapes]


def benchmark_shapes(validations: Iterable[Validation]) -> list[Shape]:
    """Pool accepted signatures into canonical mathematical GEMM workloads.

    A source NN, NT, TN, or TT call contributes frequency to the same M/N/K
    workload. The benchmark measures that workload with contiguous ``A`` and
    ``BT`` because those are the physical operands accepted by Ayla.
    """
    accepted = [item.shape for item in validations if item.supported]
    counts: Counter[tuple[int, int, int, str]] = Counter()
    suites: defaultdict[tuple[int, int, int, str], set[str]] = defaultdict(set)
    records: Counter[tuple[int, int, int, str]] = Counter()
    for shape in accepted:
        key = (shape.m, shape.n, shape.k, shape.dtype)
        counts[key] += shape.count
        suites[key].update(shape.suites)
        records[key] += shape.source_records

    result = []
    for (m, n, k, dtype), count in counts.items():
        result.append(Shape(
            op="mm", batch=None, m=m, n=n, k=k,
            args=(
                TensorSpec(shape=(m, k), stride=(k, 1), dtype=dtype),
                TensorSpec(shape=(k, n), stride=(1, k), dtype=dtype),
            ),
            count=count,
            suites=tuple(sorted(suites[m, n, k, dtype])),
            source_records=records[m, n, k, dtype],
        ))
    result.sort(key=lambda shape: (-shape.count, shape.shape_id))
    return result


def validation_row(validation: Validation) -> dict[str, object]:
    shape = validation.shape
    return {
        "shape_id": shape.shape_id,
        "status": "supported" if validation.supported else "skipped",
        "reason_codes": list(validation.reason_codes),
        "reason_details": list(validation.reason_details),
        "op": shape.op,
        "batch": shape.batch,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "dtype": shape.dtype,
        "source_layout": source_layout(shape),
        "args": [asdict(arg) for arg in shape.args],
        "frequency": shape.count,
        "suites": list(shape.suites),
        "source_records": shape.source_records,
    }


def write_json(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def write_rows_csv(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("cannot write a CSV without rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    flattened = [
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    temporary.replace(output)


def _size_bucket(value: int) -> str:
    if value <= 128:
        return "xs"
    if value <= 512:
        return "small"
    if value <= 2048:
        return "medium"
    if value <= 8192:
        return "large"
    return "xl"


def regime(shape: Shape) -> dict[str, str]:
    if shape.m >= 4 * shape.n:
        aspect = "tall"
    elif shape.n >= 4 * shape.m:
        aspect = "wide"
    else:
        aspect = "balanced"
    alignment = "k128" if shape.k % 128 == 0 else "k64"
    return {
        "dtype": shape.dtype,
        "layout": "canonical_nt",
        "m_bucket": _size_bucket(shape.m),
        "n_bucket": _size_bucket(shape.n),
        "k_bucket": _size_bucket(shape.k),
        "aspect": aspect,
        "alignment": alignment,
    }


def regime_name(shape: Shape) -> str:
    fields = regime(shape)
    # N/K buckets remain columns in raw data and are summarized as distributions
    # in the report. Putting all three axes in the key fragments this corpus
    # into 147 mostly-singleton groups; M/aspect/alignment yields 23 useful
    # behavioral regimes instead.
    return "/".join(
        fields[key]
        for key in ("dtype", "layout", "m_bucket", "aspect", "alignment")
    )


def _config(
    name: str,
    *,
    output_tiles_per_cta: int = 1,
    output_cta_count: int | None = None,
    unroll_output_tiles: bool = True,
    output_tile_fast_axis: str = "n",
    output_m_tile: int = 128,
    output_n_tile: int = 256,
    warpgroup_m_tile: int | None = None,
    warpgroup_n_tile: int | None = None,
    warpgroup_fast_axis: str = "n",
    cluster_m_ctas: int = 1,
    cluster_n_ctas: int = 1,
    multicast: bool = False,
    operand_stages: int = 3,
    operand_k_tile: int = 64,
    rhs_tma_load_slices: int | None = None,
    output_slices: int = 1,
    output_store_buffers: int = 1,
    output_tma_inner_size: int | None = 128,
    output_store: str = "stmatrix",
    unsafe_reuse_smem: bool = False,
    lhs_l2_cache_hint: str = "evict_first",
    rhs_l2_cache_hint: str = "evict_first",
    dst_l2_cache_hint: str = "evict_first",
    lhs_l2_promotion: str | None = None,
    ptx_optimization_level: int | None = None,
) -> ScheduleConfig:
    return ScheduleConfig(
        name=name,
        output_tiles_per_cta=output_tiles_per_cta,
        output_cta_count=output_cta_count,
        unroll_output_tiles=unroll_output_tiles,
        output_tile_fast_axis=output_tile_fast_axis,
        output_m_tile=output_m_tile,
        output_n_tile=output_n_tile,
        warpgroup_m_tile=warpgroup_m_tile,
        warpgroup_n_tile=(
            output_n_tile if warpgroup_n_tile is None else warpgroup_n_tile
        ),
        warpgroup_fast_axis=warpgroup_fast_axis,
        cluster_m_ctas=cluster_m_ctas,
        cluster_n_ctas=cluster_n_ctas,
        operand_stages=operand_stages,
        operand_k_tile=operand_k_tile,
        rhs_tma_load_slices=rhs_tma_load_slices,
        output_slices=output_slices,
        output_store_buffers=output_store_buffers,
        output_tma_inner_size=output_tma_inner_size,
        output_store=output_store,
        unsafe_reuse_smem=unsafe_reuse_smem,
        lhs_l2_cache_hint=lhs_l2_cache_hint,
        rhs_l2_cache_hint=rhs_l2_cache_hint,
        dst_l2_cache_hint=dst_l2_cache_hint,
        lhs_l2_promotion=lhs_l2_promotion,
        ptx_optimization_level=ptx_optimization_level,
        multicast=multicast,
    )


def seed_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Return three reliable seeds distilled from the completed H100 sweep."""
    m64_n128_k4 = _config(
        "seed_m64_n128_k4", output_m_tile=64, output_n_tile=128,
        warpgroup_n_tile=128, operand_stages=4,
    )
    m64_dst_keep = replace(
        m64_n128_k4, name="seed_m64_n128_k4_dst_keep",
        dst_l2_cache_hint="evict_last",
    )
    m64_rhs_keep = replace(
        m64_n128_k4, name="seed_m64_n128_k4_rhs_keep",
        rhs_l2_cache_hint="evict_last",
    )
    scalar_k4 = _config(
        "seed_scalar_n64_k4", output_n_tile=64, warpgroup_n_tile=64,
        operand_stages=4, output_tma_inner_size=None, output_store="scalar",
    )
    scalar_k2 = replace(
        scalar_k4, name="seed_scalar_n64_k2", operand_stages=2,
    )
    scalar_k3 = replace(
        scalar_k4, name="seed_scalar_n64_k3", operand_stages=3,
    )
    learned_m64_n64_k128_s4 = _config(
        "seed_m64_n64_k128_s4", output_m_tile=64, output_n_tile=64,
        warpgroup_n_tile=64, operand_stages=4, operand_k_tile=128,
        output_tma_inner_size=None,
    )
    learned_m64_n64_k128_s6 = replace(
        learned_m64_n64_k128_s4, name="seed_m64_n64_k128_s6",
        operand_stages=6,
    )
    learned_m64_n96_k128_s3 = _config(
        "seed_m64_n96_k128_s3", output_m_tile=64, output_n_tile=96,
        warpgroup_n_tile=96, operand_stages=3, operand_k_tile=128,
        output_tma_inner_size=None,
    )

    if (
        shape.m >= 100_000
        and shape.n % 512 == 0
        and shape.k <= 128
        and (shape.n // 128) % 4 == 0
    ):
        multicast_n4 = _config(
            "seed_huge_m_multicast_n4_m128_n128_k64_s4",
            output_m_tile=128, output_n_tile=128,
            warpgroup_n_tile=128, operand_stages=4,
            cluster_n_ctas=4, multicast=True,
        )
        wg2d_nfast = _config(
            "seed_huge_m_wg2d_m128_n256_wn128_nfast",
            output_m_tile=128, output_n_tile=256,
            warpgroup_m_tile=64, warpgroup_n_tile=128,
            warpgroup_fast_axis="n", operand_stages=3,
        )
        return multicast_n4, replace(
            wg2d_nfast,
            name="seed_huge_m_wg2d_m128_n256_wn128_mfast",
            warpgroup_fast_axis="m",
        ), wg2d_nfast

    if (
        shape.m >= 8192
        and shape.n in (256, 512)
        and shape.k <= 2048
        and shape.m % 128 == 0
    ):
        wg2d_nfast = _config(
            "seed_tall_narrow_wg2d_m128_n256_wn128_nfast",
            output_m_tile=128, output_n_tile=256,
            warpgroup_m_tile=64, warpgroup_n_tile=128,
            warpgroup_fast_axis="n", operand_stages=3,
        )
        return wg2d_nfast, replace(
            wg2d_nfast,
            name="seed_tall_narrow_wg2d_m128_n256_wn128_mfast",
            warpgroup_fast_axis="m",
        ), replace(
            wg2d_nfast,
            name="seed_tall_narrow_wg2d_m128_n256_wn64_mfast",
            warpgroup_n_tile=64,
            warpgroup_fast_axis="m",
        )

    if (
        shape.k % 128 == 0
        and shape.m <= 512
        and shape.n <= 2048
        and shape.n % 64 == 0
    ):
        return (
            learned_m64_n64_k128_s4,
            learned_m64_n64_k128_s6,
            learned_m64_n96_k128_s3,
        )

    # Block-major TMA output requires a 128-byte-compatible row stride. Odd
    # multiples of eight remain valid through the scalar-output seeds.
    if shape.n % 64:
        return scalar_k2, scalar_k3, scalar_k4

    if shape.m <= 128:
        return m64_dst_keep, m64_rhs_keep, scalar_k4

    if shape.m <= 2048:
        if shape.n >= 4 * shape.m:
            wide = _config(
                "seed_wide_n208_k4", output_n_tile=208,
                warpgroup_n_tile=208, operand_stages=4,
                output_tma_inner_size=None,
                rhs_l2_cache_hint="evict_last",
            )
            learned_n96 = _config(
                "seed_wide_n96_k3", output_m_tile=64, output_n_tile=96,
                warpgroup_n_tile=96, operand_stages=3,
                operand_k_tile=(128 if shape.k % 128 == 0 else 64),
                output_tma_inner_size=None,
            )
            return wide, learned_n96, m64_rhs_keep
        return scalar_k4, m64_dst_keep, m64_rhs_keep

    if (
        shape.m >= 12_000
        and 384 <= shape.n <= 1024
        and shape.k <= 4096
    ):
        copied_m192_n192_2tile = _config(
            "copy_m192_n192_2tile_k64_s4", output_tiles_per_cta=2,
            output_m_tile=192, output_n_tile=192,
            warpgroup_n_tile=192, operand_stages=4, output_slices=3,
            output_tma_inner_size=None,
        )
        copied_m128_n256_2tile = _config(
            "copy_m128_n256_2tile_k64_s4", output_tiles_per_cta=2,
            output_m_tile=128, output_n_tile=256,
            warpgroup_n_tile=256, operand_stages=4, output_slices=2,
        )
        return copied_m192_n192_2tile, copied_m128_n256_2tile, m64_rhs_keep

    if shape.m >= 8192 and shape.n >= 1536 and shape.k <= 1024:
        copied_tiles = (16, 6, 8) if shape.m >= 16_384 else (8, 6, 2)
        return tuple(
            _config(
                f"copy_m128_n256_{tiles}tile_k64_s4",
                output_tiles_per_cta=tiles, output_m_tile=128,
                output_n_tile=256, warpgroup_n_tile=256,
                operand_stages=4, output_slices=2,
            )
            for tiles in copied_tiles
        )

    if (
        shape.m <= 8192
        and shape.m % 192 == 0
        and shape.n <= 2048
        and shape.n % 64 == 0
        and shape.k >= 2048
    ):
        m192_n128_s4 = _config(
            "seed_m192_n128_k64_s4", output_m_tile=192,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
        )
        m192_n128_s3 = replace(
            m192_n128_s4, name="seed_m192_n128_k64_s3", operand_stages=3,
        )
        m64_n128_s2 = replace(
            m64_n128_k4, name="seed_m64_n128_k64_s2", operand_stages=2,
        )
        return m192_n128_s4, m192_n128_s3, m64_n128_s2

    configs = [m64_dst_keep, m64_rhs_keep]
    if (
        shape.m >= 8192
        and shape.n <= 2048
        and shape.n % 128 == 0
        and shape.k <= 2048
    ):
        configs.append(_config(
            "seed_large_m192_n128_2tile_k4", output_tiles_per_cta=2,
            output_m_tile=192, output_n_tile=128,
            warpgroup_n_tile=128, operand_stages=4, output_slices=2,
        ))
    elif shape.m % 512 == 0:
        configs.append(_config(
            "seed_large_4tile_k4_split2", output_tiles_per_cta=4,
            operand_stages=4, output_slices=2,
        ))
    elif shape.m % 256 == 0:
        configs.append(_config(
            "seed_large_2tile_k4_split2", output_tiles_per_cta=2,
            operand_stages=4, output_slices=2,
        ))
    else:
        configs.append(scalar_k2)
    return tuple(configs)


def exploration_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Focused one-shape probes that expose otherwise-fixed GEMM knobs."""
    base = seed_configs(shape)[1]
    probes = [
        replace(base, name="probe_ptx_o1", ptx_optimization_level=1),
        replace(
            base, name="probe_lhs_keep", lhs_l2_cache_hint="evict_last",
        ),
        replace(
            base, name="probe_rhs_keep", rhs_l2_cache_hint="evict_last",
        ),
        replace(
            base, name="probe_dst_keep", dst_l2_cache_hint="evict_last",
        ),
        replace(
            base, name="probe_lhs_promote_256b", lhs_l2_promotion="256B",
        ),
        _config(
            "probe_scalar_store", output_n_tile=64, warpgroup_n_tile=64,
            operand_stages=3, output_tma_inner_size=None,
            output_store="scalar",
        ),
        _config(
            "probe_m192_n64", output_m_tile=192, output_n_tile=64,
            warpgroup_n_tile=64, operand_stages=3,
            output_tma_inner_size=None,
        ),
    ]
    if shape.k % 128 == 0:
        probes.append(_config(
            "probe_ktile128_n64_k2", output_n_tile=64,
            warpgroup_n_tile=64, operand_stages=2, operand_k_tile=128,
            output_tma_inner_size=None,
        ))
    # Config names are resume keys. Avoid repeating a seed when a probe's
    # values happen to match it apart from the descriptive name.
    return tuple(probes)


def fallback_config() -> ScheduleConfig:
    """Conservative one-CTA schedule for odd N and multi-tile M failures."""
    return _config(
        "fallback_scalar_n64_k3", output_n_tile=64,
        warpgroup_n_tile=64, operand_stages=3,
        output_tma_inner_size=None, output_store="scalar",
    )


def refinement_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Bounded follow-up around the broadly successful scalar N=64 schedule."""
    configs = [
        _config(
            "refine_scalar_n64_k2", output_n_tile=64,
            warpgroup_n_tile=64, operand_stages=2,
            output_tma_inner_size=None, output_store="scalar",
        ),
        _config(
            "refine_scalar_n64_k4", output_n_tile=64,
            warpgroup_n_tile=64, operand_stages=4,
            output_tma_inner_size=None, output_store="scalar",
        ),
    ]
    if shape.k % 128 == 0:
        configs.append(_config(
            "refine_scalar_n64_ktile128_k2", output_n_tile=64,
            warpgroup_n_tile=64, operand_stages=2, operand_k_tile=128,
            output_tma_inner_size=None, output_store="scalar",
        ))
    return tuple(configs)


def tile_refinement_configs() -> tuple[ScheduleConfig, ...]:
    """Probe smaller output tiles enabled for general corpus shapes."""
    return (
        _config(
            "refine_m64_n128_k3", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=3,
        ),
        _config(
            "refine_m64_n128_k4", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
        ),
        _config(
            "refine_m128_n128_k3", output_m_tile=128,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=3,
        ),
        _config(
            "refine_m64_n128_k4_dst_keep", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
            dst_l2_cache_hint="evict_last",
        ),
        _config(
            "refine_m64_n128_k4_rhs_keep", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
            rhs_l2_cache_hint="evict_last",
        ),
    )


def targeted_tuning_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Focused tile/mainloop sweep for the highest-impact remaining losses."""
    configs = [
        _config(
            f"tune_m64_n64_k64_s{stages}", output_m_tile=64,
            output_n_tile=64, warpgroup_n_tile=64,
            operand_stages=stages, output_tma_inner_size=None,
        )
        for stages in (2, 3, 4)
    ]
    configs.extend(
        _config(
            f"tune_m64_n64_scalar_k64_s{stages}", output_m_tile=64,
            output_n_tile=64, warpgroup_n_tile=64,
            operand_stages=stages, output_tma_inner_size=None,
            output_store="scalar",
        )
        for stages in (2, 3, 4)
    )
    configs.extend([
        _config(
            "tune_m64_n64_k64_s5", output_m_tile=64,
            output_n_tile=64, warpgroup_n_tile=64,
            operand_stages=5, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n64_k64_s6", output_m_tile=64,
            output_n_tile=64, warpgroup_n_tile=64,
            operand_stages=6, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n96_k64_s2", output_m_tile=64,
            output_n_tile=96, warpgroup_n_tile=96,
            operand_stages=2, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n96_k64_s3", output_m_tile=64,
            output_n_tile=96, warpgroup_n_tile=96,
            operand_stages=3, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n96_k64_s4", output_m_tile=64,
            output_n_tile=96, warpgroup_n_tile=96,
            operand_stages=4, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n96_k64_s5", output_m_tile=64,
            output_n_tile=96, warpgroup_n_tile=96,
            operand_stages=5, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n96_k64_s6", output_m_tile=64,
            output_n_tile=96, warpgroup_n_tile=96,
            operand_stages=6, output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n96_tma_k64_s3", output_m_tile=64,
            output_n_tile=96, warpgroup_n_tile=96, operand_stages=3,
        ),
        replace(
            configs[2], name="tune_m64_n64_k64_s4_rhs_keep",
            rhs_l2_cache_hint="evict_last",
        ),
        replace(
            configs[2], name="tune_m64_n64_k64_s4_lhs_keep",
            lhs_l2_cache_hint="evict_last",
        ),
        replace(
            configs[2], name="tune_m64_n64_k64_s4_dst_keep",
            dst_l2_cache_hint="evict_last",
        ),
        _config(
            "tune_m64_n128_k64_s2", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=2,
        ),
        _config(
            "tune_m64_n128_wg64_k64_s2", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=64, operand_stages=2,
            output_slices=2,
        ),
        _config(
            "tune_m64_n128_wg64_k64_s4", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=64, operand_stages=4,
            output_slices=2,
        ),
        _config(
            "tune_m64_n128_k64_s4_split2", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
            output_slices=2,
        ),
        _config(
            "tune_m64_n128_k64_s4_split2_dbuf2", output_m_tile=64,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
            output_slices=2, output_store_buffers=2,
        ),
        _config(
            "tune_m64_n208_k64_s2", output_m_tile=64,
            output_n_tile=208, warpgroup_n_tile=208, operand_stages=2,
            output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n208_k64_s4", output_m_tile=64,
            output_n_tile=208, warpgroup_n_tile=208, operand_stages=4,
            output_tma_inner_size=None,
        ),
        _config(
            "tune_m64_n256_k64_s2", output_m_tile=64,
            output_n_tile=256, warpgroup_n_tile=256, operand_stages=2,
        ),
        _config(
            "tune_m64_n256_wg64_k64_s2", output_m_tile=64,
            output_n_tile=256, warpgroup_n_tile=64, operand_stages=2,
            output_slices=4,
        ),
        _config(
            "tune_m64_n256_wg128_k64_s2", output_m_tile=64,
            output_n_tile=256, warpgroup_n_tile=128, operand_stages=2,
            output_slices=2,
        ),
        _config(
            "tune_m64_n256_wg128_k64_s3", output_m_tile=64,
            output_n_tile=256, warpgroup_n_tile=128, operand_stages=3,
            output_slices=2,
        ),
        _config(
            "tune_m64_n256_k64_s4", output_m_tile=64,
            output_n_tile=256, warpgroup_n_tile=256, operand_stages=4,
        ),
        _config(
            "tune_m128_n128_k64_s2", output_m_tile=128,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=2,
        ),
        _config(
            "tune_m128_n128_k64_s4", output_m_tile=128,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
        ),
        _config(
            "tune_m128_n256_k64_s2", output_m_tile=128,
            output_n_tile=256, warpgroup_n_tile=256, operand_stages=2,
        ),
        _config(
            "tune_m128_n256_k64_s4", output_m_tile=128,
            output_n_tile=256, warpgroup_n_tile=256, operand_stages=4,
        ),
        _config(
            "tune_m192_n64_k64_s2", output_m_tile=192,
            output_n_tile=64, warpgroup_n_tile=64, operand_stages=2,
            output_tma_inner_size=None,
        ),
        _config(
            "tune_m192_n64_k64_s4", output_m_tile=192,
            output_n_tile=64, warpgroup_n_tile=64, operand_stages=4,
            output_tma_inner_size=None,
        ),
        _config(
            "tune_m192_n128_k64_s2", output_m_tile=192,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=2,
        ),
        _config(
            "tune_m192_n128_k64_s3", output_m_tile=192,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=3,
        ),
        _config(
            "tune_m192_n128_k64_s4", output_m_tile=192,
            output_n_tile=128, warpgroup_n_tile=128, operand_stages=4,
        ),
        _config(
            "tune_m192_n128_2tile_k64_s2", output_tiles_per_cta=2,
            output_m_tile=192, output_n_tile=128, warpgroup_n_tile=128,
            operand_stages=2, output_slices=2,
        ),
        _config(
            "tune_m192_n128_2tile_k64_s3", output_tiles_per_cta=2,
            output_m_tile=192, output_n_tile=128, warpgroup_n_tile=128,
            operand_stages=3, output_slices=2,
        ),
        _config(
            "tune_m192_n128_2tile_k64_s4", output_tiles_per_cta=2,
            output_m_tile=192, output_n_tile=128, warpgroup_n_tile=128,
            operand_stages=4, output_slices=2,
        ),
    ])
    if shape.k % 128 == 0:
        for m_tile, n_tile in ((64, 64), (64, 128), (128, 64), (128, 128)):
            for stages in (2, 3, 4):
                configs.append(_config(
                    f"tune_m{m_tile}_n{n_tile}_k128_s{stages}",
                    output_m_tile=m_tile, output_n_tile=n_tile,
                    warpgroup_n_tile=n_tile, operand_stages=stages,
                    operand_k_tile=128,
                    output_tma_inner_size=(128 if n_tile == 128 else None),
                ))
        configs.extend([
            _config(
                "tune_m64_n64_k128_s5", output_m_tile=64,
                output_n_tile=64, warpgroup_n_tile=64, operand_stages=5,
                operand_k_tile=128, output_tma_inner_size=None,
            ),
            _config(
                "tune_m64_n64_k128_s6", output_m_tile=64,
                output_n_tile=64, warpgroup_n_tile=64, operand_stages=6,
                operand_k_tile=128, output_tma_inner_size=None,
            ),
            _config(
                "tune_m64_n128_wg64_k128_s2", output_m_tile=64,
                output_n_tile=128, warpgroup_n_tile=64, operand_stages=2,
                operand_k_tile=128, output_slices=2,
            ),
            _config(
                "tune_m64_n128_wg64_k128_s3", output_m_tile=64,
                output_n_tile=128, warpgroup_n_tile=64, operand_stages=3,
                operand_k_tile=128, output_slices=2,
            ),
            _config(
                "tune_m64_n96_k128_s3", output_m_tile=64,
                output_n_tile=96, warpgroup_n_tile=96, operand_stages=3,
                operand_k_tile=128, output_tma_inner_size=None,
            ),
            _config(
                "tune_m64_n96_k128_s4", output_m_tile=64,
                output_n_tile=96, warpgroup_n_tile=96, operand_stages=4,
                operand_k_tile=128, output_tma_inner_size=None,
            ),
            _config(
                "tune_m64_n96_k128_s5", output_m_tile=64,
                output_n_tile=96, warpgroup_n_tile=96, operand_stages=5,
                operand_k_tile=128, output_tma_inner_size=None,
            ),
            _config(
                "tune_m192_n128_k128_s2", output_m_tile=192,
                output_n_tile=128, warpgroup_n_tile=128, operand_stages=2,
                operand_k_tile=128,
            ),
            _config(
                "tune_m192_n128_2tile_k128_s2", output_tiles_per_cta=2,
                output_m_tile=192, output_n_tile=128,
                warpgroup_n_tile=128, operand_stages=2,
                operand_k_tile=128, output_slices=2,
            ),
        ])
    return tuple(configs)


def cublas_inspired_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Schedules matching tile/stage geometry observed in cuBLAS NCU traces."""
    configs: list[ScheduleConfig] = []
    if shape.n >= 48:
        configs.extend(
            _config(
                f"copy_m64_n48_k64_s{stages}", output_m_tile=64,
                output_n_tile=48, warpgroup_n_tile=48,
                operand_stages=stages, output_tma_inner_size=None,
            )
            for stages in (12, 15)
        )
    if shape.n >= 80:
        configs.extend(
            _config(
                f"copy_m64_n80_k64_s{stages}", output_m_tile=64,
                output_n_tile=80, warpgroup_n_tile=80,
                operand_stages=stages, output_tma_inner_size=None,
            )
            for stages in (8, 10, 11)
        )
    if shape.n >= 96:
        configs.extend(
            _config(
                f"copy_m64_n96_k64_s{stages}", output_m_tile=64,
                output_n_tile=96, warpgroup_n_tile=96,
                operand_stages=stages, output_tma_inner_size=None,
            )
            for stages in (8, 10)
        )
    if shape.n >= 128:
        configs.extend(
            _config(
                f"copy_m64_n128_k64_s{stages}", output_m_tile=64,
                output_n_tile=128, warpgroup_n_tile=128,
                operand_stages=stages,
            )
            for stages in (6, 7, 8)
        )
    if shape.n >= 152:
        configs.extend(
            _config(
                f"copy_m64_n152_k64_s{stages}", output_m_tile=64,
                output_n_tile=152, warpgroup_n_tile=152,
                operand_stages=stages, output_tma_inner_size=None,
            )
            for stages in (3, 4)
        )
    if shape.n >= 160:
        configs.extend(
            _config(
                f"copy_m64_n160_k64_s{stages}", output_m_tile=64,
                output_n_tile=160, warpgroup_n_tile=160,
                operand_stages=stages, output_tma_inner_size=None,
            )
            for stages in (4, 5)
        )
    if shape.m >= 192 and shape.n >= 192:
        configs.extend([
            _config(
                "copy_m192_n192_k64_s3", output_m_tile=192,
                output_n_tile=192, warpgroup_n_tile=192,
                operand_stages=3, output_tma_inner_size=None,
            ),
            *(
                _config(
                    f"copy_m192_n192_{tiles}tile_k64_s4",
                    output_tiles_per_cta=tiles, output_m_tile=192,
                    output_n_tile=192, warpgroup_n_tile=192,
                    operand_stages=4, output_slices=3,
                    output_tma_inner_size=None,
                )
                for tiles in (2, 4, 8)
                if shape.m >= tiles * 192
            ),
        ])
    if shape.m >= 128 and shape.n >= 256:
        configs.extend(
            _config(
                f"copy_m128_n256_{tiles}tile_k64_s4",
                output_tiles_per_cta=tiles, output_m_tile=128,
                output_n_tile=256, warpgroup_n_tile=256,
                operand_stages=4, output_slices=2,
            )
            for tiles in (2, 4, 6, 8, 16)
            if shape.m >= tiles * 128
        )
    return tuple(configs)


def resource_tuning_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Sweep output tile shape, pipeline depth, and proven SMEM reuse cases."""
    configs = [
        _config(
            f"reuse_m64_n{n_tile}_k64_s{stages}",
            output_m_tile=64, output_n_tile=n_tile,
            warpgroup_n_tile=n_tile, operand_stages=stages,
            output_tma_inner_size=None,
            unsafe_reuse_smem=True,
        )
        for n_tile in (64, 80, 96)
        for stages in (2, 3, 4, 5, 6)
        if shape.n >= n_tile
    ]
    tile_specs = [
        (128, 152, stages) for stages in (2, 3, 4)
    ]
    tile_specs.extend((128, 160, stages) for stages in (2, 3, 4))
    if shape.m >= 192:
        tile_specs.extend(((192, 64, 3), (192, 96, 3)))
    if shape.m >= 256:
        tile_specs.extend(
            (256, n_tile, stages)
            for n_tile in (64, 128)
            for stages in (2, 3, 4)
        )
    configs.extend(
        _config(
            f"tile_m{m_tile}_n{n_tile}_k64_s{stages}",
            output_m_tile=m_tile, output_n_tile=n_tile,
            warpgroup_n_tile=n_tile, operand_stages=stages,
            output_tma_inner_size=(128 if n_tile == 128 else None),
        )
        for m_tile, n_tile, stages in tile_specs
        if shape.n >= n_tile
    )
    return tuple(configs)


def coarse_tile_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Cover the still-unmeasured output-tile/depth space economically.

    The initial broad sweep established that output shape is substantially more
    important than cache hints or unsafe shared-memory reuse. Keep this sweep
    orthogonal to those knobs so a win can be attributed to tile geometry,
    then refine only the winning shapes and configurations.
    """
    tile_specs: list[tuple[int, int, tuple[int, ...]]] = []
    tile_specs.extend(
        (64, n_tile, (3, 4))
        for n_tile in (112, 144, 176, 192, 224)
    )
    tile_specs.extend(
        (128, n_tile, (3, 4))
        for n_tile in (80, 96, 112, 128, 144, 176, 192, 208, 224)
    )
    if shape.m >= 192:
        tile_specs.extend(
            (192, n_tile, (3,))
            for n_tile in (64, 80, 96, 112, 128, 144, 160)
        )
    if shape.m >= 256:
        tile_specs.extend(
            (256, n_tile, (2, 3))
            for n_tile in (64, 80, 96, 112, 128)
        )

    return tuple(
        _config(
            f"coarse_m{m_tile}_n{n_tile}_k64_s{stages}",
            output_m_tile=m_tile,
            output_n_tile=n_tile,
            warpgroup_n_tile=n_tile,
            operand_stages=stages,
            output_tma_inner_size=(128 if n_tile == 128 else None),
        )
        for m_tile, n_tile, stage_options in tile_specs
        for stages in stage_options
        if shape.m >= m_tile and shape.n >= n_tile
    )


def two_dimensional_wg_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """A bounded ablation of the existing linearized workgroup-tiling design."""
    specs: list[tuple[int, int, int, int, int, int]] = [
        (128, 128, 64, 64, 64, 3),
        (128, 128, 64, 64, 64, 4),
        (128, 208, 64, 104, 64, 4),
        (128, 256, 64, 128, 64, 3),
        (128, 256, 64, 64, 64, 3),
    ]
    if shape.k % 128 == 0:
        specs.append((128, 128, 64, 64, 128, 3))
    return tuple(
        _config(
            f"wg2d_m{m_tile}_n{n_tile}_wm{wg_m}_wn{wg_n}_"
            f"k{k_tile}_s{stages}_{fast}fast",
            output_m_tile=m_tile,
            output_n_tile=n_tile,
            warpgroup_m_tile=wg_m,
            warpgroup_n_tile=wg_n,
            warpgroup_fast_axis=fast,
            operand_k_tile=k_tile,
            operand_stages=stages,
            output_slices=1,
            output_tma_inner_size=(128 if n_tile in (128, 256) else None),
        )
        for m_tile, n_tile, wg_m, wg_n, k_tile, stages in specs
        for fast in ("m", "n")
        if shape.m % m_tile == 0 and shape.n % n_tile == 0
    )


def multicast_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Measure cluster multicast independently from 2-D workgroup tiling."""
    specs = (
        (128, 128, 3),
        (128, 128, 4),
        (128, 256, 3),
    )
    configs: list[ScheduleConfig] = []
    for m_tile, n_tile, stages in specs:
        if shape.m % m_tile or shape.n % n_tile:
            continue
        m_ctas = shape.m // m_tile
        n_ctas = shape.n // n_tile
        for cluster_axis, extent in (("m", m_ctas), ("n", n_ctas)):
            for cluster_size in (2, 4):
                if extent % cluster_size:
                    continue
                configs.append(_config(
                    f"multicast_{cluster_axis}{cluster_size}_m{m_tile}_"
                    f"n{n_tile}_k64_s{stages}",
                    output_m_tile=m_tile,
                    output_n_tile=n_tile,
                    warpgroup_n_tile=(128 if n_tile == 128 else 256),
                    operand_stages=stages,
                    cluster_m_ctas=(cluster_size if cluster_axis == "m" else 1),
                    cluster_n_ctas=(cluster_size if cluster_axis == "n" else 1),
                    multicast=True,
                ))
    return tuple(configs)


def pipeline_tile_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Refine successful tile widths with deeper K64 and feasible K128 pipes."""
    specs: list[tuple[int, int, int, int]] = [
        # Deeper K64 pipelines around widths that won the broad tile sweep.
        (64, 112, 64, 5), (64, 112, 64, 6),
        (64, 144, 64, 5),
        (64, 160, 64, 5),
        (64, 176, 64, 5),
        (128, 80, 64, 5), (128, 80, 64, 6),
        (128, 96, 64, 5), (128, 96, 64, 6),
        (128, 112, 64, 5), (128, 112, 64, 6),
        (128, 144, 64, 5),
        (128, 152, 64, 5),
        (128, 160, 64, 5),
        (128, 176, 64, 5),
        (192, 64, 64, 5),
        (192, 80, 64, 4), (192, 80, 64, 5),
        (192, 96, 64, 4),
        (192, 112, 64, 4),
        (256, 80, 64, 4),
        (256, 96, 64, 4),
    ]
    if shape.k % 128 == 0:
        specs.extend([
            (64, 80, 128, 3), (64, 80, 128, 4),
            (64, 112, 128, 3), (64, 112, 128, 4),
            (64, 144, 128, 2), (64, 144, 128, 3),
            (64, 160, 128, 2), (64, 160, 128, 3),
            (64, 192, 128, 2), (64, 192, 128, 3),
            (128, 80, 128, 2), (128, 80, 128, 3),
            (128, 96, 128, 2), (128, 96, 128, 3),
            (128, 112, 128, 2), (128, 112, 128, 3),
            (128, 144, 128, 2),
            (128, 160, 128, 2),
            (192, 64, 128, 2), (192, 64, 128, 3),
            (192, 80, 128, 2),
            (192, 96, 128, 2),
            (192, 112, 128, 2),
            (256, 64, 128, 2),
            (256, 80, 128, 2),
            (256, 96, 128, 2),
        ])
    return tuple(
        _config(
            f"pipe_m{m_tile}_n{n_tile}_k{k_tile}_s{stages}",
            output_m_tile=m_tile,
            output_n_tile=n_tile,
            warpgroup_n_tile=n_tile,
            operand_k_tile=k_tile,
            operand_stages=stages,
            output_tma_inner_size=(128 if n_tile == 128 else None),
        )
        for m_tile, n_tile, k_tile, stages in specs
        if shape.m >= m_tile and shape.n >= n_tile
    )


def gap_tuning_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Fill geometry and epilogue gaps exposed by the merged-search audit."""
    configs: list[ScheduleConfig] = []
    for n_tile in (48, 56, 72):
        if shape.n < n_tile:
            continue
        configs.extend(
            _config(
                f"gap_m64_n{n_tile}_k64_s{stages}",
                output_m_tile=64, output_n_tile=n_tile,
                warpgroup_n_tile=n_tile, operand_stages=stages,
                output_tma_inner_size=None,
            )
            for stages in (3, 4, 5, 6)
        )
        if shape.k % 128 == 0:
            configs.extend(
                _config(
                    f"gap_m64_n{n_tile}_k128_s{stages}",
                    output_m_tile=64, output_n_tile=n_tile,
                    warpgroup_n_tile=n_tile, operand_k_tile=128,
                    operand_stages=stages, output_tma_inner_size=None,
                )
                for stages in (3, 4)
            )

    if shape.m >= 128 and shape.n >= 64:
        configs.extend(
            _config(
                f"gap_m128_n64_block_k64_s{stages}",
                output_m_tile=128, output_n_tile=64,
                warpgroup_n_tile=64, operand_stages=stages,
                output_tma_inner_size=64,
            )
            for stages in (2, 3, 4, 5)
        )
        if shape.k % 128 == 0:
            configs.extend(
                _config(
                    f"gap_m128_n64_block_k128_s{stages}",
                    output_m_tile=128, output_n_tile=64,
                    warpgroup_n_tile=64, operand_k_tile=128,
                    operand_stages=stages, output_tma_inner_size=64,
                )
                for stages in (2, 3, 4)
            )

    if shape.n >= 128:
        configs.extend([
            _config(
                "gap_m64_n128_wg64_k64_s3_gather",
                output_m_tile=64, output_n_tile=128,
                warpgroup_n_tile=64, operand_stages=3,
            ),
            _config(
                "gap_m64_n128_wg64_k64_s3_split2",
                output_m_tile=64, output_n_tile=128,
                warpgroup_n_tile=64, operand_stages=3, output_slices=2,
            ),
            _config(
                "gap_m64_n128_wg64_k64_s3_split2_dbuf2",
                output_m_tile=64, output_n_tile=128,
                warpgroup_n_tile=64, operand_stages=3, output_slices=2,
                output_store_buffers=2,
            ),
        ])
        if shape.k % 128 == 0:
            configs.extend([
                _config(
                    "gap_m64_n128_wg64_k128_s4_split2",
                    output_m_tile=64, output_n_tile=128,
                    warpgroup_n_tile=64, operand_k_tile=128,
                    operand_stages=4, output_slices=2,
                ),
                _config(
                    "gap_m64_n128_wg64_k128_s4_split2_dbuf2",
                    output_m_tile=64, output_n_tile=128,
                    warpgroup_n_tile=64, operand_k_tile=128,
                    operand_stages=4, output_slices=2,
                    output_store_buffers=2,
                ),
            ])

    if shape.m % 128 == 0 and shape.n % 256 == 0:
        configs.extend(
            _config(
                f"gap_wg2d_m128_n256_wn128_k64_s{stages}_{fast}fast",
                output_m_tile=128, output_n_tile=256,
                warpgroup_m_tile=64, warpgroup_n_tile=128,
                warpgroup_fast_axis=fast, operand_stages=stages,
            )
            for stages in (2, 4)
            for fast in ("m", "n")
        )

    for m_tile, n_tile, stages in (
        (128, 80, 5), (128, 80, 6),
        (128, 96, 5), (128, 96, 6),
        (192, 96, 4), (192, 112, 4),
    ):
        if shape.m < m_tile or shape.n < n_tile:
            continue
        configs.extend(
            _config(
                f"gap_m{m_tile}_n{n_tile}_k64_s{stages}_split2"
                f"{('_dbuf2' if buffers == 2 else '')}",
                output_m_tile=m_tile, output_n_tile=n_tile,
                warpgroup_n_tile=n_tile, operand_stages=stages,
                output_slices=2, output_store_buffers=buffers,
                output_tma_inner_size=None,
            )
            for buffers in (1, 2)
        )
    return tuple(configs)


def narrow_tile_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Refine the N=48/56/72 family that generalized across the corpus."""
    specs: list[tuple[int, int, int, int]] = []
    specs.extend(
        (64, n_tile, 64, stages)
        for n_tile in (48, 56, 72, 80)
        for stages in ((7, 8) if n_tile != 80 else (7,))
    )
    specs.extend(
        (m_tile, n_tile, 64, stages)
        for m_tile, stage_options in ((128, (4, 5, 6)), (192, (3, 4, 5)),
                                      (256, (2, 3, 4)))
        for n_tile in (48, 56, 72)
        for stages in stage_options
    )
    if shape.k % 128 == 0:
        specs.extend(
            (64, n_tile, 128, stages)
            for n_tile in (48, 56, 72)
            for stages in (5, 6)
        )
        specs.extend(
            (m_tile, n_tile, 128, stages)
            for m_tile, stage_options in ((128, (3, 4)), (192, (2, 3)),
                                          (256, (2,)))
            for n_tile in (48, 56, 72)
            for stages in stage_options
        )
    return tuple(
        _config(
            f"narrow_m{m_tile}_n{n_tile}_k{k_tile}_s{stages}",
            output_m_tile=m_tile, output_n_tile=n_tile,
            warpgroup_n_tile=n_tile, operand_k_tile=k_tile,
            operand_stages=stages, output_tma_inner_size=None,
        )
        for m_tile, n_tile, k_tile, stages in specs
        if shape.m >= m_tile and shape.n >= n_tile
    )


def exact_cta_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Test balanced flattened output assignment at useful H100 grid sizes."""
    configs: list[ScheduleConfig] = []

    def add(
        m_tile: int,
        n_tile: int,
        ctas: int,
        stages: int,
        k_tile: int = 64,
    ) -> None:
        logical_tiles = (
            (shape.m + m_tile - 1) // m_tile
            * ((shape.n + n_tile - 1) // n_tile)
        )
        waves = (logical_tiles + ctas - 1) // ctas
        if (
            shape.m < m_tile or shape.n < n_tile
            or waves > 8
            or logical_tiles % ctas
            or (shape.k > 2048 and waves > 2)
            or (k_tile == 128 and shape.k % 128)
        ):
            return
        for fast in ("m", "n"):
            configs.append(_config(
                f"exact{ctas}_m{m_tile}_n{n_tile}_k{k_tile}_"
                f"s{stages}_{fast}fast",
                output_cta_count=ctas,
                output_tile_fast_axis=fast,
                output_m_tile=m_tile,
                output_n_tile=n_tile,
                warpgroup_n_tile=n_tile,
                operand_k_tile=k_tile,
                operand_stages=stages,
                output_tma_inner_size=(128 if n_tile == 128 else None),
            ))

    for m_tile, n_tile in ((128, 256), (192, 192), (192, 128)):
        logical_tiles = (
            (shape.m + m_tile - 1) // m_tile
            * ((shape.n + n_tile - 1) // n_tile)
        )
        divisors = [
            count for count in range(96, 265)
            if logical_tiles % count == 0
        ]
        cta_counts = {
            min(divisors, key=lambda count: abs(count - target))
            for target in (132, 264)
        } if divisors else set()
        for ctas in sorted(cta_counts):
            for stages in (2, 3):
                add(m_tile, n_tile, ctas, stages)

    use_k128 = shape.m <= 2048 and shape.k % 128 == 0
    k_tile = 128 if use_k128 else 64
    stages = 4 if use_k128 else (6 if shape.m <= 2048 else 3)
    for m_tile, n_tile, ctas in (
        (64, 64, 144),
        (64, 72, 132),
        (64, 80, 120),
        (64, 96, 96),
        (128, 64, 128),
    ):
        add(m_tile, n_tile, ctas, stages, k_tile)
    return tuple(configs)


def exact_cta_refinement_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Refine exact-grid winners without reintroducing large/partial waves."""
    configs: list[ScheduleConfig] = []
    for base in exact_cta_configs(shape):
        if base.output_m_tile == 192 and base.output_n_tile == 128:
            if base.operand_stages == 3:
                configs.extend([
                    replace(
                        base,
                        name=base.name.replace("_s3_", "_s3_split2_"),
                        output_slices=2,
                    ),
                    replace(
                        base,
                        name=base.name.replace("_s3_", "_s3_split2_dbuf2_"),
                        output_slices=2, output_store_buffers=2,
                    ),
                    replace(
                        base,
                        name=base.name.replace("_s3_", "_s4_"),
                        operand_stages=4,
                    ),
                ])
        elif (
            base.output_m_tile == 192
            and base.output_n_tile == 192
            and base.operand_stages == 3
        ):
            configs.append(replace(
                base,
                name=base.name.replace("_s3_", "_s3_split3_"),
                output_slices=3,
            ))
        elif (
            base.output_m_tile == 128
            and base.output_n_tile == 256
            and base.operand_stages == 3
        ):
            configs.extend([
                replace(
                    base,
                    name=base.name.replace("_s3_", "_s3_split2_"),
                    output_slices=2,
                ),
                replace(
                    base,
                    name=base.name.replace("_s3_", "_s3_split2_dbuf2_"),
                    output_slices=2, output_store_buffers=2,
                ),
            ])
    return tuple(configs)


def static_loop_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Use a compact static grid-stride loop near one H100 wave.

    Unlike the exact-grid unrolled search, this remains compact for large
    output spaces and permits a guarded final wave. The loop is still fully
    static: its trip count and every tile coordinate are compile-time known.
    """
    if shape.m <= 256:
        tiles = ((64, 64), (64, 80), (64, 96), (128, 64))
    elif shape.n <= 384:
        tiles = ((64, 128), (128, 64), (128, 128), (192, 128))
    else:
        tiles = ((128, 128), (128, 256), (192, 128), (192, 192))

    if shape.k <= 512:
        k_specs = ((64, 2), (64, 3), (64, 4))
    elif shape.k <= 2048:
        k_specs = ((64, 3), (64, 4))
    else:
        k_specs = ((64, 4), (64, 6))
    if shape.m <= 1024 and shape.k % 128 == 0:
        k_specs += ((128, 3), (128, 4))

    configs: list[ScheduleConfig] = []
    for m_tile, n_tile in tiles:
        if shape.m < m_tile or shape.n < n_tile:
            continue
        logical_tiles = (
            (shape.m + m_tile - 1) // m_tile
            * ((shape.n + n_tile - 1) // n_tile)
        )
        if logical_tiles < 96:
            continue
        for cta_count in (128, 132):
            waves = (logical_tiles + cta_count - 1) // cta_count
            if logical_tiles % cta_count == 0 and waves <= 8:
                continue
            for k_tile, stages in k_specs:
                estimated_smem = (
                    2 * (m_tile + n_tile) * k_tile * stages
                    + 2 * m_tile * n_tile
                )
                if estimated_smem > 225 * 1024:
                    continue
                for fast in ("m", "n"):
                    configs.append(_config(
                        f"staticloop{cta_count}_m{m_tile}_n{n_tile}_"
                        f"k{k_tile}_s{stages}_{fast}fast",
                        output_cta_count=cta_count,
                        unroll_output_tiles=False,
                        output_tile_fast_axis=fast,
                        output_m_tile=m_tile,
                        output_n_tile=n_tile,
                        warpgroup_n_tile=n_tile,
                        operand_k_tile=k_tile,
                        operand_stages=stages,
                        output_slices=(2 if n_tile >= 176 else 1),
                        output_tma_inner_size=(
                            128 if n_tile % 128 == 0 else None
                        ),
                    ))
    return tuple(configs)


def local_refinement_configs(
    shape: Shape,
    base: ScheduleConfig,
) -> tuple[ScheduleConfig, ...]:
    """Generate a small neighborhood around one shape's measured winner."""
    candidates: list[ScheduleConfig] = []
    for stages in range(max(2, base.operand_stages - 2),
                        min(8, base.operand_stages + 2) + 1):
        if stages != base.operand_stages:
            candidates.append(replace(base, operand_stages=stages))

    if base.warpgroup_m_tile is None and not base.multicast:
        for m_tile in (64, 128, 192, 256):
            if m_tile == base.output_m_tile or m_tile > shape.m:
                continue
            candidates.append(replace(base, output_m_tile=m_tile))
        for delta in (-16, -8, 8, 16):
            n_tile = base.output_n_tile + delta
            if not 48 <= n_tile <= 384 or n_tile > shape.n:
                continue
            if base.warpgroup_n_tile == base.output_n_tile:
                wg_n = n_tile if n_tile <= 256 else 128
            elif n_tile % base.warpgroup_n_tile == 0:
                wg_n = base.warpgroup_n_tile
            else:
                wg_n = n_tile if n_tile <= 256 else 128
            if n_tile % wg_n:
                continue
            candidates.append(replace(
                base, output_n_tile=n_tile, warpgroup_n_tile=wg_n,
                output_tma_inner_size=(128 if n_tile % 64 == 0 else None),
            ))

    if base.operand_k_tile == 64 and shape.k % 128 == 0:
        for stages in (2, 3, 4):
            candidates.append(replace(
                base, operand_k_tile=128, operand_stages=stages,
            ))
    elif base.operand_k_tile == 128:
        for stages in (3, 4, 5, 6):
            candidates.append(replace(
                base, operand_k_tile=64, operand_stages=stages,
            ))

    if base.warpgroup_m_tile is None:
        for slices in (2, 3, 4, 6, 8):
            if (
                base.output_n_tile % slices
                or (base.output_n_tile // slices) % 8
            ):
                continue
            candidates.extend([
                replace(
                    base, output_slices=slices, output_store_buffers=1,
                ),
                replace(
                    base, output_slices=slices, output_store_buffers=slices,
                ),
            ])
    if base.warpgroup_m_tile is not None:
        candidates.append(replace(
            base,
            warpgroup_fast_axis=(
                "m" if base.warpgroup_fast_axis == "n" else "n"
            ),
        ))
    if base.output_cta_count is not None:
        candidates.append(replace(
            base,
            output_tile_fast_axis=(
                "m" if base.output_tile_fast_axis == "n" else "n"
            ),
        ))
        if not base.unroll_output_tiles:
            candidates.extend(
                replace(base, output_cta_count=cta_count)
                for cta_count in (96, 108, 120, 128, 132)
                if cta_count != base.output_cta_count
            )
    elif not base.multicast:
        logical_tiles = (
            (shape.m + base.output_m_tile - 1) // base.output_m_tile
            * ((shape.n + base.output_n_tile - 1) // base.output_n_tile)
        )
        candidates.extend(
            replace(
                base,
                output_tiles_per_cta=1,
                output_cta_count=cta_count,
                unroll_output_tiles=False,
            )
            for cta_count in (96, 108, 120, 128, 132)
            if cta_count < logical_tiles
        )
    candidates.extend([
        replace(base, rhs_l2_cache_hint="evict_last"),
        replace(base, dst_l2_cache_hint="evict_last"),
    ])
    for lhs_hint in ("evict_first", "evict_last"):
        for rhs_hint in ("evict_first", "evict_last"):
            for dst_hint in ("evict_first", "evict_last"):
                candidates.append(replace(
                    base,
                    lhs_l2_cache_hint=lhs_hint,
                    rhs_l2_cache_hint=rhs_hint,
                    dst_l2_cache_hint=dst_hint,
                ))
    candidates.extend(
        replace(base, ptx_optimization_level=level)
        for level in (1, 2, 3)
    )
    if (
        base.output_m_tile == 64
        and base.output_n_tile <= 96
        and base.output_cta_count is None
        and not base.multicast
    ):
        candidates.append(replace(base, unsafe_reuse_smem=True))
    if base.output_n_tile % 64 == 0:
        candidates.append(replace(
            base,
            output_tma_inner_size=(
                None if base.output_tma_inner_size is not None else 128
            ),
        ))

    unique: dict[tuple[object, ...], ScheduleConfig] = {}
    for candidate in candidates:
        if candidate == base:
            continue
        estimated_smem = (
            2
            * (candidate.output_m_tile + candidate.output_n_tile)
            * candidate.operand_k_tile
            * candidate.operand_stages
            + 2
            * candidate.output_m_tile
            * candidate.output_n_tile
            * candidate.output_store_buffers
        )
        if not candidate.unsafe_reuse_smem and estimated_smem > 225 * 1024:
            continue
        if candidate.output_cta_count is not None:
            logical_tiles = (
                (shape.m + candidate.output_m_tile - 1)
                // candidate.output_m_tile
                * ((shape.n + candidate.output_n_tile - 1)
                   // candidate.output_n_tile)
            )
            waves = (
                logical_tiles + candidate.output_cta_count - 1
            ) // candidate.output_cta_count
            if candidate.unroll_output_tiles:
                if logical_tiles % candidate.output_cta_count or waves > 8:
                    continue
                if shape.k > 2048 and waves > 2:
                    continue
            elif logical_tiles < candidate.output_cta_count:
                # A compact static loop supports guarded final waves and an
                # arbitrary trip count. Avoid launching CTAs that can never
                # receive even one tile, but retain all other partial grids.
                continue
        values = tuple(
            getattr(candidate, field.name)
            for field in fields(ScheduleConfig)
            if field.name != "name"
        )
        unique.setdefault(values, candidate)

    return tuple(
        replace(
            config,
            name=(
                f"local_m{config.output_m_tile}_n{config.output_n_tile}_"
                f"wm{config.warpgroup_m_tile or 0}_wn{config.warpgroup_n_tile}_"
                f"k{config.operand_k_tile}_s{config.operand_stages}_"
                f"sl{config.output_slices}_b{config.output_store_buffers}_"
                f"c{config.output_cta_count or 0}_"
                f"{config.output_tile_fast_axis}{config.warpgroup_fast_axis}_"
                f"l{'l' if config.lhs_l2_cache_hint == 'evict_last' else 'f'}"
                f"r{'l' if config.rhs_l2_cache_hint == 'evict_last' else 'f'}"
                f"d{'l' if config.dst_l2_cache_hint == 'evict_last' else 'f'}"
                f"_o{config.ptx_optimization_level or 0}_"
                f"u{int(bool(config.unsafe_reuse_smem))}_"
                f"t{config.output_tma_inner_size or 0}"
            ),
        )
        for config in unique.values()
    )


def rhs_tma_slicing_configs(
    shape: Shape,
    base: ScheduleConfig,
) -> tuple[ScheduleConfig, ...]:
    """Vary RHS TMA transaction granularity around a measured winner.

    A slice count divides the leading N extent of the promoted RHS tile into
    independent TMA loads.  This changes transaction and barrier geometry
    without changing the math tile, making it a useful low-risk refinement of
    configurations that are already good.
    """
    del shape  # Reserved for future shape-dependent pruning.
    effective_base = (
        base.output_n_tile // base.warpgroup_n_tile
        if base.rhs_tma_load_slices is None and base.output_n_tile == 384
        else (base.rhs_tma_load_slices or 1)
    )
    counts = (
        count for count in (1, 2, 3, 4, 6, 8)
        if base.output_n_tile % count == 0 and count != effective_base
    )
    return tuple(
        replace(
            base,
            name=f"rhsload{count}_{base.name}",
            rhs_tma_load_slices=count,
        )
        for count in counts
    )


def workgroup_parallelism_configs(
    shape: Shape,
    base: ScheduleConfig,
) -> tuple[ScheduleConfig, ...]:
    """Partition a winning CTA tile over more compute warpgroups.

    This keeps the CTA's output tile, K pipeline, and work distribution fixed.
    M=64 tiles are split along N; taller tiles are split into 64-row
    warpgroups, optionally retaining an existing N split.  It directly tests
    whether independent WGMMA issue is the missing latency-hiding mechanism.
    """
    del shape  # Geometry validity depends on the tile, not the full problem.
    candidates: list[ScheduleConfig] = []
    if base.output_m_tile == 64:
        for groups in (2, 3, 4):
            wg_n, remainder = divmod(base.output_n_tile, groups)
            if remainder or wg_n < 16 or wg_n % 8:
                continue
            candidates.append(replace(
                base,
                name=f"wgparallel_n{groups}_{base.name}",
                warpgroup_n_tile=wg_n,
            ))
    else:
        wg_n_options = {base.output_n_tile, base.warpgroup_n_tile}
        for wg_n in sorted(wg_n_options):
            if base.output_n_tile % wg_n:
                continue
            total_groups = (
                base.output_m_tile // 64
                * (base.output_n_tile // wg_n)
            )
            if total_groups > 8:
                continue
            for fast_axis in ("m", "n"):
                candidates.append(replace(
                    base,
                    name=(
                        f"wgparallel_m{base.output_m_tile // 64}_"
                        f"n{base.output_n_tile // wg_n}_{fast_axis}_"
                        f"{base.name}"
                    ),
                    warpgroup_m_tile=64,
                    warpgroup_n_tile=wg_n,
                    warpgroup_fast_axis=fast_axis,
                ))
    return tuple(candidates)


def smem_reuse_configs(
    shape: Shape,
    base: ScheduleConfig,
) -> tuple[ScheduleConfig, ...]:
    """Reuse the operand arena for one-tile winners after their final read."""
    del shape
    if (
        base.unsafe_reuse_smem
        or base.output_tiles_per_cta != 1
        or base.output_cta_count is not None
        or base.multicast
        or base.output_store != "stmatrix"
    ):
        return ()
    return (replace(
        base,
        name=f"smemreuse_{base.name}",
        unsafe_reuse_smem=True,
    ),)


def cluster_tuning_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Combine useful narrow tiles with cluster operand reuse."""
    specs: list[tuple[int, int, int, int, int, str]] = [
        (64, 96, 64, stages, 8, "n") for stages in (4, 5, 6, 8)
    ]
    specs.extend(
        (64, 72, 64, stages, 4, "m") for stages in (4, 5, 6)
    )
    specs.extend(
        (128, 64, 64, stages, 8, "n") for stages in (2, 3, 4)
    )
    specs.extend(
        (128, 128, 64, stages, cluster, "n")
        for stages in (2, 3, 4)
        for cluster in (2, 4, 8)
    )
    if shape.k % 128 == 0:
        specs.extend(
            (64, 96, 128, stages, 8, "n") for stages in (3, 4, 5)
        )
        specs.extend(
            (64, 72, 128, stages, 4, "m") for stages in (4, 5, 6)
        )
        specs.extend(
            (64, 128, 128, stages, 8, "n") for stages in (3, 4)
        )

    configs: list[ScheduleConfig] = []
    for m_tile, n_tile, k_tile, stages, cluster, axis in specs:
        m_ctas = (shape.m + m_tile - 1) // m_tile
        n_ctas = (shape.n + n_tile - 1) // n_tile
        if (
            shape.m < m_tile or shape.n < n_tile
            or (axis == "m" and m_ctas % cluster)
            or (axis == "n" and n_ctas % cluster)
        ):
            continue
        configs.append(_config(
            f"cluster_{axis}{cluster}_m{m_tile}_n{n_tile}_"
            f"k{k_tile}_s{stages}",
            output_m_tile=m_tile, output_n_tile=n_tile,
            warpgroup_n_tile=n_tile, operand_k_tile=k_tile,
            operand_stages=stages,
            cluster_m_ctas=(cluster if axis == "m" else 1),
            cluster_n_ctas=(cluster if axis == "n" else 1),
            multicast=True,
            output_tma_inner_size=(128 if n_tile in (64, 128) else None),
        ))
    return tuple(configs)


def architecture_gap_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Probe tile/depth combinations highlighted by the remaining tail.

    NCU shows that the small-output, deep-K cuBLAS kernels devote nearly the
    full SMEM budget to 11--13 K64 stages.  Its 80x64 geometry can be tested
    in the supported 64x80 orientation; partial M warpgroups remain a separate
    library TODO.  The tall, short-K tail instead benefits from wide
    M=256/N=160 tiles and two-CTA operand reuse.  Keep both families here so
    their value can be measured directly.
    """
    configs: list[ScheduleConfig] = []

    if shape.m <= 2048 and shape.k >= 2048:
        deep_specs = (
            (64, 64, range(9, 14)),
            (64, 80, range(8, 12)),
            (64, 96, range(8, 11)),
        )
        for m_tile, n_tile, stages_options in deep_specs:
            if shape.m < m_tile or shape.n < n_tile:
                continue
            for stages in stages_options:
                estimated_smem = (
                    2 * (m_tile + n_tile) * 64 * stages
                    + 2 * m_tile * n_tile
                )
                if estimated_smem > 225 * 1024:
                    continue
                configs.append(_config(
                    f"archdeep_m{m_tile}_n{n_tile}_k64_s{stages}",
                    output_m_tile=m_tile, output_n_tile=n_tile,
                    warpgroup_n_tile=n_tile, operand_k_tile=64,
                    operand_stages=stages,
                    output_tma_inner_size=(64 if n_tile == 64 else None),
                ))

                m_ctas = (shape.m + m_tile - 1) // m_tile
                n_ctas = (shape.n + n_tile - 1) // n_tile
                for axis, extent in (("m", m_ctas), ("n", n_ctas)):
                    for cluster in (2, 4, 8):
                        if extent % cluster:
                            continue
                        configs.append(_config(
                            f"archdeep_cluster_{axis}{cluster}_m{m_tile}_"
                            f"n{n_tile}_k64_s{stages}",
                            output_m_tile=m_tile, output_n_tile=n_tile,
                            warpgroup_n_tile=n_tile, operand_k_tile=64,
                            operand_stages=stages,
                            cluster_m_ctas=(cluster if axis == "m" else 1),
                            cluster_n_ctas=(cluster if axis == "n" else 1),
                            multicast=True,
                            output_tma_inner_size=(
                                64 if n_tile == 64 else None
                            ),
                        ))

                # Keep every warpgroup computational while testing whether
                # more independent WGMMA issuers plus multicast can hide the
                # deep-reduction pipeline latency seen in NCU.
                for wg_groups in (2, 3, 4):
                    wg_n, remainder = divmod(n_tile, wg_groups)
                    if remainder or wg_n < 16 or wg_n % 8:
                        continue
                    configs.append(_config(
                        f"archdeep_parallel{wg_groups}_m{m_tile}_n{n_tile}_"
                        f"k64_s{stages}",
                        output_m_tile=m_tile, output_n_tile=n_tile,
                        warpgroup_n_tile=wg_n, operand_k_tile=64,
                        operand_stages=stages, output_tma_inner_size=None,
                    ))
                    for axis, extent in (("m", m_ctas), ("n", n_ctas)):
                        for cluster in (2, 4, 8):
                            if extent % cluster:
                                continue
                            configs.append(_config(
                                f"archdeep_parallel{wg_groups}_cluster_"
                                f"{axis}{cluster}_m{m_tile}_n{n_tile}_"
                                f"k64_s{stages}",
                                output_m_tile=m_tile,
                                output_n_tile=n_tile,
                                warpgroup_n_tile=wg_n,
                                operand_k_tile=64,
                                operand_stages=stages,
                                cluster_m_ctas=(
                                    cluster if axis == "m" else 1
                                ),
                                cluster_n_ctas=(
                                    cluster if axis == "n" else 1
                                ),
                                multicast=True,
                                output_tma_inner_size=None,
                            ))

    if shape.m >= 8192 and shape.k <= 1024:
        for m_tile, n_tile in (
            (192, 128), (192, 160), (192, 192),
            (256, 128), (256, 160), (256, 192),
        ):
            if shape.n < n_tile:
                continue
            m_ctas = (shape.m + m_tile - 1) // m_tile
            n_ctas = (shape.n + n_tile - 1) // n_tile
            for stages in (2, 3, 4):
                operand_smem = 2 * (m_tile + n_tile) * 64 * stages
                if operand_smem > 225 * 1024:
                    continue
                for axis, extent in (("none", 1), ("m", m_ctas), ("n", n_ctas)):
                    if axis != "none" and extent % 2:
                        continue
                    configs.append(_config(
                        f"archtall_{axis}2_m{m_tile}_n{n_tile}_k64_s{stages}",
                        output_m_tile=m_tile, output_n_tile=n_tile,
                        warpgroup_n_tile=n_tile, operand_k_tile=64,
                        operand_stages=stages, output_slices=2,
                        cluster_m_ctas=(2 if axis == "m" else 1),
                        cluster_n_ctas=(2 if axis == "n" else 1),
                        multicast=axis != "none",
                        output_tma_inner_size=(128 if n_tile == 128 else None),
                        unsafe_reuse_smem=True,
                    ))
    return tuple(configs)


def narrow_wg2d_configs(shape: Shape) -> tuple[ScheduleConfig, ...]:
    """Use homogeneous M workgroups to reuse narrow-N operand tiles."""
    configs: list[ScheduleConfig] = []
    for m_tile in (128, 192, 256):
        if shape.m < m_tile:
            continue
        for n_tile in (48, 56, 64, 72, 80, 96):
            if shape.n < n_tile:
                continue
            specs = [(64, stage) for stage in (3, 4, 5, 6)]
            if shape.k % 128 == 0:
                specs.extend((128, stage) for stage in (2, 3, 4))
            for k_tile, stages in specs:
                estimated_smem = (
                    2 * (m_tile + n_tile) * k_tile * stages
                    + 2 * m_tile * n_tile
                )
                if estimated_smem > 220 * 1024:
                    continue
                for fast in ("m", "n"):
                    configs.append(_config(
                        f"wg2dn_m{m_tile}_n{n_tile}_wm64_"
                        f"k{k_tile}_s{stages}_{fast}fast",
                        output_m_tile=m_tile, output_n_tile=n_tile,
                        warpgroup_m_tile=64, warpgroup_n_tile=n_tile,
                        warpgroup_fast_axis=fast,
                        operand_k_tile=k_tile, operand_stages=stages,
                        output_tma_inner_size=(
                            64 if n_tile == 64 else None
                        ),
                    ))
    return tuple(configs)
