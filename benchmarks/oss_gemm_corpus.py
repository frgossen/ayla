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


from dataclasses import asdict, dataclass


from pathlib import Path


from typing import Any, Iterable


CORPUS = Path(__file__).resolve().parent / "oss_pytorch_gemm_shapes.json"


OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".benchmarks" / "oss_gemm"


VALIDATION_CSV = OUTPUT_DIR / "validation.csv"


VALIDATION_JSON = OUTPUT_DIR / "validation.json"


RAW_CSV = OUTPUT_DIR / "raw.csv"


RAW_JSON = OUTPUT_DIR / "raw.json"














SUPPORTED_DTYPES = frozenset({"float16", "bfloat16"})


MIN_GEMM_TILE = 64


SPLIT_K_REDUCTION_PRESSURE_THRESHOLD = 32


MIN_OUTPUT_TILES_WITHOUT_SPLIT_K = 128


NEAR_FULL_OUTPUT_TILES_WITHOUT_SPLIT_K = 256


NEAR_FULL_REDUCTION_PRESSURE_THRESHOLD = 2


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
    if shape.op not in ("mm", "addmm", "bmm"):
        reasons.append(("op", f"{shape.op} semantics are not implemented"))
    if shape.op == "bmm" and (shape.batch is None or shape.batch <= 1):
        reasons.append((
            "batch",
            "canonical BMM evaluation requires a batch dimension greater than 1",
        ))
    elif shape.op != "bmm" and shape.batch is not None:
        reasons.append(("batching", "unexpected batch dimension"))

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
        if shape.op == "bmm" and shape.batch is not None:
            output_tiles *= shape.batch
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
        near_full_deep_reduction = (
            output_tiles < NEAR_FULL_OUTPUT_TILES_WITHOUT_SPLIT_K
            and reduction_tiles
            >= NEAR_FULL_REDUCTION_PRESSURE_THRESHOLD * output_tiles
        )
        needs_split_k = (
            high_reduction_pressure
            or underfilled_deep_reduction
            or near_full_deep_reduction
        )
        if shape.op in ("mm", "addmm", "bmm") and needs_split_k:
            reasons.append((
                "split_k",
                "temporarily skipped: reduction pressure is "
                f"{reduction_tiles / output_tiles:.1f} K tiles per output tile "
                f"across {output_tiles} output tiles; an "
                "efficient kernel requires split-K/Stream-K",
            ))
    if (
        shape.op in ("mm", "addmm", "bmm")
        and positive_dimensions
        and shape.m < MIN_GEMM_TILE
    ):
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
    if shape.op == "addmm" and (
        len(shape.args) != 3
        or shape.args[0].shape != (shape.n,)
        or shape.args[1].shape != (shape.m, shape.k)
        or shape.args[2].shape != (shape.k, shape.n)
    ):
        reasons.append((
            "shape",
            "addmm requires a vector-N bias and operands agreeing with M/N/K",
        ))
    if shape.op == "bmm" and (
        shape.batch is None
        or len(shape.args) != 2
        or shape.args[0].shape != (shape.batch, shape.m, shape.k)
        or shape.args[1].shape != (shape.batch, shape.k, shape.n)
    ):
        reasons.append((
            "shape",
            "bmm operands must have shapes [B,M,K] and [B,K,N]",
        ))
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
    counts: Counter[tuple[str, int | None, int, int, int, str]] = Counter()
    suites: defaultdict[
        tuple[str, int | None, int, int, int, str], set[str]
    ] = defaultdict(set)
    records: Counter[
        tuple[str, int | None, int, int, int, str]
    ] = Counter()
    for shape in accepted:
        key = (
            shape.op, shape.batch, shape.m, shape.n, shape.k, shape.dtype,
        )
        counts[key] += shape.count
        suites[key].update(shape.suites)
        records[key] += shape.source_records

    result = []
    for (op, batch, m, n, k, dtype), count in counts.items():
        matrix_args = (
            (
                TensorSpec(
                    shape=(batch, m, k), stride=(m * k, k, 1), dtype=dtype,
                ),
                TensorSpec(
                    shape=(batch, n, k), stride=(n * k, k, 1), dtype=dtype,
                ),
            )
            if op == "bmm" and batch is not None
            else (
                TensorSpec(shape=(m, k), stride=(k, 1), dtype=dtype),
                TensorSpec(shape=(k, n), stride=(1, k), dtype=dtype),
            )
        )
        result.append(Shape(
            op=op, batch=batch, m=m, n=n, k=k,
            args=(
                (TensorSpec(shape=(n,), stride=(1,), dtype=dtype),)
                if op == "addmm" else ()
            ) + matrix_args,
            count=count,
            suites=tuple(sorted(suites[op, batch, m, n, k, dtype])),
            source_records=records[op, batch, m, n, k, dtype],
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
        writer = csv.DictWriter(output_file, fieldnames=list(dict.fromkeys(key for row in flattened for key in row)))
        writer.writeheader()
        writer.writerows(flattened)
    temporary.replace(output)
