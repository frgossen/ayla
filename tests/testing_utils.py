"""Helpers shared across the test suite.

Deliberately not part of the ayla package: `pyproject.toml` ships `ayla*` and
excludes `tests*`, so anything pytest-flavoured belongs here rather than in
the wheel. Fixtures and hooks go in `conftest.py` instead -- importing from a
conftest is not something pytest supports.
"""
import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"
)


def assert_contains_in_order(text: str, *fragments: str) -> None:
    """Assert that `text` contains each fragment's lines in order.

    Each fragment is a multi-line string (typically a dedented raw string).
    Lines are stripped and matched in sequence — arbitrary lines may appear
    between fragments and between lines within a fragment.
    """
    lines = [line.strip() for line in text.splitlines()]
    pos = 0
    for fragment in fragments:
        for expected_line in fragment.strip().splitlines():
            expected_line = expected_line.strip()
            if not expected_line:
                continue
            found = False
            while pos < len(lines):
                if lines[pos] == expected_line:
                    found = True
                    pos += 1
                    break
                pos += 1
            assert found, (
                f"expected line not found: {expected_line!r}\n"
                f"searched from position {pos} in:\n{text}"
            )


def assert_dominance(block, incoming=frozenset()):
    defined = set(incoming) | set(block.arguments)
    for op in block.ops:
        assert set(op.operands) <= defined, op.op_id
        for child in op.blocks:
            assert_dominance(child, defined)
        defined.update(op.results)


DTYPES = ("f16", "bf16")
TORCH_DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
# Error bounds against cuBLAS for reductions too long to match it bitwise.
LARGE_K = dict(atol={"f16": 0.125, "bf16": 1.0}, rms=0.002)
LOOSE = dict(atol={"f16": 4, "bf16": 32}, rms=0.002)


def _relative_rms(actual, expected):
    delta = actual.float() - expected.float()
    return (delta.square().mean() / expected.float().square().mean()).sqrt().item()


def validate(compute, a, b, c, storage, *, replays=6, rms=None, atol=None):
    """Poison output and change both operands between captured graph replays.

    `storage` is `c`'s guard-banded allocation (32 sentinel elements of 123
    on either side). Bounds default to relative RMS 0.002 (f16) / 0.01 (bf16)
    and max absolute error 4 (f16) / 16 (bf16).
    """
    half = c.dtype == torch.float16
    rms = (0.002 if half else 0.01) if rms is None else rms
    atol = (4 if half else 16) if atol is None else atol
    graph = torch.cuda.CUDAGraph()
    compute(a, b)
    with torch.cuda.graph(graph):
        compute(a, b)
    checks = []
    for seed in range(replays):
        torch.manual_seed(20261110 + seed)
        a.normal_()
        b.normal_()
        c.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = (a.float() @ b.float().T).to(c.dtype)
        error = _relative_rms(c, expected)
        maximum = (c.float() - expected.float()).abs().max().item()
        assert torch.isfinite(c).all(), "missing or nonfinite output"
        assert error < rms and maximum <= atol, (error, maximum)
        assert torch.all(storage[:32] == 123) and torch.all(storage[-32:] == 123), (
            "output guard overwritten"
        )
        checks.append(dict(seed=seed, relative_rms=error, max_absolute_error=maximum))
    return checks


def check_gemm(launch, m, n, k, dtype="bf16", *, atol=None, rtol=0.0, rms=None,
               runs=None, seed=0, mutate=False, graph=False, bias=0.0,
               out_dtype=None, transpose_out=False):
    """Check that `launch(a, bt, c)` writes exactly `c = a @ bt.T (+ bias)`.

    `c` is NaN-poisoned before every launch and sits in guard-banded storage
    that must stay untouched. By default the output must match cuBLAS
    bitwise; `atol` (a number or a per-dtype dict; `math.inf` disables it),
    `rtol` and `rms` (relative RMS error) relax that. `runs` launches (5) are
    checked; `mutate` changes the operands before each. `graph` instead runs
    :func:`validate` for `runs` replays (6), with its default bounds unless
    `atol`/`rms` are given. `out_dtype="f32"` compares with a float64
    reference; `transpose_out` makes `c` column-major.
    """
    typ = TORCH_DTYPES[dtype]
    out = TORCH_DTYPES[out_dtype or dtype]
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=typ)
    bt = torch.randn(n, k, device="cuda", dtype=typ)
    storage = torch.full((m * n + 64,), 123, device="cuda", dtype=out)
    c = storage[32:-32].view(n, m).t() if transpose_out else storage[32:-32].view(m, n)
    atol = atol.get(dtype) if isinstance(atol, dict) else atol
    if graph:
        compute = lambda a, b: (launch(a, b, c), c)[1]  # noqa: E731
        return validate(compute, a, bt, c, storage, replays=runs or 6, rms=rms, atol=atol)
    for _ in range(runs or 5):
        if mutate:
            a.mul_(-0.75)
            bt.mul_(0.875)
        c.fill_(float("nan"))
        launch(a, bt, c)
        torch.cuda.synchronize()
        if out == typ and not bias:
            expected = a @ bt.t()
        else:
            expected = (a.double() @ bt.double().t() + bias).to(out)
        assert torch.isfinite(c).all(), "missing or nonfinite output"
        torch.testing.assert_close(
            c.float(), expected.float(), atol=atol or 0.0, rtol=rtol
        )
        if rms is not None:
            assert _relative_rms(c, expected) <= rms
        assert torch.all(storage[:32] == 123) and torch.all(storage[-32:] == 123), (
            "output guard overwritten"
        )
