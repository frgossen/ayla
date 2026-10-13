import pytest

from benchmarks.benchmark_oss_gemm_seeds import main


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_show_only_selects_the_requested_dtype_without_using_a_gpu(
    monkeypatch, capsys, dtype,
):
    monkeypatch.setattr(
        "sys.argv", ["seed", "128", "4096", "4096", "--dtype", dtype, "--show-only"],
    )

    assert main() == 0
    output = capsys.readouterr().out
    assert f"dtype: {dtype}" in output
    assert "leaf: exact:" in output
    if dtype == "bfloat16":
        assert "heuristic_bf16_exact_winner" in output
    else:
        assert "heuristic_bf16_exact_winner" not in output
