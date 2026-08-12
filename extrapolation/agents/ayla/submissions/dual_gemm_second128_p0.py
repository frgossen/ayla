from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submissions._dual_common import prepare_config

NAME = "ayla_dual_k4_second128_p0"
PROBLEM = "dual_gemm"
METADATA = {"backend": "Ayla PTX", "stages": 4, "second_k_tile": 128, "padding": 0}


def prepare(inputs):
    return prepare_config(
        inputs, first_stages=4, second_k_tile=128, output_padding=0,
    )
