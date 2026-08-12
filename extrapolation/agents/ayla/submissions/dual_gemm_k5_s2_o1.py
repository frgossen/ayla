from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submissions._dual_common import prepare_config

NAME = "ayla_dual_k5_second2_obuf1"
PROBLEM = "dual_gemm"
METADATA = {"backend": "Ayla PTX", "stages": 5, "second_stages": 2, "obuf": 1}


def prepare(inputs):
    return prepare_config(
        inputs, first_stages=5, second_stages=2, output_store_buffers=1,
    )
