from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submissions._dual_common import prepare_config

NAME = "ayla_dual_k3_part1_obuf2"
PROBLEM = "dual_gemm"
METADATA = {"backend": "Ayla PTX", "stages": 3, "partitions": 1, "obuf": 2}


def prepare(inputs):
    return prepare_config(inputs, columns_per_cta=4096)
