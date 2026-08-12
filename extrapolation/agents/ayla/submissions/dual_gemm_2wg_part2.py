from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submissions._dual_common import prepare_config

NAME = "ayla_dual_2wg_part2_active0"
PROBLEM = "dual_gemm"
METADATA = {
    "backend": "Ayla PTX", "warp_groups": 2, "partitions": 2,
    "gemm1_active_warpgroup": 0, "stages": 4, "second_stages": 1,
    "obuf": 1,
}


def prepare(inputs):
    return prepare_config(
        inputs, columns_per_cta=2048, warp_groups=2, first_stages=4,
        second_stages=1, output_store_buffers=1,
    )
