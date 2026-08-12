from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _softplus_common import prepare_config
NAME = "ayla_fused_softplus_s1_obuf1"
PROBLEM = "softplus"
METADATA = {"backend": "Ayla PTX", "fused": True, "output_tiles_per_cta": 4, "operand_stages": 1, "output_store_buffers": 1}
def prepare(inputs):
    return prepare_config(inputs, interleave=4, operand_stages=1, output_store_buffers=1)
