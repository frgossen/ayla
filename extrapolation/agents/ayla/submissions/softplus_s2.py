from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _softplus_common import prepare_config
NAME = "ayla_fused_softplus_s2"
PROBLEM = "softplus"
METADATA = {"backend": "Ayla PTX", "fused": True, "output_tiles_per_cta": 4, "operand_stages": 2, "output_store_buffers": 2}
def prepare(inputs):
    return prepare_config(inputs, interleave=4, operand_stages=2)
