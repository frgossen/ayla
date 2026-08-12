from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _softplus_common import prepare_config
NAME = "ayla_fused_softplus_i1"
PROBLEM = "softplus"
METADATA = {"backend": "Ayla PTX", "fused": True, "softplus_interleave": 1}
def prepare(inputs):
    return prepare_config(inputs, interleave=1)
