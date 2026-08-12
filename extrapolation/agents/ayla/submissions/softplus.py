"""Ayla fused GEMM + stable Softplus submission."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _softplus_common import prepare_config


NAME = "ayla_fused_softplus_i4"
PROBLEM = "softplus"
METADATA = {
    "backend": "Ayla PTX",
    "fused": True,
    "stable_softplus": True,
    "softplus_math": "predicated ex2.approx + lg2.approx in FP32",
    "softplus_interleave": 4,
    "intermediate_hbm_bytes": 0,
    "output_tiles_per_cta": 4,
    "operand_stages": 3,
    "output_slices": 2,
    "output_tma_inner_padding": 16,
    "output_store_buffers": 2,
}


def prepare(inputs):
    return prepare_config(inputs, interleave=4)
