"""Emit the PTX for the submitted Dual GEMM schedule."""

from pathlib import Path
import sys


AYLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AYLA_ROOT))

from ayla.gemm_collection.dual_gemm_4096 import dual_gemm_4096_module
from ayla.ptx_codegen import make_ptx_kernel


def main() -> None:
    kernel = make_ptx_kernel(dual_gemm_4096_module(
        first_stages=4,
        second_k_tile=128,
        output_slices=2,
        output_padding=16,
        output_store_buffers=2,
    ))
    output = AYLA_ROOT / "results" / "dual_gemm_final.ptx"
    output.write_text(kernel.ptx)
    print(f"wrote {output} ({len(kernel.ptx)} bytes)")
    print(
        f"kernel={kernel.name} grid={kernel.grid_dim} "
        f"block={kernel.block_dim}"
    )


if __name__ == "__main__":
    main()
