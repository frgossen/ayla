"""A small collection of pre-built GEMM kernels.

Each kernel lives in its own module here: a ``*_module()`` builder plus
self-registration as a ``torch.ops.ayla.*`` overload at import time. Importing
this package imports them all, so the ops are ready to use:

    import ayla.gemm_collection  # registers all ops
    torch.ops.ayla.naive_matmul_4096_loop(a, b)  # returns c
"""

from . import (  # noqa: F401  (each import registers its op)
    auto_matmul_4096,
    auto_matmul_4096_wgmma256_wg,
    auto_matmul_4096_wgmma256_wg_tma_store_k3,
    fastcu_5_1,
    loop_4x4_matmul_4096,
    loop_8x8_matmul_4096,
    naive_matmul_4096_loop,
    pranjalssh_4096,
    pranjalssh_4096_pipelined,
    tma_matmul_4096,
)
